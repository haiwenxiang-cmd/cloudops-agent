"""Transactional service for the durable mutation execution journal."""

import uuid
from datetime import timedelta
from typing import Any, Dict, Optional

from tortoise import timezone
from tortoise.exceptions import IntegrityError
from tortoise.expressions import F, Q
from tortoise.transactions import in_transaction

from ..config import settings
from ..models.mutation import (
    ApprovalStatus,
    ExecutionStatus,
    MutationOperation,
    PolicyDecision,
    VerificationStatus,
)
from .mutation_policy import MutationPolicyDecision
from .mutation_utils import hmac_fingerprint, redact_sensitive, redact_tool_args


class InvalidMutationTransition(RuntimeError):
    pass


def verification_budget_exhausted(operation: MutationOperation, now) -> bool:
    verification_age = (now - operation.created_at).total_seconds()
    return bool(
        operation.verification_attempt_count
        >= settings.MUTATION_MAX_VERIFICATION_ATTEMPTS
        or verification_age >= settings.MUTATION_MAX_VERIFICATION_AGE_SECONDS
    )


class MutationJournalService:
    """Owns state transitions; callers may not update journal rows directly."""

    async def prepare(
        self,
        *,
        operation_id: str,
        run_id: str,
        call_id: str,
        tool_name: str,
        args: Dict[str, Any],
        context: Dict[str, Any],
        policy: MutationPolicyDecision,
        verification_plan: Dict[str, Any],
    ) -> tuple[MutationOperation, bool]:
        op_uuid = uuid.UUID(str(operation_id))
        idempotency_key = str(
            context.get("idempotency_key")
            or hmac_fingerprint(
                settings.INTERNAL_API_KEY,
                context.get("conversation_id"),
                call_id,
                tool_name,
                args,
            )
        )
        args_hmac = hmac_fingerprint(settings.INTERNAL_API_KEY, tool_name, args)
        existing = await MutationOperation.filter(
            Q(id=op_uuid) | Q(idempotency_key=idempotency_key)
        ).first()
        if existing:
            if existing.tool_name != tool_name or existing.args_hmac != args_hmac:
                raise InvalidMutationTransition(
                    "Idempotency key collision with different mutation intent"
                )
            return existing, False

        redacted_args = redact_tool_args(tool_name, args)
        try:
            operation = await MutationOperation.create(
                id=op_uuid,
                idempotency_key=idempotency_key,
                workspace_id=context.get("workspace_id"),
                conversation_id=context.get("conversation_id"),
                run_id=run_id,
                call_id=call_id,
                user_id=context.get("user_id"),
                user_role=context.get("user_role"),
                tool_name=tool_name,
                tool_category=tool_name.split("_", 1)[0],
                target_identity={"targets": policy.targets},
                redacted_args=redacted_args,
                args_hmac=args_hmac,
                desired_state_hmac=args_hmac,
                policy_decision=PolicyDecision.ALLOW,
                policy_reason=policy.reason,
                approval_status=ApprovalStatus.APPROVED,
                approved_by_user_id=context.get("user_id"),
                approved_at=timezone.now(),
                approval_reason=(context.get("approval_reasons") or {}).get(call_id),
                execution_status=ExecutionStatus.PREPARED,
                verification_status=(
                    VerificationStatus.PENDING
                    if verification_plan.get("supported")
                    else VerificationStatus.NOT_REQUIRED
                ),
                verification_plan=verification_plan,
            )
        except IntegrityError as integrity_error:
            operation = await MutationOperation.get(idempotency_key=idempotency_key)
            if operation.tool_name != tool_name or operation.args_hmac != args_hmac:
                raise InvalidMutationTransition(
                    "Idempotency key collision with different mutation intent"
                ) from integrity_error
            return operation, False
        return operation, True

    async def mark_executing(self, operation_id: str, lease_owner: str) -> MutationOperation:
        async with in_transaction():
            operation = await MutationOperation.filter(id=operation_id).select_for_update().get()
            if operation.execution_status != ExecutionStatus.PREPARED:
                raise InvalidMutationTransition(
                    f"Cannot execute operation {operation_id} from "
                    f"{operation.execution_status.value}"
                )
            operation.execution_status = ExecutionStatus.EXECUTING
            operation.attempt_count += 1
            operation.started_at = operation.started_at or timezone.now()
            operation.lease_owner = lease_owner
            operation.lease_expires_at = timezone.now() + timedelta(
                seconds=settings.MUTATION_LEASE_SECONDS
            )
            operation.version += 1
            await operation.save()
            return operation

    async def record_execution(
        self,
        operation_id: str,
        *,
        status: ExecutionStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> MutationOperation:
        if status not in {
            ExecutionStatus.REPORTED_SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.UNKNOWN,
            ExecutionStatus.CANCELLED,
        }:
            raise InvalidMutationTransition(f"Invalid execution outcome: {status.value}")
        async with in_transaction():
            operation = await MutationOperation.filter(id=operation_id).select_for_update().get()
            if operation.execution_status != ExecutionStatus.EXECUTING:
                raise InvalidMutationTransition(
                    f"Cannot record outcome for {operation_id} from "
                    f"{operation.execution_status.value}"
                )
            operation.execution_status = status
            operation.execution_result = redact_sensitive(result or {})
            operation.execution_error = redact_sensitive(error or {})
            operation.lease_owner = None
            operation.lease_expires_at = None
            operation.finished_at = timezone.now()
            if status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                operation.verification_status = VerificationStatus.NOT_REQUIRED
                operation.verification_lease_owner = None
                operation.verification_lease_expires_at = None
            operation.version += 1
            await operation.save()
            return operation

    async def get(self, operation_id: str) -> Optional[MutationOperation]:
        return await MutationOperation.get_or_none(id=operation_id)

    async def claim_verification(
        self, operation_id: str, *, lease_owner: str
    ) -> tuple[MutationOperation, bool]:
        """Claim one verifier lease without regressing a terminal result."""
        now = timezone.now()
        async with in_transaction():
            operation = await MutationOperation.filter(id=operation_id).select_for_update().get()
            if operation.execution_status not in {
                ExecutionStatus.REPORTED_SUCCESS,
                ExecutionStatus.UNKNOWN,
            }:
                raise InvalidMutationTransition(
                    f"Cannot verify operation {operation_id} from "
                    f"{operation.execution_status.value}"
                )
            if operation.verification_status in {
                VerificationStatus.PASSED,
                VerificationStatus.FAILED,
                VerificationStatus.NOT_REQUIRED,
                VerificationStatus.NEEDS_REVIEW,
            }:
                return operation, False

            verification_age = (now - operation.created_at).total_seconds()
            if verification_budget_exhausted(operation, now):
                operation.verification_status = VerificationStatus.NEEDS_REVIEW
                operation.verification_result = {
                    "reason": "verification_budget_exhausted",
                    "verification_attempt_count": operation.verification_attempt_count,
                    "age_seconds": int(verification_age),
                    "manual_review_required": True,
                }
                operation.verification_lease_owner = None
                operation.verification_lease_expires_at = None
                operation.version += 1
                await operation.save()
                return operation, False

            lease_active = bool(
                operation.verification_status == VerificationStatus.VERIFYING
                and operation.verification_lease_expires_at
                and operation.verification_lease_expires_at > now
            )
            if lease_active and operation.verification_lease_owner != lease_owner:
                return operation, False

            operation.verification_status = VerificationStatus.VERIFYING
            operation.verification_attempt_count += 1
            operation.verification_lease_owner = lease_owner
            verification_lease_seconds = max(
                settings.MUTATION_VERIFICATION_LEASE_SECONDS,
                int(
                    settings.MUTATION_VERIFICATION_TIMEOUT_SECONDS
                    + settings.MUTATION_VERIFICATION_INTERVAL_SECONDS
                    + 5
                ),
            )
            operation.verification_lease_expires_at = now + timedelta(
                seconds=verification_lease_seconds
            )
            operation.version += 1
            await operation.save()
            return operation, True

    async def record_verification(
        self,
        operation_id: str,
        *,
        lease_owner: str,
        status: VerificationStatus,
        result: Dict[str, Any],
        external_reference: Optional[Dict[str, Any]] = None,
    ) -> MutationOperation:
        if status not in {
            VerificationStatus.PASSED,
            VerificationStatus.FAILED,
            VerificationStatus.INCONCLUSIVE,
        }:
            raise InvalidMutationTransition(f"Invalid verification outcome: {status.value}")
        async with in_transaction():
            operation = await MutationOperation.filter(id=operation_id).select_for_update().get()
            if operation.verification_status != VerificationStatus.VERIFYING:
                raise InvalidMutationTransition(
                    f"Cannot record verification for {operation_id} from "
                    f"{operation.verification_status.value}"
                )
            if operation.verification_lease_owner != lease_owner:
                raise InvalidMutationTransition(
                    f"Verification lease for {operation_id} is owned by another worker"
                )
            operation.verification_status = status
            operation.verification_result = redact_sensitive(result)
            if external_reference:
                operation.external_reference = redact_sensitive(external_reference)
            operation.verification_lease_owner = None
            operation.verification_lease_expires_at = None
            operation.version += 1
            await operation.save()
            return operation

    async def abandon_verification(
        self,
        operation_id: str,
        *,
        lease_owner: str,
        reason: str,
    ) -> bool:
        """Release a verifier lease after cooperative cancellation."""
        async with in_transaction():
            operation = await MutationOperation.filter(id=operation_id).select_for_update().get()
            if (
                operation.verification_status != VerificationStatus.VERIFYING
                or operation.verification_lease_owner != lease_owner
            ):
                return False
            operation.verification_status = VerificationStatus.INCONCLUSIVE
            operation.verification_result = redact_sensitive({"reason": reason})
            operation.verification_lease_owner = None
            operation.verification_lease_expires_at = None
            operation.version += 1
            await operation.save()
            return True

    async def recover_stale_executions(self) -> int:
        """Mark abandoned executing rows unknown; verification is handled by the graph/worker."""
        now = timezone.now()
        return await MutationOperation.filter(
            Q(execution_status=ExecutionStatus.EXECUTING)
            & (Q(lease_expires_at__lt=now) | Q(lease_expires_at__isnull=True))
        ).update(
            execution_status=ExecutionStatus.UNKNOWN,
            verification_status=VerificationStatus.PENDING,
            lease_owner=None,
            lease_expires_at=None,
            verification_lease_owner=None,
            verification_lease_expires_at=None,
            version=F("version") + 1,
            updated_at=now,
        )

    async def recover_stale_verifications(self) -> int:
        """Release verifier leases abandoned by a crash or hard cancellation."""
        now = timezone.now()
        return await MutationOperation.filter(
            Q(verification_status=VerificationStatus.VERIFYING)
            & (
                Q(verification_lease_expires_at__lt=now)
                | Q(verification_lease_expires_at__isnull=True)
            )
        ).update(
            verification_status=VerificationStatus.INCONCLUSIVE,
            verification_result={"reason": "verification_lease_expired"},
            verification_lease_owner=None,
            verification_lease_expires_at=None,
            version=F("version") + 1,
            updated_at=now,
        )

    async def converge_stopped_run(self, run_id: str) -> Dict[str, int]:
        """Make local journal states recoverable after a workflow Stop."""
        now = timezone.now()
        cancelled = await MutationOperation.filter(
            run_id=run_id,
            execution_status=ExecutionStatus.PREPARED,
        ).update(
            execution_status=ExecutionStatus.CANCELLED,
            verification_status=VerificationStatus.NOT_REQUIRED,
            execution_error={"error_type": "cancelled_before_execution"},
            finished_at=now,
            lease_owner=None,
            lease_expires_at=None,
            verification_lease_owner=None,
            verification_lease_expires_at=None,
            version=F("version") + 1,
            updated_at=now,
        )
        unknown = await MutationOperation.filter(
            run_id=run_id,
            execution_status=ExecutionStatus.EXECUTING,
        ).update(
            execution_status=ExecutionStatus.UNKNOWN,
            verification_status=VerificationStatus.PENDING,
            execution_error={"error_type": "cancelled", "ambiguous_outcome": True},
            finished_at=now,
            lease_owner=None,
            lease_expires_at=None,
            verification_lease_owner=None,
            verification_lease_expires_at=None,
            version=F("version") + 1,
            updated_at=now,
        )
        return {"cancelled": cancelled, "unknown": unknown}

    @staticmethod
    def public_summary(operation: MutationOperation) -> Dict[str, Any]:
        return {
            "operation_id": str(operation.id),
            "tool": operation.tool_name,
            "execution_status": operation.execution_status.value,
            "verification_status": operation.verification_status.value,
            "attempt_count": operation.attempt_count,
            "verification_attempt_count": operation.verification_attempt_count,
        }
