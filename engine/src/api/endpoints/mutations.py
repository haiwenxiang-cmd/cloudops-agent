"""Administrative, read-only access to the durable mutation journal."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..models.mutation import MutationOperation
from ..models.user import User
from ..services.auth import verify_admin_role

router = APIRouter()


def _iso(value):
    return value.isoformat() if value else None


def _summary(operation: MutationOperation) -> dict:
    return {
        "operation_id": str(operation.id),
        "conversation_id": operation.conversation_id,
        "run_id": operation.run_id,
        "call_id": operation.call_id,
        "tool": operation.tool_name,
        "target_identity": operation.target_identity,
        "policy_decision": operation.policy_decision.value,
        "policy_reason": operation.policy_reason,
        "approval_status": operation.approval_status.value,
        "approval_reason": operation.approval_reason,
        "execution_status": operation.execution_status.value,
        "verification_status": operation.verification_status.value,
        "attempt_count": operation.attempt_count,
        "verification_attempt_count": operation.verification_attempt_count,
        "created_at": _iso(operation.created_at),
        "updated_at": _iso(operation.updated_at),
    }


@router.get("")
async def list_mutation_operations(
    execution_status: Optional[str] = None,
    verification_status: Optional[str] = None,
    conversation_id: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(verify_admin_role),
):
    query = MutationOperation.all()
    if execution_status:
        query = query.filter(execution_status=execution_status)
    if verification_status:
        query = query.filter(verification_status=verification_status)
    if conversation_id:
        query = query.filter(conversation_id=conversation_id)
    operations = await query.order_by("-created_at").limit(limit)
    return {"items": [_summary(operation) for operation in operations]}


@router.get("/{operation_id}")
async def get_mutation_operation(
    operation_id: str,
    _: User = Depends(verify_admin_role),
):
    try:
        parsed_operation_id = uuid.UUID(operation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid operation ID") from exc
    operation = await MutationOperation.get_or_none(id=parsed_operation_id)
    if operation is None:
        raise HTTPException(status_code=404, detail="Mutation operation not found")
    payload = _summary(operation)
    payload.update(
        {
            "redacted_args": operation.redacted_args,
            "args_hmac": operation.args_hmac,
            "desired_state_hmac": operation.desired_state_hmac,
            "external_reference": operation.external_reference,
            "execution_error": operation.execution_error,
            "verification_plan": operation.verification_plan,
            "verification_result": operation.verification_result,
            "started_at": _iso(operation.started_at),
            "finished_at": _iso(operation.finished_at),
            "lease_expires_at": _iso(operation.lease_expires_at),
            "verification_lease_owner": operation.verification_lease_owner,
            "verification_lease_expires_at": _iso(
                operation.verification_lease_expires_at
            ),
            "version": operation.version,
        }
    )
    return payload
