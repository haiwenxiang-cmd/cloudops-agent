"""Durable journal models for infrastructure mutations."""

import uuid
from enum import Enum

from tortoise import fields
from tortoise.models import Model


class ExecutionStatus(str, Enum):
    PREPARED = "prepared"
    EXECUTING = "executing"
    REPORTED_SUCCESS = "reported_success"
    FAILED = "failed"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


class VerificationStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    VERIFYING = "verifying"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    NEEDS_REVIEW = "needs_review"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class MutationOperation(Model):
    """Write-ahead record for one externally visible mutation attempt."""

    id = fields.UUIDField(pk=True, default=uuid.uuid4)
    idempotency_key = fields.CharField(max_length=160, unique=True)
    workspace_id = fields.UUIDField(null=True, index=True)
    conversation_id = fields.CharField(max_length=160, null=True, index=True)
    run_id = fields.CharField(max_length=160, null=True, index=True)
    call_id = fields.CharField(max_length=160, null=True, index=True)
    user_id = fields.UUIDField(null=True, index=True)
    user_role = fields.CharField(max_length=32, null=True)

    tool_name = fields.CharField(max_length=160, index=True)
    tool_category = fields.CharField(max_length=40)
    target_identity = fields.JSONField(default=dict)
    redacted_args = fields.JSONField(default=dict)
    args_hmac = fields.CharField(max_length=64)
    desired_state_hmac = fields.CharField(max_length=64, null=True)

    policy_decision = fields.CharEnumField(PolicyDecision, max_length=20)
    policy_reason = fields.TextField(default="")
    approval_status = fields.CharEnumField(
        ApprovalStatus, max_length=20, default=ApprovalStatus.PENDING
    )
    approved_by_user_id = fields.UUIDField(null=True)
    approved_at = fields.DatetimeField(null=True)
    approval_reason = fields.TextField(null=True)

    execution_status = fields.CharEnumField(
        ExecutionStatus, max_length=32, default=ExecutionStatus.PREPARED
    )
    verification_status = fields.CharEnumField(
        VerificationStatus, max_length=32, default=VerificationStatus.PENDING
    )
    attempt_count = fields.IntField(default=0)
    verification_attempt_count = fields.IntField(default=0)

    external_reference = fields.JSONField(default=dict)
    execution_result = fields.JSONField(default=dict)
    execution_error = fields.JSONField(default=dict)
    verification_plan = fields.JSONField(default=dict)
    verification_result = fields.JSONField(default=dict)

    lease_owner = fields.CharField(max_length=160, null=True)
    lease_expires_at = fields.DatetimeField(null=True, index=True)
    verification_lease_owner = fields.CharField(max_length=160, null=True)
    verification_lease_expires_at = fields.DatetimeField(null=True, index=True)
    started_at = fields.DatetimeField(null=True)
    finished_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    version = fields.IntField(default=1)

    class Meta:
        table = "mutation_operations"
        indexes = (
            ("execution_status", "lease_expires_at"),
            ("verification_status", "updated_at"),
            ("verification_status", "verification_lease_expires_at"),
            ("conversation_id", "created_at"),
        )
