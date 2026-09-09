import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tortoise import timezone

from src.api.endpoints import agent as agent_endpoint
from src.api.models.conversation import Conversation
from src.api.models.mutation import ExecutionStatus, MutationOperation, VerificationStatus
from src.api.services import mutation_journal as journal_module
from src.api.services.mutation_journal import MutationJournalService
from src.api.services.mutation_verifier import MutationRecoveryWorker, MutationVerifier


class FakePubSub:
    async def subscribe(self, channel):
        return None

    async def unsubscribe(self, channel):
        return None

    async def close(self):
        return None


class GapRedis:
    def pubsub(self):
        return FakePubSub()

    async def lrange(self, key, start, end):
        return [json.dumps({"id": 500, "event": "token", "data": {"text": "new"}})]

    async def get(self, key):
        return "700"


class NeverDisconnectedRequest:
    async def is_disconnected(self):
        return False


@pytest.mark.asyncio
async def test_trimmed_sse_history_emits_authoritative_reload_event(monkeypatch):
    async def get_redis():
        return GapRedis()

    monkeypatch.setattr(agent_endpoint, "get_redis_client", get_redis)
    events = []
    async for event in agent_endpoint.create_sse_event_generator(
        request=NeverDisconnectedRequest(),
        channel="run:run-gap",
        run_id="run-gap",
        workflow_kwargs={},
        endpoint_name="acceptance-gap",
        start_workflow=False,
        last_event_id=20,
    ):
        events.append(event.decode())

    assert len(events) == 1
    assert "event: stream.gap" in events[0]
    assert '"recovery": "reload_conversation"' in events[0]
    assert '"expected_event_id": 21' in events[0]
    assert '"oldest_available_event_id": 500' in events[0]


@pytest.mark.asyncio
async def test_expired_run_mapping_emits_same_authoritative_reload_event(monkeypatch):
    run_id = str(uuid.uuid4())
    conversation_id = str(uuid.uuid4())

    class Request:
        async def json(self):
            return {"conversation_id": conversation_id, "last_event_id": 0}

    class MissingMappingRedis:
        async def get(self, key):
            return None

    async def get_redis():
        return MissingMappingRedis()

    async def get_conversation(**kwargs):
        return SimpleNamespace(id=conversation_id, user_id=None)

    monkeypatch.setattr(agent_endpoint, "get_redis_client", get_redis)
    monkeypatch.setattr(Conversation, "get", get_conversation)
    monkeypatch.setattr(agent_endpoint, "check_conversation_authorization", lambda *args: None)

    response = await agent_endpoint.resume_run_events(run_id, Request(), user=None)
    body = b"".join([chunk async for chunk in response.body_iterator]).decode()

    assert response.status_code == 200
    assert "event: stream.gap" in body
    assert '"recovery": "reload_conversation"' in body
    assert '"reason": "run_mapping_expired"' in body
    assert f'"conversation_id": "{conversation_id}"' in body


@pytest.mark.asyncio
async def test_background_verification_uses_read_only_tools_outside_stop_tombstone(
    monkeypatch,
):
    operation_id = uuid.uuid4()
    plan = {
        "adapter": "kubernetes_apply",
        "targets": [
            {
                "kind": "ConfigMap",
                "name": "demo",
                "namespace": "default",
                "desired_spec": {},
            }
        ],
    }
    operation = SimpleNamespace(
        id=operation_id,
        run_id="stopped-run",
        tool_name="k8s_apply",
        execution_status=ExecutionStatus.UNKNOWN,
        verification_status=VerificationStatus.PENDING,
        verification_plan=plan,
        desired_state_hmac="desired-hmac",
        verification_result={},
    )

    class Journal:
        async def get(self, operation_id):
            return operation

        async def claim_verification(self, operation_id, *, lease_owner):
            operation.verification_status = VerificationStatus.VERIFYING
            return operation, True

        async def record_verification(self, operation_id, **kwargs):
            operation.verification_status = kwargs["status"]
            operation.verification_result = kwargs["result"]
            return operation

        @staticmethod
        def public_summary(item):
            return {
                "operation_id": str(item.id),
                "execution_status": item.execution_status.value,
                "verification_status": item.verification_status.value,
            }

    class MCP:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments, conversation_id=None):
            self.calls.append((name, conversation_id))
            resource = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "demo",
                    "namespace": "default",
                    "annotations": {
                        "skyflo.ai/operation-id": str(operation_id),
                        "skyflo.ai/desired-state-hmac": "desired-hmac",
                    },
                },
            }
            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(resource)}],
            }

    mcp = MCP()
    monkeypatch.setattr(
        "src.api.services.mutation_verifier.settings.MUTATION_VERIFICATION_TIMEOUT_SECONDS", 0
    )
    result = await MutationVerifier(mcp, Journal()).verify(
        str(operation_id), lease_owner="recovery:worker-1"
    )

    assert result["passed"] is True
    assert [name for name, _ in mcp.calls] == ["k8s_get"]
    assert all(run_id != operation.run_id for _, run_id in mcp.calls)
    assert all(
        run_id.startswith(f"verification:{operation_id}:recovery:")
        for _, run_id in mcp.calls
    )


@pytest.mark.asyncio
async def test_periodic_worker_recovers_lease_that_expires_after_start(monkeypatch):
    operation = SimpleNamespace(
        id=uuid.uuid4(),
        execution_status=ExecutionStatus.EXECUTING,
        verification_status=VerificationStatus.PENDING,
    )

    class Journal:
        def __init__(self):
            self.passes = 0

        async def recover_stale_executions(self):
            self.passes += 1
            if self.passes == 2:
                operation.execution_status = ExecutionStatus.UNKNOWN
                return 1
            return 0

        async def recover_stale_verifications(self):
            return 0

    class Query:
        async def limit(self, count):
            return [operation] if operation.execution_status == ExecutionStatus.UNKNOWN else []

    class Verifier:
        def __init__(self):
            self.calls = []

        async def verify(self, operation_id, *, lease_owner):
            self.calls.append((operation_id, lease_owner))
            return {"status": VerificationStatus.PASSED.value}

    monkeypatch.setattr(MutationOperation, "filter", lambda **kwargs: Query())
    worker = MutationRecoveryWorker.__new__(MutationRecoveryWorker)
    worker.journal = Journal()
    worker.verifier = Verifier()
    worker.on_reconciled = None

    assert await worker.reconcile_once() == 0
    assert await worker.reconcile_once() == 1
    assert worker.journal.passes == 2
    assert len(worker.verifier.calls) == 1
    assert worker.verifier.calls[0][0] == str(operation.id)
    assert worker.verifier.calls[0][1].startswith("recovery:")


@pytest.mark.asyncio
async def test_verification_budget_persists_needs_review_only_once(monkeypatch):
    operation = SimpleNamespace(
        id=uuid.uuid4(),
        execution_status=ExecutionStatus.UNKNOWN,
        verification_status=VerificationStatus.INCONCLUSIVE,
        verification_attempt_count=5,
        created_at=timezone.now(),
        verification_result={},
        verification_lease_owner="old-worker",
        verification_lease_expires_at=timezone.now(),
        version=7,
        save=AsyncMock(),
    )

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class Query:
        def select_for_update(self):
            return self

        async def get(self):
            return operation

    monkeypatch.setattr(journal_module, "in_transaction", lambda: Transaction())
    monkeypatch.setattr(MutationOperation, "filter", lambda **kwargs: Query())
    monkeypatch.setattr(
        journal_module.settings, "MUTATION_MAX_VERIFICATION_ATTEMPTS", 5
    )

    journal = MutationJournalService()
    first, first_claimed = await journal.claim_verification(
        str(operation.id), lease_owner="recovery:first"
    )
    assert first_claimed is False
    assert first.verification_status == VerificationStatus.NEEDS_REVIEW
    assert first.verification_result["manual_review_required"] is True
    operation.save.assert_awaited_once()

    operation.save.reset_mock()
    second, second_claimed = await journal.claim_verification(
        str(operation.id), lease_owner="recovery:second"
    )
    assert second_claimed is False
    assert second.verification_status == VerificationStatus.NEEDS_REVIEW
    operation.save.assert_not_awaited()
