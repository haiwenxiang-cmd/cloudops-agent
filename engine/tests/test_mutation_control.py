import json
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
from tortoise import timezone

from src.api.agent.graph import (
    WorkflowGraph,
    route_after_gate,
    route_after_model,
    route_after_verification,
    route_from_entry,
)
from src.api.models.mutation import ExecutionStatus, VerificationStatus
from src.api.services.mutation_journal import verification_budget_exhausted
from src.api.services.mutation_policy import MutationPolicyService
from src.api.services.mutation_utils import (
    attach_operation_metadata,
    hmac_fingerprint,
    parse_manifest_targets,
    redact_manifest_content,
    redact_sensitive,
    redact_tool_args,
)
from src.api.services.mutation_verifier import (
    MutationVerifier,
    _subset_matches,
    _workload_health,
    build_verification_plan,
)
from src.api.services.tool_capabilities import capability_error, get_tool_capability
from src.api.services.tool_executor import ToolExecutor

MANIFEST = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: demo
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: demo
  template:
    metadata:
      labels:
        app: demo
    spec:
      containers:
        - name: nginx
          image: nginx:1.27-alpine
"""


def test_redacts_nested_secrets_but_preserves_shape():
    value = {
        "password": "hunter2",
        "nested": {"api_key": "secret", "name": "safe"},
        "tokens": [{"access_token": "abc"}],
    }
    redacted = redact_sensitive(value)
    assert redacted["password"] == "[REDACTED]"
    assert redacted["nested"] == {"api_key": "[REDACTED]", "name": "safe"}
    assert redacted["tokens"] == "[REDACTED]"


def test_hmac_is_stable_across_dictionary_order():
    assert hmac_fingerprint("key", {"b": 2, "a": 1}) == hmac_fingerprint(
        "key", {"a": 1, "b": 2}
    )


def test_manifest_secret_payload_is_not_journaled():
    manifest = """
apiVersion: v1
kind: Secret
metadata:
  name: demo-secret
stringData:
  password: super-secret
data:
  token: ZG9ub3Rsb2c=
"""
    redacted = redact_manifest_content(manifest)
    assert "super-secret" not in redacted
    assert "ZG9ub3Rsb2c=" not in redacted
    assert "[REDACTED]" in redacted
    event_args = redact_tool_args("k8s_apply", {"content": manifest})
    assert "super-secret" not in event_args["content"]
    helm_args = redact_tool_args(
        "helm_install_with_values",
        {"values": "database:\n  password: super-secret\n  replicas: 2\n"},
    )
    assert "super-secret" not in helm_args["values"]
    assert "replicas: 2" in helm_args["values"]


def test_k8s_apply_gets_external_operation_marker():
    operation_id = str(uuid.uuid4())
    prepared = attach_operation_metadata(
        "k8s_apply",
        {"content": MANIFEST, "namespace": "default"},
        operation_id,
        "abc123",
    )
    assert f"skyflo.ai/operation-id: {operation_id}" in prepared["content"]
    assert "skyflo.ai/desired-state-hmac: abc123" in prepared["content"]
    targets = parse_manifest_targets(prepared["content"])
    assert targets[0]["name"] == "demo"
    assert targets[0]["desired_spec"]["replicas"] == 2


def test_policy_requires_authenticated_admin(monkeypatch):
    monkeypatch.setattr(
        "src.api.services.mutation_policy.settings.MUTATION_ALLOWED_NAMESPACES",
        "default,skyflo",
    )
    policy = MutationPolicyService()
    anonymous = policy.evaluate(
        tool_name="k8s_apply",
        args={"content": MANIFEST},
        user_id=None,
        user_role=None,
    )
    member = policy.evaluate(
        tool_name="k8s_apply",
        args={"content": MANIFEST},
        user_id=str(uuid.uuid4()),
        user_role="member",
    )
    admin = policy.evaluate(
        tool_name="k8s_apply",
        args={"content": MANIFEST},
        user_id=str(uuid.uuid4()),
        user_role="admin",
    )
    assert not anonymous.allowed
    assert not member.allowed
    assert admin.allowed


def test_policy_denies_rbac_and_cluster_scoped_manifests(monkeypatch):
    monkeypatch.setattr(
        "src.api.services.mutation_policy.settings.MUTATION_ALLOWED_NAMESPACES", "default"
    )
    policy = MutationPolicyService()
    manifest = """
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: pwn
roleRef: {}
subjects: []
"""
    decision = policy.evaluate(
        tool_name="k8s_apply",
        args={"content": manifest},
        user_id=str(uuid.uuid4()),
        user_role="admin",
    )
    assert not decision.allowed
    assert decision.rule == "cluster_or_rbac_resource_denied"


def test_subset_match_keys_lists_by_name():
    desired = {"containers": [{"name": "app", "image": "v2"}]}
    actual = {
        "containers": [
            {"name": "sidecar", "image": "metrics"},
            {"name": "app", "image": "v2", "imagePullPolicy": "Always"},
        ]
    }
    assert _subset_matches(desired, actual)


def test_deployment_health_requires_observed_and_available():
    resource = {
        "kind": "Deployment",
        "metadata": {"generation": 3},
        "spec": {"replicas": 2},
        "status": {
            "observedGeneration": 3,
            "updatedReplicas": 2,
            "availableReplicas": 2,
            "unavailableReplicas": 0,
        },
    }
    assert _workload_health(resource)[0]
    resource["status"]["observedGeneration"] = 2
    assert not _workload_health(resource)[0]


def test_graph_routes_mutation_through_verification():
    assert route_from_entry({"verification_required": True}) == "verification"
    assert (
        route_from_entry(
            {"verification_required": True, "pending_tools": [{"name": "tool"}]}
        )
        == "gate"
    )
    assert route_after_gate({"verification_required": True}) == "verification"
    assert route_after_model({"verification_required": True, "pending_tools": []}) == "verification"
    assert route_after_verification({"verification_required": False}) == "model"
    assert (
        route_after_verification(
            {"verification_required": True, "verification_blocked": True}
        )
        == "final"
    )


def test_capability_registry_overrides_conflicting_mcp_annotation():
    capability = get_tool_capability("k8s_apply")
    assert capability is not None and capability.controlled
    assert capability_error(
        "k8s_apply", {"annotations": {"readOnlyHint": True}}
    ) is not None


def test_verification_budget_has_bounded_manual_review_exit(monkeypatch):
    monkeypatch.setattr(
        "src.api.services.mutation_journal.settings.MUTATION_MAX_VERIFICATION_ATTEMPTS", 3
    )
    monkeypatch.setattr(
        "src.api.services.mutation_journal.settings.MUTATION_MAX_VERIFICATION_AGE_SECONDS", 60
    )
    now = timezone.now()
    by_attempts = SimpleNamespace(
        verification_attempt_count=3,
        created_at=now - timedelta(seconds=10),
    )
    by_age = SimpleNamespace(
        verification_attempt_count=1,
        created_at=now - timedelta(seconds=61),
    )
    within_budget = SimpleNamespace(
        verification_attempt_count=2,
        created_at=now - timedelta(seconds=10),
    )
    assert verification_budget_exhausted(by_attempts, now)
    assert verification_budget_exhausted(by_age, now)
    assert not verification_budget_exhausted(within_budget, now)
    assert capability_error(
        "k8s_apply", {"annotations": {"readOnlyHint": False}}
    ) is None
    assert capability_error(
        "k8s_delete", {"annotations": {"readOnlyHint": False}}
    ) is not None


@pytest.mark.asyncio
async def test_controlled_mutation_error_after_start_is_unknown(monkeypatch):
    operation = SimpleNamespace(
        id=uuid.uuid4(),
        call_id="call-partial",
        tool_name="k8s_apply",
        execution_status=ExecutionStatus.PREPARED,
        verification_status=VerificationStatus.PENDING,
        desired_state_hmac="desired",
        attempt_count=0,
        verification_attempt_count=0,
    )

    class ErrorMCP:
        async def call_tool(self, tool_name, parameters, conversation_id=None):
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": "service rejected after deployment/demo configured",
                    }
                ],
                "reliability": {"error_type": "invalid_input"},
            }

    class Journal:
        recorded = None

        async def prepare(self, **kwargs):
            return operation, True

        async def mark_executing(self, operation_id, lease_owner):
            operation.execution_status = ExecutionStatus.EXECUTING
            return operation

        async def record_execution(self, operation_id, *, status, result=None, error=None):
            self.recorded = (status, error)
            operation.execution_status = status
            operation.execution_error = error or {}
            return operation

    executor = ToolExecutor(mcp_client=ErrorMCP())
    executor.mutation_journal = Journal()

    async def metadata(_name):
        return {
            "name": "k8s_apply",
            "title": "Apply Kubernetes Manifest",
            "annotations": {"readOnlyHint": False},
            "inputSchema": {"type": "object"},
        }

    async def passthrough_integration(**kwargs):
        return kwargs["args"], None

    monkeypatch.setattr(executor, "_get_tool_metadata", metadata)
    monkeypatch.setattr(executor, "inject_integration_tool_params", passthrough_integration)
    monkeypatch.setattr(
        "src.api.services.mutation_policy.settings.MUTATION_ALLOWED_NAMESPACES",
        "default",
    )
    result = await executor.execute(
        run_id="run-partial",
        name="k8s_apply",
        args={"content": MANIFEST, "namespace": "default"},
        call_id="call-partial",
        operation_id=str(operation.id),
        context={
            "user_id": str(uuid.uuid4()),
            "user_role": "admin",
            "conversation_id": "conversation-1",
            "approval_decisions": {"call-partial": True},
        },
    )

    assert executor.mutation_journal.recorded[0] == ExecutionStatus.UNKNOWN
    assert executor.mutation_journal.recorded[1]["reliability"]["ambiguous_outcome"] is True
    assert any(
        block.get("type") == "skyflo.mutation" and block.get("requires_verification")
        for block in result
    )
@pytest.mark.asyncio
async def test_resumed_gate_preserves_earlier_verification_requirement(monkeypatch):
    async def no_stop(_state):
        return None

    class DeniedToolExecutor:
        async def execute(self, **_kwargs):
            return [{"type": "text", "text": "Tool call was denied by the user"}]

    monkeypatch.setattr("src.api.agent.graph.check_stop", no_stop)
    graph = WorkflowGraph(event_callback=None)
    graph.tool_executor = DeniedToolExecutor()
    operation_id = str(uuid.uuid4())
    result = await graph._gate_node(
        {
            "run_id": str(uuid.uuid4()),
            "pending_tools": [
                {
                    "id": "provider-call-2",
                    "call_id": "call-2",
                    "operation_id": str(uuid.uuid4()),
                    "name": "jenkins_build",
                    "args": {},
                }
            ],
            "mutation_operations": [
                {
                    "type": "skyflo.mutation",
                    "operation_id": operation_id,
                    "requires_verification": True,
                }
            ],
        }
    )
    assert result["verification_required"] is True
    assert result["mutation_operations"][0]["operation_id"] == operation_id


class FakeMCP:
    def __init__(self, resource):
        self.resource = resource
        self.calls = []

    async def call_tool(self, name, arguments, conversation_id=None):
        self.calls.append((name, arguments, conversation_id))
        return {
            "isError": False,
            "content": [{"type": "text", "text": json.dumps(self.resource)}],
        }


class FakeJournal:
    def __init__(self, operation):
        self.operation = operation
        self.recorded_status = None

    async def get(self, operation_id):
        return self.operation

    async def claim_verification(self, operation_id, *, lease_owner):
        self.operation.verification_status = VerificationStatus.VERIFYING
        return self.operation, True

    async def record_verification(
        self,
        operation_id,
        *,
        lease_owner,
        status,
        result,
        external_reference=None,
    ):
        self.recorded_status = status
        self.operation.verification_status = status
        self.operation.verification_result = result
        return self.operation

    async def abandon_verification(self, operation_id, *, lease_owner, reason):
        self.operation.verification_status = VerificationStatus.INCONCLUSIVE
        return True

    @staticmethod
    def public_summary(operation):
        return {
            "operation_id": str(operation.id),
            "tool": operation.tool_name,
            "execution_status": operation.execution_status.value,
            "verification_status": operation.verification_status.value,
            "attempt_count": 1,
        }


@pytest.mark.asyncio
async def test_unknown_k8s_outcome_is_reconciled_by_external_marker(monkeypatch):
    operation_id = uuid.uuid4()
    plan = build_verification_plan(
        "k8s_apply", {"content": MANIFEST, "namespace": "default"}, str(operation_id)
    )
    operation = SimpleNamespace(
        id=operation_id,
        run_id="run-1",
        tool_name="k8s_apply",
        execution_status=ExecutionStatus.UNKNOWN,
        verification_status=VerificationStatus.PENDING,
        verification_plan=plan,
        desired_state_hmac="desired-hmac",
        verification_result={},
    )
    resource = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "demo",
            "namespace": "default",
            "uid": "uid-1",
            "generation": 1,
            "annotations": {
                "skyflo.ai/operation-id": str(operation_id),
                "skyflo.ai/desired-state-hmac": "desired-hmac",
            },
        },
        "spec": plan["targets"][0]["desired_spec"],
        "status": {
            "observedGeneration": 1,
            "updatedReplicas": 2,
            "availableReplicas": 2,
            "unavailableReplicas": 0,
        },
    }
    journal = FakeJournal(operation)
    verifier = MutationVerifier(FakeMCP(resource), journal)
    monkeypatch.setattr(
        "src.api.services.mutation_verifier.settings.MUTATION_VERIFICATION_TIMEOUT_SECONDS", 0
    )
    result = await verifier.verify(str(operation_id))
    assert result["passed"] is True
    assert journal.recorded_status == VerificationStatus.PASSED


@pytest.mark.asyncio
async def test_multi_resource_partial_commit_becomes_verified_failure(monkeypatch):
    operation_id = uuid.uuid4()
    manifest = MANIFEST + """
---
apiVersion: v1
kind: Service
metadata:
  name: demo
  namespace: default
spec:
  selector:
    app: demo
  ports:
    - port: 80
"""
    plan = build_verification_plan(
        "k8s_apply", {"content": manifest, "namespace": "default"}, str(operation_id)
    )
    operation = SimpleNamespace(
        id=operation_id,
        run_id="run-partial",
        tool_name="k8s_apply",
        execution_status=ExecutionStatus.UNKNOWN,
        verification_status=VerificationStatus.PENDING,
        verification_plan=plan,
        desired_state_hmac="desired-hmac",
        verification_result={},
    )

    class PartialMCP:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments, conversation_id=None):
            self.calls.append((name, arguments, conversation_id))
            if arguments["resource_type"] == "Service":
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": "NotFound"}],
                    "reliability": {"error_type": "not_found"},
                }
            target = plan["targets"][0]
            resource = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "demo",
                    "namespace": "default",
                    "generation": 1,
                    "annotations": {
                        "skyflo.ai/operation-id": str(operation_id),
                        "skyflo.ai/desired-state-hmac": "desired-hmac",
                    },
                },
                "spec": target["desired_spec"],
                "status": {
                    "observedGeneration": 1,
                    "updatedReplicas": 2,
                    "availableReplicas": 2,
                    "unavailableReplicas": 0,
                },
            }
            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(resource)}],
            }

    mcp = PartialMCP()
    journal = FakeJournal(operation)
    verifier = MutationVerifier(mcp, journal)
    monkeypatch.setattr(
        "src.api.services.mutation_verifier.settings.MUTATION_VERIFICATION_TIMEOUT_SECONDS", 0
    )

    result = await verifier.verify(str(operation_id))

    assert result["passed"] is False
    assert result["status"] == VerificationStatus.FAILED.value
    assert result["evidence"]["reason"] == "partial_commit_detected"
    assert result["operation"]["operation_id"] == str(operation_id)
    assert result["evidence"]["partial_commit_detected"] is True
    committed = result["evidence"]["targets"][0]
    assert committed["operation_marker_matches"] is True
    assert committed["desired_marker_matches"] is True
    assert committed["desired_spec_matches"] is True
    assert [call[0] for call in mcp.calls] == ["k8s_get", "k8s_get"]


@pytest.mark.asyncio
async def test_helm_verification_checks_rendered_workload_health(monkeypatch):
    operation_id = uuid.uuid4()
    plan = build_verification_plan(
        "helm_upgrade",
        {"release_name": "demo", "namespace": "default"},
        str(operation_id),
    )
    operation = SimpleNamespace(
        id=operation_id,
        run_id="run-helm",
        tool_name="helm_upgrade",
        execution_status=ExecutionStatus.REPORTED_SUCCESS,
        verification_status=VerificationStatus.PENDING,
        verification_plan=plan,
        desired_state_hmac="unused-for-helm",
        verification_result={},
    )

    class HelmMCP:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments, conversation_id=None):
            self.calls.append(name)
            if name == "helm_status":
                payload = {
                    "version": 2,
                    "info": {
                        "status": "deployed",
                        "description": f"skyflo-operation:{operation_id}",
                    },
                }
                text = json.dumps(payload)
            elif name == "helm_get_manifest":
                text = MANIFEST
            else:
                target = parse_manifest_targets(MANIFEST)[0]
                text = json.dumps(
                    {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {
                            "name": "demo",
                            "namespace": "default",
                            "uid": "helm-workload-1",
                            "generation": 1,
                        },
                        "spec": target["desired_spec"],
                        "status": {
                            "observedGeneration": 1,
                            "updatedReplicas": 2,
                            "availableReplicas": 2,
                            "unavailableReplicas": 0,
                        },
                    }
                )
            return {"isError": False, "content": [{"type": "text", "text": text}]}

    mcp = HelmMCP()
    journal = FakeJournal(operation)
    verifier = MutationVerifier(mcp, journal)
    monkeypatch.setattr(
        "src.api.services.mutation_verifier.settings.MUTATION_VERIFICATION_TIMEOUT_SECONDS", 0
    )

    result = await verifier.verify(str(operation_id))

    assert result["passed"] is True
    assert result["evidence"]["verification_scope"] == "release_and_workloads"
    assert result["evidence"]["workloads"][0]["passed"] is True
    assert mcp.calls == ["helm_status", "helm_get_manifest", "k8s_get"]


@pytest.mark.asyncio
async def test_helm_deployed_release_does_not_pass_with_unready_workload(monkeypatch):
    operation_id = uuid.uuid4()
    plan = build_verification_plan(
        "helm_upgrade",
        {"release_name": "demo", "namespace": "default"},
        str(operation_id),
    )
    operation = SimpleNamespace(
        id=operation_id,
        run_id="run-helm-unready",
        tool_name="helm_upgrade",
        execution_status=ExecutionStatus.REPORTED_SUCCESS,
        verification_status=VerificationStatus.PENDING,
        verification_plan=plan,
        desired_state_hmac="unused-for-helm",
        verification_result={},
    )

    class UnreadyHelmMCP:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments, conversation_id=None):
            self.calls.append(name)
            if name == "helm_status":
                text = json.dumps(
                    {
                        "version": 3,
                        "info": {
                            "status": "deployed",
                            "description": f"skyflo-operation:{operation_id}",
                        },
                    }
                )
            elif name == "helm_get_manifest":
                text = MANIFEST
            else:
                target = parse_manifest_targets(MANIFEST)[0]
                text = json.dumps(
                    {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {
                            "name": "demo",
                            "namespace": "default",
                            "generation": 2,
                        },
                        "spec": target["desired_spec"],
                        "status": {
                            "observedGeneration": 1,
                            "updatedReplicas": 1,
                            "availableReplicas": 0,
                            "unavailableReplicas": 2,
                        },
                    }
                )
            return {"isError": False, "content": [{"type": "text", "text": text}]}

    mcp = UnreadyHelmMCP()
    journal = FakeJournal(operation)
    verifier = MutationVerifier(mcp, journal)
    monkeypatch.setattr(
        "src.api.services.mutation_verifier.settings.MUTATION_VERIFICATION_TIMEOUT_SECONDS", 0
    )

    result = await verifier.verify(str(operation_id))

    assert result["passed"] is False
    assert result["status"] == VerificationStatus.INCONCLUSIVE.value
    assert result["evidence"]["status"] == "deployed"
    assert result["evidence"]["marker_matches"] is True
    assert result["evidence"]["workloads"][0]["passed"] is False
    assert "available=0/2" in result["evidence"]["workloads"][0]["detail"]
    assert mcp.calls == ["helm_status", "helm_get_manifest", "k8s_get"]


@pytest.mark.asyncio
async def test_active_verification_lease_suppresses_duplicate_reader():
    operation = SimpleNamespace(
        id=uuid.uuid4(),
        run_id="run-1",
        tool_name="k8s_apply",
        execution_status=ExecutionStatus.UNKNOWN,
        verification_status=VerificationStatus.VERIFYING,
        verification_plan={"adapter": "kubernetes_apply", "targets": []},
        verification_result={},
        desired_state_hmac="desired-hmac",
        attempt_count=1,
    )

    class BusyJournal(FakeJournal):
        async def claim_verification(self, operation_id, *, lease_owner):
            return self.operation, False

    mcp = FakeMCP({})
    result = await MutationVerifier(mcp, BusyJournal(operation)).verify(
        str(operation.id), lease_owner="foreground:run-2"
    )
    assert result["status"] == VerificationStatus.VERIFYING.value
    assert result["inconclusive"] is True
    assert mcp.calls == []
