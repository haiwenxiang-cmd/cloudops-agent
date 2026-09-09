"""Deterministic postcondition verification for controlled mutations."""

import asyncio
import json
import logging
import time
import uuid
from datetime import timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from tortoise import timezone

from ..config import settings
from ..models.mutation import ExecutionStatus, MutationOperation, VerificationStatus
from .mcp_client import MCPClient
from .mutation_journal import MutationJournalService
from .mutation_utils import parse_manifest_targets

CONTROLLED_MUTATION_TOOLS = frozenset(
    {
        "k8s_apply",
        "helm_install",
        "helm_install_with_values",
        "helm_upgrade",
        "helm_rollback",
    }
)

logger = logging.getLogger(__name__)

RecoveryCallback = Callable[[MutationOperation, Dict[str, Any]], Awaitable[None]]


def build_verification_plan(
    tool_name: str, args: Dict[str, Any], operation_id: str
) -> Dict[str, Any]:
    if tool_name == "k8s_apply":
        targets = parse_manifest_targets(
            str(args.get("content") or ""), str(args.get("namespace") or "default")
        )
        return {
            "supported": bool(targets),
            "adapter": "kubernetes_apply",
            "operation_id": operation_id,
            "targets": targets,
        }
    if tool_name in {
        "helm_install",
        "helm_install_with_values",
        "helm_upgrade",
        "helm_rollback",
    }:
        return {
            "supported": bool(args.get("release_name")),
            "adapter": "helm_release",
            "operation_id": operation_id,
            "release_name": args.get("release_name"),
            "namespace": args.get("namespace") or "default",
        }
    return {"supported": False, "adapter": "unsupported", "operation_id": operation_id}


def _result_text(result: Dict[str, Any]) -> str:
    parts: List[str] = []
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(parts).strip()


def _terminal_read_error(result: Dict[str, Any]) -> bool:
    reliability = result.get("reliability") or {}
    if isinstance(reliability, list):
        reliability = reliability[0] if reliability else {}
    error_type = reliability.get("error_type") if isinstance(reliability, dict) else None
    return error_type in {"authorization", "invalid_input", "executable_not_found"}


def _read_error_type(result: Dict[str, Any]) -> Optional[str]:
    reliability = result.get("reliability") or {}
    if isinstance(reliability, list):
        reliability = reliability[0] if reliability else {}
    return reliability.get("error_type") if isinstance(reliability, dict) else None


def _subset_matches(desired: Any, actual: Any) -> bool:
    if isinstance(desired, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset_matches(value, actual[key]) for key, value in desired.items()
        )
    if isinstance(desired, list):
        if not isinstance(actual, list) or len(desired) > len(actual):
            return False
        # Kubernetes lists such as containers are usually keyed by name.
        if all(isinstance(item, dict) and item.get("name") for item in desired):
            actual_by_name = {
                item.get("name"): item
                for item in actual
                if isinstance(item, dict) and item.get("name")
            }
            return all(
                item["name"] in actual_by_name
                and _subset_matches(item, actual_by_name[item["name"]])
                for item in desired
            )
        return all(_subset_matches(value, actual[index]) for index, value in enumerate(desired))
    return desired == actual


def _workload_health(resource: Dict[str, Any]) -> Tuple[bool, bool, str]:
    kind = str(resource.get("kind") or "").lower()
    metadata = resource.get("metadata") or {}
    spec = resource.get("spec") or {}
    status = resource.get("status") or {}
    generation = int(metadata.get("generation") or 0)
    observed = int(status.get("observedGeneration") or 0)

    if kind == "deployment":
        for condition in status.get("conditions") or []:
            if (
                condition.get("type") == "Progressing"
                and condition.get("status") == "False"
                and condition.get("reason") == "ProgressDeadlineExceeded"
            ):
                return False, True, "deployment exceeded its progress deadline"
        replicas = int(spec.get("replicas", 1))
        ready = int(status.get("availableReplicas") or 0)
        updated = int(status.get("updatedReplicas") or 0)
        unavailable = int(status.get("unavailableReplicas") or 0)
        passed = (
            observed >= generation
            and ready >= replicas
            and updated >= replicas
            and unavailable == 0
        )
        return passed, False, (
            f"deployment observed={observed}/{generation}, updated={updated}/{replicas}, "
            f"available={ready}/{replicas}, unavailable={unavailable}"
        )

    if kind == "statefulset":
        replicas = int(spec.get("replicas", 1))
        ready = int(status.get("readyReplicas") or 0)
        updated = int(status.get("updatedReplicas") or 0)
        revisions_match = status.get("currentRevision") == status.get("updateRevision")
        passed = (
            observed >= generation
            and ready >= replicas
            and updated >= replicas
            and revisions_match
        )
        return passed, False, f"statefulset ready={ready}/{replicas}, updated={updated}/{replicas}"

    if kind == "daemonset":
        desired = int(status.get("desiredNumberScheduled") or 0)
        ready = int(status.get("numberReady") or 0)
        updated = int(status.get("updatedNumberScheduled") or 0)
        passed = observed >= generation and desired > 0 and ready >= desired and updated >= desired
        return passed, False, f"daemonset ready={ready}/{desired}, updated={updated}/{desired}"

    if kind == "pod":
        phase = status.get("phase")
        containers = status.get("containerStatuses") or []
        ready = bool(containers) and all(bool(item.get("ready")) for item in containers)
        return phase == "Running" and ready, phase == "Failed", f"pod phase={phase}, ready={ready}"

    if kind == "job":
        conditions = {
            item.get("type"): item.get("status") for item in status.get("conditions") or []
        }
        if conditions.get("Failed") == "True":
            return False, True, "job reported Failed=True"
        completed = conditions.get("Complete") == "True"
        return completed, False, f"job complete={completed}"

    return True, False, "resource exists and desired fields match"


class MutationVerifier:
    def __init__(self, mcp_client: MCPClient, journal: Optional[MutationJournalService] = None):
        self.mcp = mcp_client
        self.journal = journal or MutationJournalService()

    async def verify(
        self, operation_id: str, *, lease_owner: Optional[str] = None
    ) -> Dict[str, Any]:
        operation = await self.journal.get(operation_id)
        if operation is None:
            return {"passed": False, "inconclusive": True, "reason": "operation_not_found"}
        owner = lease_owner or f"verifier:{uuid.uuid4()}"
        # A Stop tombstone intentionally blocks late commands from the original
        # workflow run. Background reconciliation is a new, read-only execution
        # chain, so it must not reuse the cancelled run id.
        verification_run_id = (
            f"verification:{operation.id}:{owner}"
            if owner.startswith("recovery:")
            else operation.run_id
        )
        operation, claimed = await self.journal.claim_verification(
            operation_id, lease_owner=owner
        )
        if not claimed:
            status = operation.verification_status
            if status == VerificationStatus.PASSED:
                return {
                    "passed": True,
                    "inconclusive": False,
                    "status": status.value,
                    "operation": self.journal.public_summary(operation),
                    "evidence": operation.verification_result or {},
                }
            return {
                "passed": False,
                "inconclusive": status not in {
                    VerificationStatus.FAILED,
                    VerificationStatus.NEEDS_REVIEW,
                },
                "status": status.value,
                "operation": self.journal.public_summary(operation),
                "evidence": operation.verification_result
                or {"reason": "verification_already_in_progress"},
            }
        deadline = time.monotonic() + settings.MUTATION_VERIFICATION_TIMEOUT_SECONDS
        last_result: Dict[str, Any] = {}

        try:
            while True:
                try:
                    if operation.verification_plan.get("adapter") == "kubernetes_apply":
                        passed, terminal, last_result = await self._verify_kubernetes(
                            operation, verification_run_id
                        )
                    elif operation.verification_plan.get("adapter") == "helm_release":
                        passed, terminal, last_result = await self._verify_helm(
                            operation, verification_run_id
                        )
                    else:
                        passed, terminal, last_result = False, True, {
                            "reason": "verification_adapter_not_supported"
                        }
                except Exception as exc:
                    logger.exception("Verification adapter failed for %s", operation_id)
                    passed, terminal, last_result = False, False, {
                        "reason": "verification_adapter_error",
                        "detail": type(exc).__name__,
                    }
                if passed:
                    status = VerificationStatus.PASSED
                    break
                if terminal:
                    status = VerificationStatus.FAILED
                    break
                if time.monotonic() >= deadline:
                    if last_result.get("partial_commit_detected"):
                        status = VerificationStatus.FAILED
                        last_result["reason"] = "partial_commit_detected"
                    else:
                        status = VerificationStatus.INCONCLUSIVE
                        last_result["reason"] = "verification_timeout"
                    break
                await asyncio.sleep(settings.MUTATION_VERIFICATION_INTERVAL_SECONDS)

            operation = await self.journal.record_verification(
                operation_id,
                lease_owner=owner,
                status=status,
                result=last_result,
                external_reference=last_result.get("external_reference"),
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                self.journal.abandon_verification(
                    operation_id,
                    lease_owner=owner,
                    reason="verification_cancelled",
                )
            )
            raise
        return {
            "passed": status == VerificationStatus.PASSED,
            "inconclusive": status == VerificationStatus.INCONCLUSIVE,
            "status": status.value,
            "operation": self.journal.public_summary(operation),
            "evidence": last_result,
        }

    async def _verify_kubernetes(
        self, operation: MutationOperation, verification_run_id: str
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        references: List[Dict[str, Any]] = []
        all_passed = True
        terminal_failure = False
        committed_count = 0
        missing_count = 0
        for target in operation.verification_plan.get("targets", []):
            result = await self.mcp.call_tool(
                "k8s_get",
                {
                    "resource_type": target["kind"],
                    "name": target["name"],
                    "namespace": target.get("namespace") or "default",
                    "output": "json",
                },
                conversation_id=verification_run_id,
            )
            text = _result_text(result)
            item = {"target": {k: target.get(k) for k in ("kind", "name", "namespace")}}
            if result.get("isError"):
                item.update({"passed": False, "detail": text[:500]})
                evidence.append(item)
                all_passed = False
                if _read_error_type(result) == "not_found" or "not found" in text.lower():
                    missing_count += 1
                terminal_failure = terminal_failure or _terminal_read_error(result)
                continue
            try:
                resource = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                item.update({"passed": False, "detail": "invalid Kubernetes JSON response"})
                evidence.append(item)
                all_passed = False
                continue

            metadata = resource.get("metadata") or {}
            annotations = metadata.get("annotations") or {}
            operation_marker_matches = (
                annotations.get("skyflo.ai/operation-id") == str(operation.id)
            )
            desired_marker_matches = (
                annotations.get("skyflo.ai/desired-state-hmac")
                == operation.desired_state_hmac
            )
            marker_matches = operation_marker_matches and desired_marker_matches
            conflicting_marker = bool(annotations.get("skyflo.ai/operation-id")) and not (
                operation_marker_matches
            )
            desired_matches = _subset_matches(
                target.get("desired_spec", {}), resource.get("spec", {})
            )
            healthy, terminal, detail = _workload_health(resource)
            passed = marker_matches and desired_matches and healthy
            if marker_matches and desired_matches:
                committed_count += 1
            item.update(
                {
                    "passed": passed,
                    "marker_matches": marker_matches,
                    "operation_marker_matches": operation_marker_matches,
                    "desired_marker_matches": desired_marker_matches,
                    "desired_spec_matches": desired_matches,
                    "detail": detail,
                }
            )
            evidence.append(item)
            references.append(
                {
                    "api_version": resource.get("apiVersion"),
                    "kind": resource.get("kind"),
                    "namespace": metadata.get("namespace"),
                    "name": metadata.get("name"),
                    "uid": metadata.get("uid"),
                    "resource_version": metadata.get("resourceVersion"),
                    "generation": metadata.get("generation"),
                }
            )
            all_passed = all_passed and passed
            terminal_failure = terminal_failure or terminal or conflicting_marker

        return all_passed and bool(evidence), terminal_failure, {
            "targets": evidence,
            "partial_commit_detected": committed_count > 0 and missing_count > 0,
            "external_reference": {"resources": references},
        }

    async def _verify_helm(
        self, operation: MutationOperation, verification_run_id: str
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        plan = operation.verification_plan
        result = await self.mcp.call_tool(
            "helm_status",
            {
                "release_name": plan["release_name"],
                "namespace": plan["namespace"],
                "output": "json",
            },
            conversation_id=verification_run_id,
        )
        text = _result_text(result)
        if result.get("isError"):
            return False, _terminal_read_error(result), {
                "reason": "helm_status_error",
                "detail": text[:500],
            }
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return False, False, {"reason": "invalid_helm_status_json", "detail": text[:500]}
        info = payload.get("info") or {}
        status = str(info.get("status") or payload.get("status") or "").lower()
        description = str(info.get("description") or "")
        marker_matches = f"skyflo-operation:{operation.id}" in description
        release_passed = status == "deployed" and marker_matches
        terminal = status in {"failed", "uninstalled"}
        if not release_passed:
            return False, terminal, {
                "release": plan["release_name"],
                "namespace": plan["namespace"],
                "status": status,
                "marker_matches": marker_matches,
                "verification_scope": "release",
                "external_reference": {
                    "release": plan["release_name"],
                    "namespace": plan["namespace"],
                    "version": payload.get("version"),
                },
            }

        manifest_result = await self.mcp.call_tool(
            "helm_get_manifest",
            {
                "release_name": plan["release_name"],
                "namespace": plan["namespace"],
            },
            conversation_id=verification_run_id,
        )
        manifest_text = _result_text(manifest_result)
        if manifest_result.get("isError"):
            return False, _terminal_read_error(manifest_result), {
                "release": plan["release_name"],
                "namespace": plan["namespace"],
                "status": status,
                "marker_matches": marker_matches,
                "verification_scope": "release_and_workloads",
                "reason": "helm_manifest_unavailable",
                "detail": manifest_text[:500],
            }

        try:
            manifest_targets = parse_manifest_targets(manifest_text, plan["namespace"])
        except Exception as exc:
            return False, False, {
                "release": plan["release_name"],
                "namespace": plan["namespace"],
                "status": status,
                "marker_matches": marker_matches,
                "verification_scope": "release_and_workloads",
                "reason": "helm_manifest_invalid",
                "detail": type(exc).__name__,
            }

        workload_kinds = {"deployment", "statefulset", "daemonset", "pod", "job"}
        workloads = [
            target
            for target in manifest_targets
            if str(target.get("kind") or "").lower() in workload_kinds
        ]
        workload_evidence: List[Dict[str, Any]] = []
        workload_references: List[Dict[str, Any]] = []
        workloads_passed = True
        workload_terminal = False
        for target in workloads:
            workload_result = await self.mcp.call_tool(
                "k8s_get",
                {
                    "resource_type": target["kind"],
                    "name": target["name"],
                    "namespace": target.get("namespace") or plan["namespace"],
                    "output": "json",
                },
                conversation_id=verification_run_id,
            )
            item = {"target": {k: target.get(k) for k in ("kind", "name", "namespace")}}
            workload_text = _result_text(workload_result)
            if workload_result.get("isError"):
                item.update({"passed": False, "detail": workload_text[:500]})
                workload_evidence.append(item)
                workloads_passed = False
                workload_terminal = workload_terminal or _terminal_read_error(workload_result)
                continue
            try:
                resource = json.loads(workload_text)
            except (TypeError, json.JSONDecodeError):
                item.update({"passed": False, "detail": "invalid Kubernetes JSON response"})
                workload_evidence.append(item)
                workloads_passed = False
                continue
            desired_matches = _subset_matches(
                target.get("desired_spec", {}), resource.get("spec", {})
            )
            healthy, health_terminal, detail = _workload_health(resource)
            item.update(
                {
                    "passed": desired_matches and healthy,
                    "desired_spec_matches": desired_matches,
                    "detail": detail,
                }
            )
            workload_evidence.append(item)
            metadata = resource.get("metadata") or {}
            workload_references.append(
                {
                    "kind": resource.get("kind"),
                    "namespace": metadata.get("namespace"),
                    "name": metadata.get("name"),
                    "uid": metadata.get("uid"),
                    "generation": metadata.get("generation"),
                }
            )
            workloads_passed = workloads_passed and desired_matches and healthy
            workload_terminal = workload_terminal or health_terminal

        passed = release_passed and (not workloads or workloads_passed)
        return passed, terminal or workload_terminal, {
            "release": plan["release_name"],
            "namespace": plan["namespace"],
            "status": status,
            "marker_matches": marker_matches,
            "verification_scope": (
                "release_and_workloads" if workloads else "release_no_workloads_declared"
            ),
            "workloads": workload_evidence,
            "external_reference": {
                "release": plan["release_name"],
                "namespace": plan["namespace"],
                "version": payload.get("version"),
                "workloads": workload_references,
            },
        }


class MutationRecoveryWorker:
    """Reconciles abandoned/ambiguous operations without repeating mutations."""

    def __init__(self, on_reconciled: Optional[RecoveryCallback] = None) -> None:
        self.journal = MutationJournalService()
        self.mcp = MCPClient()
        self.verifier = MutationVerifier(self.mcp, self.journal)
        self.on_reconciled = on_reconciled
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        recovered = await self.journal.recover_stale_executions()
        if recovered:
            logger.warning("Recovered %s stale mutation operations as UNKNOWN", recovered)
        recovered_verifications = await self.journal.recover_stale_verifications()
        if recovered_verifications:
            logger.warning(
                "Recovered %s expired verification leases as INCONCLUSIVE",
                recovered_verifications,
            )
        self._task = asyncio.create_task(self._run(), name="mutation-recovery-worker")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Mutation recovery pass failed")
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=settings.MUTATION_RECOVERY_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                pass

    async def reconcile_once(self) -> int:
        recovered = await self.journal.recover_stale_executions()
        if recovered:
            logger.warning("Recovered %s expired mutation leases as UNKNOWN", recovered)
        recovered_verifications = await self.journal.recover_stale_verifications()
        if recovered_verifications:
            logger.warning(
                "Recovered %s expired verification leases as INCONCLUSIVE",
                recovered_verifications,
            )
        cutoff = timezone.now() - timedelta(seconds=settings.MUTATION_RECOVERY_GRACE_SECONDS)
        operations = await MutationOperation.filter(
            execution_status__in=[
                ExecutionStatus.UNKNOWN,
                ExecutionStatus.REPORTED_SUCCESS,
            ],
            verification_status__in=[
                VerificationStatus.PENDING,
                VerificationStatus.INCONCLUSIVE,
            ],
            updated_at__lt=cutoff,
        ).limit(settings.MUTATION_RECOVERY_BATCH_SIZE)
        reconciled = 0
        for operation in operations:
            try:
                result = await self.verifier.verify(
                    str(operation.id),
                    lease_owner=f"recovery:{uuid.uuid4()}",
                )
                if self.on_reconciled and result.get("status") in {
                    VerificationStatus.PASSED.value,
                    VerificationStatus.FAILED.value,
                    VerificationStatus.NEEDS_REVIEW.value,
                }:
                    current = await self.journal.get(str(operation.id))
                    if current is not None:
                        try:
                            await self.on_reconciled(current, result)
                        except Exception:
                            logger.exception(
                                "Failed to publish recovered mutation %s", operation.id
                            )
                reconciled += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to reconcile mutation %s", operation.id)
        return reconciled
