from copy import deepcopy
from unittest import TestCase

from src.api.models.mutation import ExecutionStatus, VerificationStatus
from src.api.services.conversation_persistence import (
    mark_active_tool_segments_cancelled,
    project_mutation_operations,
)


class CancellationStateTests(TestCase):
    def setUp(self) -> None:
        self.messages = [
            {"type": "user", "content": "run a tool"},
            {
                "id": "assistant-1",
                "type": "assistant",
                "segments": [
                    {
                        "kind": "tool",
                        "id": "call-active",
                        "toolExecution": {
                            "call_id": "call-active",
                            "tool": "wait_for_x_seconds",
                            "status": "executing",
                            "run_id": "run-1",
                        },
                    },
                    {
                        "kind": "tool",
                        "id": "call-other-run",
                        "toolExecution": {
                            "call_id": "call-other-run",
                            "tool": "k8s_get",
                            "status": "executing",
                            "run_id": "run-2",
                        },
                    },
                    {
                        "kind": "tool",
                        "id": "call-complete",
                        "toolExecution": {
                            "call_id": "call-complete",
                            "tool": "memory_search",
                            "status": "completed",
                            "run_id": "run-1",
                        },
                    },
                ],
            },
        ]

    def test_only_active_tools_from_requested_run_are_cancelled(self) -> None:
        messages = deepcopy(self.messages)
        cancelled = mark_active_tool_segments_cancelled(messages, "run-1", "Cancelled by user")

        self.assertEqual([item["call_id"] for item in cancelled], ["call-active"])
        statuses = {
            segment["id"]: segment["toolExecution"]["status"]
            for segment in messages[-1]["segments"]
        }
        self.assertEqual(statuses["call-active"], "cancelled")
        self.assertEqual(statuses["call-other-run"], "executing")
        self.assertEqual(statuses["call-complete"], "completed")

    def test_legacy_active_tool_without_run_id_is_cancelled(self) -> None:
        messages = deepcopy(self.messages)
        messages[-1]["segments"][0]["toolExecution"].pop("run_id")

        cancelled = mark_active_tool_segments_cancelled(messages, "run-1", "Cancelled by user")

        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["status"], "cancelled")
        self.assertEqual(cancelled[0]["result"][0]["text"], "Cancelled by user")

    def test_no_assistant_turn_is_a_noop(self) -> None:
        messages = [{"type": "user", "content": "hello"}]
        self.assertEqual(mark_active_tool_segments_cancelled(messages, "run-1", "Cancelled"), [])

    def test_journal_projection_repairs_stale_cancelled_card(self) -> None:
        operation = type(
            "Operation",
            (),
            {
                "id": "operation-1",
                "call_id": "call-active",
                "execution_status": ExecutionStatus.UNKNOWN,
                "verification_status": VerificationStatus.PASSED,
                "verification_result": {},
            },
        )()
        projected = project_mutation_operations(deepcopy(self.messages), [operation])
        execution = projected[-1]["segments"][0]["toolExecution"]
        self.assertEqual(execution["status"], "completed")
        self.assertEqual(execution["verification_status"], "passed")
        self.assertEqual(execution["operation_id"], "operation-1")
        self.assertFalse(execution["requires_verification"])

    def test_projection_exposes_safe_partial_commit_recovery_metadata(self) -> None:
        operation = type(
            "Operation",
            (),
            {
                "id": "operation-2",
                "call_id": "call-active",
                "execution_status": ExecutionStatus.UNKNOWN,
                "verification_status": VerificationStatus.FAILED,
                "verification_result": {
                    "reason": "partial_commit_detected",
                    "partial_commit_detected": True,
                    "targets": [{"sensitive": "must-not-be-projected"}],
                },
            },
        )()

        projected = project_mutation_operations(deepcopy(self.messages), [operation])
        execution = projected[-1]["segments"][0]["toolExecution"]

        self.assertEqual(execution["status"], "error")
        self.assertEqual(execution["verification_reason"], "partial_commit_detected")
        self.assertTrue(execution["partial_commit_detected"])
        self.assertNotIn("verification_result", execution)
        self.assertNotIn("targets", execution)

    def test_projection_does_not_expose_unrecognised_verification_reason(self) -> None:
        operation = type(
            "Operation",
            (),
            {
                "id": "operation-3",
                "call_id": "call-active",
                "execution_status": ExecutionStatus.UNKNOWN,
                "verification_status": VerificationStatus.INCONCLUSIVE,
                "verification_result": {"reason": "customer-secret-value"},
            },
        )()

        projected = project_mutation_operations(deepcopy(self.messages), [operation])
        execution = projected[-1]["segments"][0]["toolExecution"]

        self.assertNotIn("verification_reason", execution)
