import json
from unittest import TestCase

from src.api.endpoints.agent import (
    _decode_buffered_event,
    _decode_published_event,
    _has_event_history_gap,
    _is_terminal_payload,
    sse_format,
)


class SSERecoveryTests(TestCase):
    def test_event_id_round_trip(self) -> None:
        payload = {"type": "tool.result", "run_id": "run-1", "status": "running"}
        encoded = sse_format("tool.result", payload, event_id=42)

        self.assertEqual(
            _decode_published_event(encoded),
            {"id": 42, "event": "tool.result", "data": payload},
        )

    def test_buffered_event_validation(self) -> None:
        event = {"id": 9, "event": "token", "data": {"text": "a"}}
        self.assertEqual(_decode_buffered_event(json.dumps(event)), event)
        self.assertIsNone(_decode_buffered_event("not-json"))
        self.assertIsNone(_decode_buffered_event(json.dumps({"id": "9"})))

    def test_terminal_statuses(self) -> None:
        for status in (
            "completed",
            "error",
            "awaiting_approval",
            "stopped",
            "stop_partial",
            "recovered",
        ):
            self.assertTrue(_is_terminal_payload({"status": status}))
        self.assertFalse(_is_terminal_payload({"status": "running"}))

    def test_detects_trimmed_or_expired_history(self) -> None:
        self.assertTrue(
            _has_event_history_gap(
                last_event_id=20, oldest_event_id=500, latest_event_id=700
            )
        )
        self.assertTrue(
            _has_event_history_gap(
                last_event_id=20, oldest_event_id=None, latest_event_id=700
            )
        )
        self.assertFalse(
            _has_event_history_gap(
                last_event_id=500, oldest_event_id=100, latest_event_id=700
            )
        )
