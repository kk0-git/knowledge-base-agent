from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.workflows.conversation_schema import (
    apply_assistant_completion,
    apply_assistant_failure,
    build_pending_assistant_message,
    build_user_message,
    next_turn_id,
    normalize_message,
    normalize_session_messages,
)


class ConversationSchemaTests(unittest.TestCase):
    def test_next_turn_id_increments(self) -> None:
        messages = [
            {"turn_id": "turn-0001", "role": "user"},
            {"turn_id": "turn-0001", "role": "assistant"},
        ]
        self.assertEqual(next_turn_id(messages), "turn-0002")

    def test_build_pending_turn_pair_shares_turn_id(self) -> None:
        user = build_user_message(
            message_id="msg-0001",
            turn_id="turn-0001",
            content="hello",
            created_at="2026-06-24T00:00:00+00:00",
        )
        assistant = build_pending_assistant_message(
            message_id="msg-0002",
            turn_id="turn-0001",
            created_at="2026-06-24T00:00:00+00:00",
        )
        self.assertEqual(user["turn_id"], assistant["turn_id"])
        self.assertEqual(assistant["status"], "pending")
        self.assertIn("error", assistant)
        self.assertEqual(assistant["error_type"], "")

    def test_normalize_message_merges_flat_error_fields(self) -> None:
        msg = normalize_message(
            {
                "id": "msg-0001",
                "role": "assistant",
                "content": "",
                "status": "failed",
                "error_type": "TimeoutError",
                "error_message": "timed out",
                "retryable": True,
            }
        )
        self.assertEqual(msg["error"]["type"], "TimeoutError")
        self.assertEqual(msg["error"]["message"], "timed out")
        self.assertTrue(msg["error"]["retryable"])

    def test_normalize_session_messages_backfills_turn_ids(self) -> None:
        session = normalize_session_messages(
            {
                "mode": "interview",
                "messages": [
                    {"id": "msg-0001", "role": "user", "content": "a"},
                    {"id": "msg-0002", "role": "assistant", "content": "b"},
                    {"id": "msg-0003", "role": "user", "content": "c"},
                    {"id": "msg-0004", "role": "assistant", "content": "d"},
                ],
            }
        )
        messages = session["messages"]
        self.assertEqual(messages[0]["turn_id"], "turn-0001")
        self.assertEqual(messages[1]["turn_id"], "turn-0001")
        self.assertEqual(messages[2]["turn_id"], "turn-0002")
        self.assertEqual(messages[3]["turn_id"], "turn-0002")
        self.assertEqual(session["kind"], "interview")

    def test_apply_assistant_completion_and_failure_dual_write_error(self) -> None:
        pending = build_pending_assistant_message(
            message_id="msg-0002",
            turn_id="turn-0001",
            created_at="2026-06-24T00:00:00+00:00",
        )
        completed = apply_assistant_completion(
            pending,
            assistant_content="done",
            updated_at="2026-06-24T00:00:01+00:00",
            citations=[{"path": "note.md"}],
        )
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["citations"][0]["path"], "note.md")

        failed = apply_assistant_failure(
            pending,
            updated_at="2026-06-24T00:00:02+00:00",
            error_type="Error",
            error_message="boom",
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_message"], "boom")
        self.assertEqual(failed["error"]["message"], "boom")


if __name__ == "__main__":
    unittest.main()
