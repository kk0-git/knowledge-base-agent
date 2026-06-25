from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ports.file_review_run_repository import FileReviewRunRepository
from services.workflows.conversation_schema import project_dialogue_history
from services.workflows.review_run_service import ReviewRunService
from services.workflows.review_runs import ReviewRunStore, normalize_dialogue_run


class ReviewDialogueStorageTests(unittest.TestCase):
    def test_dialogue_run_starts_with_empty_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewRunStore(Path(tmp))
            service = ReviewRunService(
                repository=FileReviewRunRepository(store),
                profile_store=MagicMock(),
                review_cache_dir=Path(tmp),
                project_root=PROJECT_ROOT,
                executor=MagicMock(),
            )
            snapshot = service.create_dialogue_run(["Agent Memory"])
            self.assertEqual(snapshot.get("type"), "dialogue")
            self.assertEqual(snapshot.get("messages"), [])

    def test_append_and_complete_dialogue_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewRunStore(Path(tmp))
            service = ReviewRunService(
                repository=FileReviewRunRepository(store),
                profile_store=MagicMock(),
                review_cache_dir=Path(tmp),
                project_root=PROJECT_ROOT,
                executor=MagicMock(),
            )
            created = service.create_dialogue_run(["MCP"])
            run_id = str(created["review_run_id"])
            pending = service.append_dialogue_pending_turn(run_id, "第一题")
            assistant_id = str(pending["assistant_message"]["id"])
            service.complete_dialogue_assistant(
                run_id,
                assistant_message_id=assistant_id,
                assistant_content="请说明 MCP 的三个角色。",
                suggested_commits=[{"weak_point_id": "weak-abc", "action": "retry"}],
            )
            snapshot = service.snapshot(run_id) or {}
            messages = snapshot.get("messages") or []
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["turn_id"], messages[1]["turn_id"])
            self.assertEqual(messages[1]["status"], "completed")
            history = project_dialogue_history(messages)
            self.assertEqual(len(history), 2)
            workspace = snapshot.get("workspace") or {}
            dlg = workspace.get("dialogueReviewState") or {}
            self.assertEqual(dlg.get("history"), history)
            self.assertEqual(dlg.get("pendingSuggestions")[0]["weak_point_id"], "weak-abc")

    def test_migrate_workspace_history_to_messages_on_load(self) -> None:
        run = normalize_dialogue_run(
            {
                "review_run_id": "review-migrate01",
                "type": "dialogue",
                "created_at": "2026-06-24T00:00:00+00:00",
                "messages": [],
                "workspace": {
                    "dialogueReviewState": {
                        "reviewRunId": "review-migrate01",
                        "history": [
                            {"role": "user", "content": "开始"},
                            {"role": "assistant", "content": "第一题"},
                        ],
                    }
                },
            }
        )
        self.assertEqual(len(run.get("messages") or []), 2)
        self.assertEqual(run["workspace"]["dialogueReviewState"]["history"][0]["content"], "开始")
        self.assertEqual(run["workspace"]["dialogueReviewState"]["reviewRunId"], "review-migrate01")

    def test_prepare_dialogue_turn_builds_chat_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewRunStore(Path(tmp))
            service = ReviewRunService(
                repository=FileReviewRunRepository(store),
                profile_store=MagicMock(),
                review_cache_dir=Path(tmp),
                project_root=PROJECT_ROOT,
                executor=MagicMock(),
            )
            run_id = str(service.create_dialogue_run(["RAG"])["review_run_id"])
            service.append_dialogue_pending_turn(run_id, "round 1")
            service.complete_dialogue_assistant(
                run_id,
                assistant_message_id=str(service.snapshot(run_id)["messages"][1]["id"]),
                assistant_content="answer 1",
            )
            prepared = service.prepare_dialogue_turn(run_id, message="round 2")
            self.assertEqual(prepared["user_text"], "round 2")
            self.assertEqual(len(prepared["chat_history"]), 2)
            self.assertEqual(prepared["chat_history"][0]["content"], "round 1")


if __name__ == "__main__":
    unittest.main()
