from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ports.file_answer_session_repository import FileAnswerSessionRepository
from services.workflows.answer_sessions import AnswerSessionStore


class AnswerSessionRepositoryTests(unittest.TestCase):
    def test_create_pending_complete_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AnswerSessionStore(Path(tmp))
            repo = FileAnswerSessionRepository(store)
            session = store.create_session(scope_type="folder", scope_value="demo")
            pending = repo.append_pending_turn(session_id=session["session_id"], user_content="hello")
            self.assertEqual(pending["assistant_message"]["status"], "pending")
            self.assertEqual(pending["user_message"]["turn_id"], pending["assistant_message"]["turn_id"])
            completed = repo.complete_assistant(
                session_id=session["session_id"],
                assistant_message_id=pending["assistant_message"]["id"],
                assistant_content="world",
                citations=[{"path": "note.md"}],
            )
            self.assertEqual(completed["assistant_message"]["status"], "completed")
            self.assertEqual(completed["assistant_message"]["citations"][0]["path"], "note.md")
            archived = store.archive_session(session["session_id"])
            self.assertEqual(archived["status"], "archived")

    def test_fail_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AnswerSessionStore(Path(tmp))
            repo = FileAnswerSessionRepository(store)
            session = store.create_session(scope_type="all", scope_value=None)
            pending = repo.append_pending_turn(session_id=session["session_id"], user_content="hello")
            failed = repo.fail_assistant(
                session_id=session["session_id"],
                assistant_message_id=pending["assistant_message"]["id"],
                error_message="boom",
            )
            self.assertEqual(failed["assistant_message"]["status"], "failed")
            self.assertIn("boom", failed["assistant_message"]["error_message"])


if __name__ == "__main__":
    unittest.main()
