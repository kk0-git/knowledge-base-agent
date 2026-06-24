from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ports.file_session_repository import FileSessionRepository
from services.workflows.interview_sessions import InterviewSessionStore


class FileSessionRepositoryTests(unittest.TestCase):
    def test_append_pending_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = InterviewSessionStore(Path(tmp))
            repo = FileSessionRepository(store)
            session = store.create_session(source_type="folder", source_value="demo")
            pending = repo.append_pending_turn(
                session_id=session["session_id"],
                user_content="hello",
            )
            self.assertEqual(pending["assistant_message"]["status"], "pending")
            completed = repo.complete_assistant(
                session_id=session["session_id"],
                assistant_message_id=pending["assistant_message"]["id"],
                assistant_content="world",
            )
            self.assertEqual(completed["assistant_message"]["status"], "completed")
            self.assertEqual(completed["assistant_message"]["content"], "world")

    def test_fail_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = InterviewSessionStore(Path(tmp))
            repo = FileSessionRepository(store)
            session = store.create_session(source_type="folder", source_value="demo")
            pending = repo.append_pending_turn(
                session_id=session["session_id"],
                user_content="hello",
            )
            failed = repo.fail_assistant(
                session_id=session["session_id"],
                assistant_message_id=pending["assistant_message"]["id"],
                error_message="boom",
            )
            self.assertEqual(failed["assistant_message"]["status"], "failed")
            self.assertIn("boom", failed["assistant_message"]["error_message"])


if __name__ == "__main__":
    unittest.main()
