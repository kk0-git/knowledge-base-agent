from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ports.file_review_run_repository import FileReviewRunRepository
from services.workflows.review_runs import ReviewRunStore, empty_workspace


class ReviewRunRepositoryTests(unittest.TestCase):
    def test_create_save_load_and_patch_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewRunStore(Path(tmp))
            repo = FileReviewRunRepository(store)
            run = {
                "review_run_id": "review-test001",
                "type": "dialogue",
                "status": "done",
                "cards": [],
                "workspace": empty_workspace(),
            }
            saved = repo.save_run(run)
            self.assertEqual(saved["review_run_id"], "review-test001")

            loaded = repo.load_run("review-test001")
            assert loaded is not None
            self.assertEqual(loaded.get("workspace", {}).get("mode"), "selecting")

            patched = repo.patch_workspace(
                "review-test001",
                {
                    "mode": "dialogue_review",
                    "dialogueReviewState": {"active": True, "topics": ["MCP"], "history": []},
                },
            )
            self.assertEqual(patched["workspace"]["mode"], "dialogue_review")

            reloaded = repo.load_run("review-test001")
            assert reloaded is not None
            self.assertEqual(reloaded["workspace"]["mode"], "dialogue_review")

            snapshot = repo.snapshot("review-test001")
            assert snapshot is not None
            self.assertIn("workspace", snapshot)
            self.assertEqual(snapshot["ready_count"], 0)


if __name__ == "__main__":
    unittest.main()
