from __future__ import annotations

from typing import Any

from ports.review_run_repository import ReviewRunRepository
from services.workflows.review_runs import ReviewRunStore


class FileReviewRunRepository:
    def __init__(self, store: ReviewRunStore) -> None:
        self._store = store

    def load_run(self, review_run_id: str) -> dict[str, Any] | None:
        return self._store.load_run(review_run_id)

    def save_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return self._store.save_run(run)

    def patch_workspace(self, review_run_id: str, workspace: dict[str, Any]) -> dict[str, Any]:
        return self._store.patch_workspace(review_run_id, workspace)

    def snapshot(self, review_run_id: str) -> dict[str, Any] | None:
        run = self._store.load_run(review_run_id)
        if run is None:
            return None
        return ReviewRunStore.snapshot_payload(run)
