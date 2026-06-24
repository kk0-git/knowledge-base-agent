from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.workflows.review_practice import REVIEW_PROMPT_VERSION


RUN_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_workspace() -> dict[str, Any]:
    return {
        "mode": "selecting",
        "selectionState": None,
        "cardReviewState": None,
        "dialogueReviewState": None,
    }


class ReviewRunStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_path(self, review_run_id: str) -> Path:
        return self.root / f"{review_run_id}.json"

    def load_run(self, review_run_id: str) -> dict[str, Any] | None:
        path = self.run_path(review_run_id)
        if not path.exists():
            return None
        run = json.loads(path.read_text(encoding="utf-8-sig"))
        if "workspace" not in run or run["workspace"] is None:
            run["workspace"] = empty_workspace()
        return run

    def save_run(self, run: dict[str, Any]) -> dict[str, Any]:
        review_run_id = str(run.get("review_run_id") or "")
        if not review_run_id:
            raise ValueError("review_run_id is required")
        run["schema_version"] = RUN_SCHEMA_VERSION
        run["updated_at"] = utc_now_iso()
        if "workspace" not in run or run["workspace"] is None:
            run["workspace"] = empty_workspace()
        path = self.run_path(review_run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        return run

    def patch_workspace(self, review_run_id: str, workspace: dict[str, Any]) -> dict[str, Any]:
        run = self.load_run(review_run_id)
        if run is None:
            raise FileNotFoundError(f"review run not found: {review_run_id}")
        current = run.get("workspace") or empty_workspace()
        merged = {**current, **(workspace or {})}
        run["workspace"] = merged
        return self.save_run(run)

    @staticmethod
    def snapshot_payload(run: dict[str, Any]) -> dict[str, Any]:
        cards = [dict(card) for card in run.get("cards", [])]
        ready_count = len([card for card in cards if card.get("status") == "ready"])
        failed_count = len([card for card in cards if card.get("status") == "failed"])
        pending_count = len([card for card in cards if card.get("status") == "pending"])
        payload = {
            key: value
            for key, value in run.items()
            if key not in {"cards", "weak_points", "card_weak_points", "workspace"}
        }
        payload["cards"] = cards
        payload["ready_count"] = ready_count
        payload["failed_count"] = failed_count
        payload["pending_count"] = pending_count
        payload["prompt_version"] = str(run.get("prompt_version") or REVIEW_PROMPT_VERSION)
        workspace = run.get("workspace")
        if workspace is not None:
            payload["workspace"] = workspace
        return payload
