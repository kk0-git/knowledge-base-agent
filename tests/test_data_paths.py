from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.data_paths import (
    answer_sessions_root,
    app_data_root,
    interview_sessions_root,
    learner_model_path,
    migrate_legacy_runtime_dirs,
    review_cache_root,
    review_runs_root,
)


def test_app_data_roots(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    assert interview_sessions_root(root) == root / "data" / "interview-sessions"
    assert answer_sessions_root(root) == root / "data" / "answer-sessions"
    assert review_runs_root(root) == root / "data" / "review-runs"
    assert review_cache_root(root) == root / "data" / "review-cache"
    assert learner_model_path(root) == root / "data" / "profile" / "learner_model.json"
    assert app_data_root(root) == root / "data"


def test_migrate_legacy_runtime_dirs(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    legacy_run = root / "review-runs" / "abc.json"
    legacy_run.parent.mkdir(parents=True)
    legacy_run.write_text(json.dumps({"review_run_id": "abc"}), encoding="utf-8")
    legacy_answer = root / "answer-sessions" / "2026-06" / "sess.json"
    legacy_answer.parent.mkdir(parents=True)
    legacy_answer.write_text("{}", encoding="utf-8")

    migrated = migrate_legacy_runtime_dirs(root)

    assert set(migrated) == {"review-runs", "answer-sessions"}
    assert (review_runs_root(root) / "abc.json").is_file()
    assert (answer_sessions_root(root) / "2026-06" / "sess.json").is_file()
    assert not legacy_run.exists()
    assert not (root / "answer-sessions").exists()


def test_migrate_skips_conflicting_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = review_runs_root(root) / "dup.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"review_run_id":"new"}', encoding="utf-8")
    legacy = root / "review-runs" / "dup.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"review_run_id":"old"}', encoding="utf-8")

    migrated = migrate_legacy_runtime_dirs(root)

    assert migrated == ["review-runs"]
    assert target.read_text(encoding="utf-8") == '{"review_run_id":"new"}'
    assert not legacy.exists()
    assert not (root / "review-runs").exists()
