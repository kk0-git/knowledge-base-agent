from __future__ import annotations

import shutil
from pathlib import Path

APP_DATA_DIRNAME = "data"

LEGACY_RUNTIME_DIRS = (
    "interview-sessions",
    "answer-sessions",
    "review-runs",
    "review-cache",
)


def app_data_root(project_root: Path | str) -> Path:
    return Path(project_root) / APP_DATA_DIRNAME


def interview_sessions_root(project_root: Path | str) -> Path:
    return app_data_root(project_root) / "interview-sessions"


def answer_sessions_root(project_root: Path | str) -> Path:
    return app_data_root(project_root) / "answer-sessions"


def review_runs_root(project_root: Path | str) -> Path:
    return app_data_root(project_root) / "review-runs"


def review_cache_root(project_root: Path | str) -> Path:
    return app_data_root(project_root) / "review-cache"


def ensure_app_data_dirs(project_root: Path | str) -> Path:
    root = app_data_root(project_root)
    for path in (
        interview_sessions_root(project_root),
        answer_sessions_root(project_root),
        review_runs_root(project_root),
        review_cache_root(project_root),
    ):
        path.mkdir(parents=True, exist_ok=True)
    return root


def _merge_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            _merge_tree(item, target)
            continue
        if target.exists():
            item.unlink(missing_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), str(target))
    try:
        source.rmdir()
    except OSError:
        pass


def migrate_legacy_runtime_dirs(project_root: Path | str) -> list[str]:
    """Move runtime session dirs from repo root into data/ once."""
    root = Path(project_root)
    ensure_app_data_dirs(root)
    migrated: list[str] = []
    for name in LEGACY_RUNTIME_DIRS:
        legacy = root / name
        if not legacy.exists():
            continue
        target = app_data_root(root) / name
        if not any(legacy.iterdir()) if legacy.is_dir() else False:
            try:
                legacy.rmdir()
            except OSError:
                pass
            continue
        _merge_tree(legacy, target)
        migrated.append(name)
    return migrated
