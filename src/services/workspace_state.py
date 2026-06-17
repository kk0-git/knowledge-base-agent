from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from services.rag.incremental import calculate_content_hash
from services.wiki.manager import scan_markdown_files


WORKSPACE_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WorkspaceFileRecord:
    note_path: str
    content_hash: str
    size: int
    mtime_ns: int
    rag_indexed_hash: str | None = None
    tag_extracted_hash: str | None = None

    @property
    def embed_dirty(self) -> bool:
        return self.content_hash != self.rag_indexed_hash

    @property
    def tags_dirty(self) -> bool:
        return self.content_hash != self.tag_extracted_hash


@dataclass(frozen=True)
class WorkspaceState:
    schema_version: int = WORKSPACE_STATE_SCHEMA_VERSION
    files: dict[str, WorkspaceFileRecord] | None = None

    def file_map(self) -> dict[str, WorkspaceFileRecord]:
        return dict(self.files or {})


@dataclass(frozen=True)
class WorkspaceDirtyPlan:
    current_files: dict[str, WorkspaceFileRecord]
    deleted_files: list[str]
    embed_dirty_files: list[str]
    tags_dirty_files: list[str]

    def to_summary(self) -> dict:
        return {
            "files": len(self.current_files),
            "deleted_files": len(self.deleted_files),
            "embed_dirty_files": len(self.embed_dirty_files),
            "tags_dirty_files": len(self.tags_dirty_files),
        }


class WorkspaceStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> WorkspaceState:
        if not self.path.exists():
            return WorkspaceState(files={})
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return workspace_state_from_dict(data)

    def save(self, state: WorkspaceState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(workspace_state_to_dict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


def scan_workspace_files(
    *,
    vault_root: Path,
    excluded_roots: Iterable[Path] = (),
) -> dict[str, WorkspaceFileRecord]:
    current: dict[str, WorkspaceFileRecord] = {}
    for file_path in scan_markdown_files(vault_root, excluded_roots=list(excluded_roots)):
        rel_path = file_path.relative_to(vault_root).as_posix()
        stat = file_path.stat()
        current[rel_path] = WorkspaceFileRecord(
            note_path=rel_path,
            content_hash=calculate_content_hash(file_path),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
    return current


def build_workspace_dirty_plan(
    *,
    previous_state: WorkspaceState,
    current_files: dict[str, WorkspaceFileRecord],
) -> WorkspaceDirtyPlan:
    previous_files = previous_state.file_map()
    deleted_files = sorted(set(previous_files) - set(current_files))

    merged_current: dict[str, WorkspaceFileRecord] = {}
    embed_dirty: list[str] = []
    tags_dirty: list[str] = []

    for note_path, current in sorted(current_files.items()):
        previous = previous_files.get(note_path)
        if previous and previous.content_hash == current.content_hash:
            record = replace(
                current,
                rag_indexed_hash=previous.rag_indexed_hash,
                tag_extracted_hash=previous.tag_extracted_hash,
            )
        else:
            record = current

        merged_current[note_path] = record
        if record.embed_dirty:
            embed_dirty.append(note_path)
        if record.tags_dirty:
            tags_dirty.append(note_path)

    return WorkspaceDirtyPlan(
        current_files=merged_current,
        deleted_files=deleted_files,
        embed_dirty_files=embed_dirty,
        tags_dirty_files=tags_dirty,
    )


def mark_rag_indexed(
    state: WorkspaceState,
    note_paths: Iterable[str],
) -> WorkspaceState:
    files = state.file_map()
    for note_path in note_paths:
        record = files.get(note_path)
        if record:
            files[note_path] = replace(record, rag_indexed_hash=record.content_hash)
    return WorkspaceState(files=files)


def mark_tags_extracted(
    state: WorkspaceState,
    note_paths: Iterable[str],
) -> WorkspaceState:
    files = state.file_map()
    for note_path in note_paths:
        record = files.get(note_path)
        if record:
            files[note_path] = replace(record, tag_extracted_hash=record.content_hash)
    return WorkspaceState(files=files)


def remove_deleted(
    state: WorkspaceState,
    note_paths: Iterable[str],
) -> WorkspaceState:
    files = state.file_map()
    for note_path in note_paths:
        files.pop(note_path, None)
    return WorkspaceState(files=files)


def workspace_state_to_dict(state: WorkspaceState) -> dict:
    return {
        "schema_version": state.schema_version,
        "files": {
            note_path: asdict(record)
            for note_path, record in sorted(state.file_map().items())
        },
    }


def workspace_file_record_from_dict(data: dict) -> WorkspaceFileRecord:
    return WorkspaceFileRecord(
        note_path=str(data.get("note_path", "")),
        content_hash=str(data.get("content_hash", "")),
        size=int(data.get("size", 0)),
        mtime_ns=int(data.get("mtime_ns", 0)),
        rag_indexed_hash=data.get("rag_indexed_hash"),
        tag_extracted_hash=data.get("tag_extracted_hash"),
    )


def workspace_state_from_dict(data: dict) -> WorkspaceState:
    return WorkspaceState(
        schema_version=int(data.get("schema_version", WORKSPACE_STATE_SCHEMA_VERSION)),
        files={
            str(note_path): workspace_file_record_from_dict(record)
            for note_path, record in data.get("files", {}).items()
        },
    )
