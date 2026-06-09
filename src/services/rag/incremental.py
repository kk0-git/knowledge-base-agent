from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from services.rag.chunker import ChunkerConfig


VECTOR_INDEX_SCHEMA_VERSION = 1
CHUNKER_VERSION = "heading-line-v1"
EMBEDDING_TEXT_VERSION = "file-heading-content-v1"


@dataclass(frozen=True)
class FileState:
    note_path: str
    file_path: Path
    content_hash: str
    size: int
    mtime: float

    def to_metadata(self, chunk_ids: list[str]) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "size": self.size,
            "mtime": self.mtime,
            "chunk_ids": list(chunk_ids),
        }


@dataclass(frozen=True)
class IncrementalPlan:
    added: list[str]
    modified: list[str]
    deleted: list[str]
    unchanged: list[str]
    current_files: dict[str, FileState]

    @property
    def changed(self) -> list[str]:
        return sorted(set(self.added) | set(self.modified))

    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)


def build_index_config(
    *,
    embedding_model: str,
    embedding_provider: str = "local",
    embedding_batch_size: int = 32,
    max_seq_length: int | None = None,
    chunker_config: ChunkerConfig,
) -> dict[str, Any]:
    return {
        "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_batch_size": embedding_batch_size,
        "max_seq_length": max_seq_length,
        "chunker_version": CHUNKER_VERSION,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "max_chunk_chars": chunker_config.max_chunk_chars,
        "target_chunk_chars": chunker_config.target_chunk_chars,
        "min_chunk_chars": chunker_config.min_chunk_chars,
        "chunk_overlap": chunker_config.chunk_overlap,
        "chunk_split_mode": chunker_config.chunk_split_mode,
        "strip_code_blocks": chunker_config.strip_code_blocks,
    }


def index_config_matches(
    old_config: dict[str, Any],
    current_config: dict[str, Any],
) -> bool:
    return old_config == current_config


def calculate_content_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_current_file_states(
    *,
    vault_root: Path,
    markdown_files: list[Path],
) -> dict[str, FileState]:
    states: dict[str, FileState] = {}

    for file_path in markdown_files:
        relative_path = file_path.relative_to(vault_root).as_posix()
        stat = file_path.stat()
        states[relative_path] = FileState(
            note_path=relative_path,
            file_path=file_path,
            content_hash=calculate_content_hash(file_path),
            size=stat.st_size,
            mtime=stat.st_mtime,
        )

    return states


def plan_incremental_update(
    *,
    old_files: dict[str, dict[str, Any]],
    current_files: dict[str, FileState],
) -> IncrementalPlan:
    old_paths = set(old_files)
    current_paths = set(current_files)

    added = sorted(current_paths - old_paths)
    deleted = sorted(old_paths - current_paths)

    modified: list[str] = []
    unchanged: list[str] = []
    for note_path in sorted(old_paths & current_paths):
        old_hash = old_files.get(note_path, {}).get("content_hash")
        current_hash = current_files[note_path].content_hash
        if old_hash == current_hash:
            unchanged.append(note_path)
        else:
            modified.append(note_path)

    return IncrementalPlan(
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
        current_files=current_files,
    )


def metadata_chunk_ids(metadata: dict[str, Any]) -> list[str]:
    chunk_ids = metadata.get("chunk_ids", [])
    if not isinstance(chunk_ids, list):
        return []
    return [str(chunk_id) for chunk_id in chunk_ids]


def plan_to_dict(plan: IncrementalPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["current_files"] = sorted(plan.current_files)
    return payload
