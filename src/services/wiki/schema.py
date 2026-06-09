from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


WIKI_STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class NoteTagRecord:
    note_path: str
    content_hash: str
    user_tags: list[str] = field(default_factory=list)
    llm_tags: list[str] = field(default_factory=list)
    candidate_tags: list[str] = field(default_factory=list)
    title: str = ""
    tagged_at: str | None = None
    status: str = "clean"
    error: str | None = None


@dataclass(frozen=True)
class WikiTagRecord:
    tag: str
    source_paths: list[str] = field(default_factory=list)
    candidate_kind: str = "tag_derived"
    evidence: dict[str, list[str]] = field(default_factory=dict)
    wiki_path: str | None = None
    wiki_policy: str = "generate"
    wiki_policy_source: str = "auto"
    dirty: bool = False
    generated_at: str | None = None
    updated_at: str | None = None
    source_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WikiState:
    schema_version: int = WIKI_STATE_SCHEMA_VERSION
    files: dict[str, NoteTagRecord] = field(default_factory=dict)
    tags: dict[str, WikiTagRecord] = field(default_factory=dict)


@dataclass(frozen=True)
class TagExtractionResult:
    tags: list[str]
    candidate_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    raw_response: str = ""


@dataclass(frozen=True)
class TagMergeProposal:
    source_tag: str
    target_tag: str
    reason: str


def note_tag_record_to_dict(record: NoteTagRecord) -> dict[str, Any]:
    return asdict(record)


def wiki_tag_record_to_dict(record: WikiTagRecord) -> dict[str, Any]:
    return asdict(record)


def wiki_state_to_dict(state: WikiState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "files": {
            note_path: note_tag_record_to_dict(record)
            for note_path, record in sorted(state.files.items())
        },
        "tags": {
            tag: wiki_tag_record_to_dict(record)
            for tag, record in sorted(state.tags.items())
        },
    }


def note_tag_record_from_dict(data: dict[str, Any]) -> NoteTagRecord:
    return NoteTagRecord(
        note_path=str(data.get("note_path", "")),
        content_hash=str(data.get("content_hash", "")),
        user_tags=[str(tag) for tag in data.get("user_tags", [])],
        llm_tags=[str(tag) for tag in data.get("llm_tags", [])],
        candidate_tags=[str(tag) for tag in data.get("candidate_tags", [])],
        title=str(data.get("title", "")),
        tagged_at=data.get("tagged_at"),
        status=str(data.get("status", "clean")),
        error=data.get("error"),
    )


def wiki_tag_record_from_dict(data: dict[str, Any]) -> WikiTagRecord:
    return WikiTagRecord(
        tag=str(data.get("tag", "")),
        source_paths=[str(path) for path in data.get("source_paths", [])],
        candidate_kind=str(data.get("candidate_kind", "tag_derived")),
        evidence={
            str(source): [str(path) for path in paths]
            for source, paths in data.get("evidence", {}).items()
        },
        wiki_path=data.get("wiki_path"),
        wiki_policy=str(data.get("wiki_policy", "generate")),
        wiki_policy_source=str(data.get("wiki_policy_source", "auto")),
        dirty=bool(data.get("dirty", False)),
        generated_at=data.get("generated_at"),
        updated_at=data.get("updated_at"),
        source_hashes={
            str(path): str(content_hash)
            for path, content_hash in data.get("source_hashes", {}).items()
        },
    )


def wiki_state_from_dict(data: dict[str, Any]) -> WikiState:
    return WikiState(
        schema_version=int(data.get("schema_version", WIKI_STATE_SCHEMA_VERSION)),
        files={
            str(note_path): note_tag_record_from_dict(record)
            for note_path, record in data.get("files", {}).items()
        },
        tags={
            str(tag): wiki_tag_record_from_dict(record)
            for tag, record in data.get("tags", {}).items()
        },
    )
