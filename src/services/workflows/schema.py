from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ScopeType = Literal["search", "tag", "selected_notes", "folder", "all_vault", "current_context"]
TaskType = Literal[
    "answer",
    "synthesize_wiki",
    "audit",
    "generate_review",
    "organize_suggestions",
    "organize",
]
WritebackType = Literal["none", "wiki_file", "report_file", "new_note", "state"]


@dataclass(frozen=True)
class ScopeSpec:
    type: ScopeType
    value: str | None = None
    paths: tuple[str, ...] = ()
    top_k: int = 8
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeResult:
    scope: ScopeSpec
    notes: tuple[dict[str, Any], ...] = ()
    chunks: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextPack:
    mode: str
    scope_result: ScopeResult
    context_text: str = ""
    items: tuple[dict[str, Any], ...] = ()
    citations: tuple[dict[str, Any], ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WritebackSpec:
    type: WritebackType = "none"
    path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSpec:
    task_type: TaskType
    scope: ScopeSpec
    user_request: str = ""
    context_mode: str | None = None
    writeback: WritebackSpec = field(default_factory=WritebackSpec)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    task_type: TaskType
    scope: ScopeResult
    context: ContextPack
    output: dict[str, Any]
    writeback: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
