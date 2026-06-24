from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentTurnInput:
    query: str = ""
    chat_mode: str = "answer"
    command: str = "auto"
    scope_type: str = "tag"
    scope_value: str | None = None
    scope_paths: list[str] = field(default_factory=list)
    chat_history: list[dict[str, str]] = field(default_factory=list)
    notes_top_k: int = 5
    dense_top_k: int = 50
    hybrid_bm25_top_k: int = 50
    rrf_k: int = 60
    online_provider: str | None = None
    strict_evidence: bool = False
    speculative_notes_search: bool = True
    interview_max_chars_per_note: int = 4000
    interview_max_context_chars: int = 24000
    interview_plan: dict[str, Any] | None = None
    interview_state: dict[str, Any] | None = None
    session_id: str | None = None
    assistant_message_id: str | None = None
    source_note_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AgentTurnInput":
        data = dict(payload or {})
        return cls(
            query=str(data.get("query") or ""),
            chat_mode=str(data.get("chat_mode") or "answer"),
            command=str(data.get("command") or "auto"),
            scope_type=str(data.get("scope_type") or "tag"),
            scope_value=data.get("scope_value"),
            scope_paths=[str(item) for item in data.get("scope_paths") or [] if str(item).strip()],
            chat_history=list(data.get("chat_history") or []),
            notes_top_k=int(data.get("notes_top_k") or 5),
            dense_top_k=int(data.get("dense_top_k") or 50),
            hybrid_bm25_top_k=int(data.get("hybrid_bm25_top_k") or 50),
            rrf_k=int(data.get("rrf_k") or 60),
            online_provider=data.get("online_provider"),
            strict_evidence=bool(data.get("strict_evidence")),
            speculative_notes_search=bool(data.get("speculative_notes_search", True)),
            interview_max_chars_per_note=int(data.get("interview_max_chars_per_note") or 4000),
            interview_max_context_chars=int(data.get("interview_max_context_chars") or 24000),
            interview_plan=data.get("interview_plan"),
            interview_state=data.get("interview_state"),
            session_id=data.get("session_id"),
            assistant_message_id=data.get("assistant_message_id"),
            source_note_paths=[str(item) for item in data.get("source_note_paths") or [] if str(item).strip()],
        )


@dataclass
class AgentTurnResult:
    answer_text: str = ""
    interview_plan: dict[str, Any] | None = None
    interview_state: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    agent_actions: list[dict[str, Any]] = field(default_factory=list)
    source_note_paths: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurnRunnerDeps:
    vault_path: Any
    wiki_state_path: Any
    wiki_dir: Any
    project_root: Any
    overview_note_threshold: int
    interview_session_store: Any
    interview_profile_store: Any
    llm_model: str
    llm_temperature: float
    build_agent_runtime: Callable[[Any], Any]
    build_interview_rag_manager: Callable[[AgentTurnInput], Any]
    build_librarian_rag_manager: Callable[[AgentTurnInput], Any]
    resolve_librarian_scope: Callable[[AgentTurnInput], tuple[Any, tuple[str, ...], dict[str, Any]]]
    librarian_online_enabled: Callable[[AgentTurnInput], bool]
    prewarm_interview_rag: Callable[[AgentTurnInput], None] | None = None
    load_session_state: Callable[[str], dict[str, Any] | None] | None = None
