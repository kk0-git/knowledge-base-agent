from __future__ import annotations

from typing import Any, Protocol


class SessionRepository(Protocol):
    def load_session(self, session_id: str) -> dict[str, Any]: ...

    def append_pending_turn(
        self,
        *,
        session_id: str,
        user_content: str,
        interview_plan: dict[str, Any] | None = None,
        interview_state: dict[str, Any] | None = None,
        source_note_paths: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]: ...

    def complete_assistant(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        assistant_content: str,
        interview_plan: dict[str, Any] | None = None,
        interview_state: dict[str, Any] | None = None,
        source_note_paths: list[str] | tuple[str, ...] | None = None,
        agent_actions: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...

    def fail_assistant(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        assistant_content: str = "",
        error_type: str = "Error",
        error_message: str = "",
        retryable: bool = True,
    ) -> dict[str, Any]: ...
