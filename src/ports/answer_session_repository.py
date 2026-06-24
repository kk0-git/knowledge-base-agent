from __future__ import annotations

from typing import Any, Protocol


class AnswerSessionRepository(Protocol):
    def load_session(self, session_id: str) -> dict[str, Any]: ...

    def append_pending_turn(
        self,
        *,
        session_id: str,
        user_content: str,
    ) -> dict[str, Any]: ...

    def complete_assistant(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        assistant_content: str,
        agent_actions: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
        process_summary: dict[str, Any] | None = None,
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
