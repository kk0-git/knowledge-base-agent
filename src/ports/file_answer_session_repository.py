from __future__ import annotations

from typing import Any

from ports.answer_session_repository import AnswerSessionRepository
from services.workflows.answer_sessions import AnswerSessionStore


class FileAnswerSessionRepository:
    def __init__(self, store: AnswerSessionStore) -> None:
        self._store = store

    def load_session(self, session_id: str) -> dict[str, Any]:
        return self._store.load_session(session_id)

    def append_pending_turn(
        self,
        *,
        session_id: str,
        user_content: str,
    ) -> dict[str, Any]:
        return self._store.append_pending_turn(session_id=session_id, user_content=user_content)

    def complete_assistant(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        assistant_content: str,
        agent_actions: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
        process_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._store.complete_assistant_message(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            assistant_content=assistant_content,
            agent_actions=agent_actions,
            citations=citations,
            process_summary=process_summary,
        )

    def fail_assistant(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        assistant_content: str = "",
        error_type: str = "Error",
        error_message: str = "",
        retryable: bool = True,
    ) -> dict[str, Any]:
        return self._store.fail_assistant_message(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            assistant_content=assistant_content,
            error_type=error_type,
            error_message=error_message,
            retryable=retryable,
        )
