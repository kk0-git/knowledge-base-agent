from __future__ import annotations

from typing import Any

from ports.session_repository import SessionRepository
from services.workflows.interview_sessions import InterviewSessionStore


class FileSessionRepository:
    def __init__(self, store: InterviewSessionStore) -> None:
        self._store = store

    def load_session(self, session_id: str) -> dict[str, Any]:
        return self._store.load_session(session_id)

    def append_pending_turn(
        self,
        *,
        session_id: str,
        user_content: str,
        interview_plan: dict[str, Any] | None = None,
        interview_state: dict[str, Any] | None = None,
        source_note_paths: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        return self._store.append_pending_turn(
            session_id=session_id,
            user_content=user_content,
            interview_plan=interview_plan,
            interview_state=interview_state,
            source_note_paths=source_note_paths,
        )

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
    ) -> dict[str, Any]:
        return self._store.complete_assistant_message(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            assistant_content=assistant_content,
            interview_plan=interview_plan,
            interview_state=interview_state,
            source_note_paths=source_note_paths,
            agent_actions=agent_actions,
            citations=citations,
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
