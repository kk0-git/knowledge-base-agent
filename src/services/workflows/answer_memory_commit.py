from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from services.workflows.answer_sessions import AnswerSessionStore
from services.workflows.interview_profile import InterviewProfileStore


def commit_answer_memory(
    *,
    session_store: AnswerSessionStore,
    profile_store: InterviewProfileStore,
    session: dict[str, Any],
    llm_client: Any | None = None,
    model: str | None = None,
    temperature: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_update, updated_session = profile_store.update_from_answer_session(
        session=session,
        llm_client=llm_client,
        model=model,
        temperature=temperature,
    )
    session_id = str(updated_session.get("session_id") or "")
    if session_id:
        session_store.save_session(updated_session)
    audit = {
        "source": "answer_memory_commit",
        "observation_count": int(profile_update.get("observation_count") or 0),
        "filtered_low_count": int(profile_update.get("filtered_low_count") or 0),
        "canonical_revision": int(profile_update.get("canonical_revision") or 0),
        "extraction_error": str(profile_update.get("extraction_error") or ""),
        "skipped": bool((profile_update.get("operations") or {}).get("skipped")),
        "updated_at": now_iso(),
    }
    profile_update = {**profile_update, "memory_commit": audit}
    return audit, profile_update


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
