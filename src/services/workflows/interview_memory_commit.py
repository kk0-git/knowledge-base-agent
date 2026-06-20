from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from services.workflows.interview_profile import InterviewProfileStore
from services.workflows.interview_sessions import InterviewSessionStore


def commit_interview_memory(
    *,
    session_store: InterviewSessionStore,
    profile_store: InterviewProfileStore,
    session: dict[str, Any],
    reviews: list[dict[str, Any]],
    llm_client: Any | None = None,
    model: str | None = None,
    temperature: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    session_id = str(session.get("session_id") or "")
    memory_signals = session_store.list_memory_signals(session_id=session_id) if session_id else []
    observation_drafts = session_store.list_observation_drafts(session_id=session_id) if session_id else []
    enriched_reviews = append_memory_review(
        reviews=reviews,
        memory_signals=memory_signals,
        observation_drafts=observation_drafts,
    )
    enriched_session = deepcopy(session)
    enriched_session.setdefault("memory", {})
    enriched_session["memory"]["commit_bridge"] = {
        "profile_signal_count": len(memory_signals),
        "observation_draft_count": len(observation_drafts),
        "review_count": len(reviews),
        "agent_trace_count": len(((session.get("agent") or {}).get("traces") or [])),
        "created_at": now_iso(),
    }
    final_review, profile_update = profile_store.update_from_session(
        session=enriched_session,
        reviews=enriched_reviews,
        llm_client=llm_client,
        model=model,
        temperature=temperature,
    )
    audit = build_commit_audit(
        profile_update=profile_update,
        memory_signals=memory_signals,
        observation_drafts=observation_drafts,
        original_review_count=len(reviews),
        enriched_review_count=len(enriched_reviews),
    )
    profile_update = {
        **profile_update,
        "source": "commit_bridge",
        "base_source": profile_update.get("source", ""),
        "commit_bridge": audit,
    }
    final_review = {
        **final_review,
        "memory_commit": audit,
    }
    if session_id:
        session_store.append_trace_event(
            session_id=session_id,
            event="memory_commit",
            summary=(
                "memory commit: "
                f"{audit['profile_signal_count']} signal(s), "
                f"{audit['observation_draft_count']} draft(s), "
                f"{audit['operation_counts'].get('added', 0)} ADD"
            ),
            details=audit,
        )
    return final_review, profile_update


def append_memory_review(
    *,
    reviews: list[dict[str, Any]],
    memory_signals: list[dict[str, Any]],
    observation_drafts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = [deepcopy(review) for review in reviews]
    synthetic_signals = [normalize_signal(signal) for signal in memory_signals if isinstance(signal, dict)]
    synthetic_signals.extend(normalize_draft_as_signal(draft) for draft in observation_drafts if isinstance(draft, dict))
    synthetic_signals = [signal for signal in synthetic_signals if signal.get("point") and signal.get("evidence")]
    if not synthetic_signals:
        return result
    result.append(
        {
            "turn_id": "memory-commit-bridge",
            "user_message_id": "",
            "assistant_message_id": "",
            "feedback": {
                "coach_note": "Session-level memory signals collected by Agent tools.",
                "covered": [],
                "gaps": [],
                "thinking_framework": "",
                "interviewer_followup_note": "",
            },
            "expression_example": "",
            "context_note_paths": dedupe_paths(
                path
                for signal in synthetic_signals
                for path in signal.get("context_note_paths", [])
            ),
            "profile_signals": synthetic_signals,
            "status": "completed",
            "created_at": now_iso(),
            "source": "commit_bridge",
        }
    )
    return result


def normalize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": str(signal.get("type") or signal.get("signal_type") or "possible_weak_point").strip(),
        "topic": str(signal.get("topic") or "").strip(),
        "planned_layer": str(signal.get("planned_layer") or "").strip(),
        "category": str(signal.get("category") or "knowledge_gap").strip(),
        "scope_suggestion": str(signal.get("scope_suggestion") or signal.get("scope") or "domain").strip(),
        "point": str(signal.get("point") or signal.get("summary") or signal.get("weak_point_ref") or "").strip(),
        "weak_point_ref": str(signal.get("weak_point_ref") or "").strip(),
        "evidence": str(signal.get("evidence") or "").strip(),
        "confidence": str(signal.get("confidence") or "medium").strip(),
        "context_note_paths": dedupe_paths(signal.get("context_note_paths") or []),
    }


def normalize_draft_as_signal(draft: dict[str, Any]) -> dict[str, Any]:
    return normalize_signal(
        {
            **draft,
            "type": draft.get("type") or "possible_weak_point",
            "scope_suggestion": draft.get("scope_suggestion") or draft.get("scope"),
            "confidence": draft.get("confidence") or "medium",
        }
    )


def build_commit_audit(
    *,
    profile_update: dict[str, Any],
    memory_signals: list[dict[str, Any]],
    observation_drafts: list[dict[str, Any]],
    original_review_count: int,
    enriched_review_count: int,
) -> dict[str, Any]:
    operations = profile_update.get("operations", {}) if isinstance(profile_update, dict) else {}
    return {
        "source": "commit_bridge",
        "base_source": profile_update.get("source", "") if isinstance(profile_update, dict) else "",
        "profile_signal_count": len(memory_signals),
        "observation_draft_count": len(observation_drafts),
        "consumed_signal_count": len(memory_signals),
        "consumed_draft_count": len(observation_drafts),
        "ignored_signal_count": 0,
        "original_review_count": original_review_count,
        "enriched_review_count": enriched_review_count,
        "operation_counts": {
            "added": len(operations.get("added") or []),
            "updated": len(operations.get("updated") or []),
            "partial": len(operations.get("partial") or []),
            "improved": len(operations.get("improved") or []),
        },
        "updated_at": now_iso(),
    }


def dedupe_paths(value: Any) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for item in value or []:
        path = str(item or "").replace("\\", "/").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
