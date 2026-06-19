from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SESSION_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str | None, fallback: str = "interview") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\\/:*?\"<>|\s]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:48] or fallback


def make_session_id(source_value: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    slug = slugify(source_value, fallback="interview")
    return f"{timestamp}-{slug}-{suffix}"


def build_context(
    *,
    source_type: str,
    source_value: str | None,
    source_paths: list[str] | tuple[str, ...] | None = None,
    source_note_paths: list[str] | tuple[str, ...] | None = None,
    interview_plan_signature: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    note_paths = [str(path) for path in (source_note_paths or []) if str(path).strip()]
    return {
        "source_type": source_type,
        "source_value": source_value,
        "source_paths": [str(path) for path in (source_paths or []) if str(path).strip()],
        "source_note_paths": note_paths,
        "source_note_count": len(note_paths),
        "interview_plan_signature": interview_plan_signature,
        "extra": extra or {},
    }


class InterviewSessionStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def create_session(
        self,
        *,
        source_type: str,
        source_value: str | None,
        source_paths: list[str] | tuple[str, ...] | None = None,
        source_note_paths: list[str] | tuple[str, ...] | None = None,
        interview_plan: dict[str, Any] | None = None,
        interview_state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        session_id = make_session_id(source_value)
        session = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": session_id,
            "mode": "interview",
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "ended_at": None,
            "end_error": None,
            "context": build_context(
                source_type=source_type,
                source_value=source_value,
                source_paths=source_paths,
                source_note_paths=source_note_paths,
                extra=extra,
            ),
            "interview_plan": interview_plan or None,
            "interview_state": interview_state or None,
            "messages": [],
            "final_review": None,
            "profile_update": None,
        }
        self.save_session(session)
        self.save_reviews({"schema_version": REVIEW_SCHEMA_VERSION, "session_id": session_id, "reviews": []})
        return session

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.root.glob("*/*.json"):
            if path.name.endswith(".reviews.json"):
                continue
            try:
                session = self.load_session(path.stem)
            except Exception:
                continue
            context = session.get("context", {})
            messages = session.get("messages", [])
            sessions.append(
                {
                    "session_id": session.get("session_id"),
                    "status": session.get("status"),
                    "created_at": session.get("created_at"),
                    "updated_at": session.get("updated_at"),
                    "ended_at": session.get("ended_at"),
                    "source_type": context.get("source_type"),
                    "source_value": context.get("source_value"),
                    "source_note_count": context.get("source_note_count", 0),
                    "message_count": len(messages),
                    "topic_label": self._infer_topic_label(session),
                }
            )
        sessions.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return sessions[:limit]

    def load_session_bundle(self, session_id: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        reviews = self.load_reviews(session_id)
        return {"session": session, "reviews": reviews.get("reviews", [])}

    def load_session(self, session_id: str) -> dict[str, Any]:
        path = self.session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"interview session not found: {session_id}")
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def save_session(self, session: dict[str, Any]) -> None:
        session_id = str(session["session_id"])
        path = self.session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_reviews(self, session_id: str) -> dict[str, Any]:
        path = self.reviews_path(session_id)
        if not path.exists():
            return {"schema_version": REVIEW_SCHEMA_VERSION, "session_id": session_id, "reviews": []}
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def save_reviews(self, reviews: dict[str, Any]) -> None:
        session_id = str(reviews["session_id"])
        path = self.reviews_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_turn(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
        interview_plan: dict[str, Any] | None = None,
        interview_state: dict[str, Any] | None = None,
        source_note_paths: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        if session.get("status") not in {"active", "end_failed"}:
            raise ValueError(f"cannot append to session with status {session.get('status')}")

        now = utc_now()
        messages = session.setdefault("messages", [])
        user_message = {
            "id": self._next_message_id(messages),
            "role": "user",
            "content": user_content,
            "created_at": now,
        }
        assistant_message = {
            "id": self._next_message_id([*messages, user_message]),
            "role": "assistant",
            "content": assistant_content,
            "created_at": now,
        }
        messages.extend([user_message, assistant_message])

        if interview_plan is not None:
            session["interview_plan"] = interview_plan
        if interview_state is not None:
            session["interview_state"] = interview_state
        if source_note_paths:
            context = session.setdefault("context", {})
            existing = list(context.get("source_note_paths") or [])
            for path in source_note_paths:
                text = str(path).strip()
                if text and text not in existing:
                    existing.append(text)
            context["source_note_paths"] = existing
            context["source_note_count"] = len(existing)

        session["status"] = "active"
        session["updated_at"] = now
        session["end_error"] = None
        self.save_session(session)
        return {"user_message": user_message, "assistant_message": assistant_message, "session": session}

    def append_pending_turn(
        self,
        *,
        session_id: str,
        user_content: str,
        interview_plan: dict[str, Any] | None = None,
        interview_state: dict[str, Any] | None = None,
        source_note_paths: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        if session.get("status") not in {"active", "end_failed"}:
            raise ValueError(f"cannot append to session with status {session.get('status')}")

        now = utc_now()
        messages = session.setdefault("messages", [])
        user_message = {
            "id": self._next_message_id(messages),
            "role": "user",
            "content": user_content,
            "status": "completed",
            "created_at": now,
        }
        assistant_message = {
            "id": self._next_message_id([*messages, user_message]),
            "role": "assistant",
            "content": "",
            "status": "pending",
            "error_type": "",
            "error_message": "",
            "retryable": False,
            "created_at": now,
            "updated_at": now,
        }
        messages.extend([user_message, assistant_message])

        self._update_session_context(
            session,
            interview_plan=interview_plan,
            interview_state=interview_state,
            source_note_paths=source_note_paths,
        )
        session["status"] = "active"
        session["updated_at"] = now
        session["end_error"] = None
        self.save_session(session)
        return {"user_message": user_message, "assistant_message": assistant_message, "session": session}

    def complete_assistant_message(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        assistant_content: str,
        interview_plan: dict[str, Any] | None = None,
        interview_state: dict[str, Any] | None = None,
        source_note_paths: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        message = self._find_message(session, assistant_message_id, role="assistant")
        now = utc_now()
        message["content"] = assistant_content
        message["status"] = "completed"
        message["error_type"] = ""
        message["error_message"] = ""
        message["retryable"] = False
        message["updated_at"] = now
        self._update_session_context(
            session,
            interview_plan=interview_plan,
            interview_state=interview_state,
            source_note_paths=source_note_paths,
        )
        session["status"] = "active"
        session["updated_at"] = now
        session["end_error"] = None
        self.save_session(session)
        return {"assistant_message": message, "session": session}

    def fail_assistant_message(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        assistant_content: str = "",
        error_type: str = "Error",
        error_message: str = "",
        retryable: bool = True,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        message = self._find_message(session, assistant_message_id, role="assistant")
        now = utc_now()
        message["content"] = assistant_content or message.get("content", "")
        message["status"] = "failed"
        message["error_type"] = error_type
        message["error_message"] = error_message
        message["retryable"] = bool(retryable)
        message["updated_at"] = now
        session["updated_at"] = now
        self.save_session(session)
        return {"assistant_message": message, "session": session}

    def save_turn_review(
        self,
        *,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str,
        feedback: dict[str, Any] | None,
        reference_answer: str,
        context_note_paths: list[str] | tuple[str, ...] | None = None,
        profile_signals: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        reviews_doc = self.load_reviews(session_id)
        reviews = reviews_doc.setdefault("reviews", [])
        existing = self._find_review(reviews, user_message_id=user_message_id, assistant_message_id=assistant_message_id)
        if existing is not None:
            existing["feedback"] = feedback or {}
            existing["expression_example"] = reference_answer or ""
            existing["reference_answer"] = reference_answer or ""
            existing["context_note_paths"] = [str(path) for path in (context_note_paths or []) if str(path).strip()]
            existing["profile_signals"] = profile_signals or []
            existing["status"] = "completed"
            existing["error"] = ""
            existing["updated_at"] = utc_now()
            self.save_reviews(reviews_doc)
            return existing
        review = {
            "turn_id": f"turn-{len(reviews) + 1:04d}",
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "feedback": feedback or {},
            "expression_example": reference_answer or "",
            "reference_answer": reference_answer or "",
            "context_note_paths": [str(path) for path in (context_note_paths or []) if str(path).strip()],
            "profile_signals": profile_signals or [],
            "status": "completed",
            "error": "",
            "retry_count": 0,
            "created_at": utc_now(),
        }
        reviews.append(review)
        self.save_reviews(reviews_doc)
        return review

    def create_pending_review(
        self,
        *,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str,
        context_note_paths: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        reviews_doc = self.load_reviews(session_id)
        reviews = reviews_doc.setdefault("reviews", [])
        existing = self._find_review(reviews, user_message_id=user_message_id, assistant_message_id=assistant_message_id)
        now = utc_now()
        if existing is not None:
            existing["status"] = "pending"
            existing["error"] = ""
            existing["retry_count"] = int(existing.get("retry_count", 0) or 0) + 1
            existing["updated_at"] = now
            self.save_reviews(reviews_doc)
            return existing
        review = {
            "turn_id": f"turn-{len(reviews) + 1:04d}",
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "feedback": {},
            "expression_example": "",
            "reference_answer": "",
            "context_note_paths": [str(path) for path in (context_note_paths or []) if str(path).strip()],
            "profile_signals": [],
            "status": "pending",
            "error": "",
            "retry_count": 0,
            "created_at": now,
        }
        reviews.append(review)
        self.save_reviews(reviews_doc)
        return review

    def mark_review_failed(
        self,
        *,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str,
        error: str,
        context_note_paths: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        reviews_doc = self.load_reviews(session_id)
        reviews = reviews_doc.setdefault("reviews", [])
        review = self._find_review(reviews, user_message_id=user_message_id, assistant_message_id=assistant_message_id)
        now = utc_now()
        if review is None:
            review = {
                "turn_id": f"turn-{len(reviews) + 1:04d}",
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
                "feedback": {},
                "expression_example": "",
                "reference_answer": "",
                "context_note_paths": [str(path) for path in (context_note_paths or []) if str(path).strip()],
                "profile_signals": [],
                "created_at": now,
            }
            reviews.append(review)
        review["status"] = "failed"
        review["error"] = error
        review["retry_count"] = int(review.get("retry_count", 0) or 0)
        review["updated_at"] = now
        self.save_reviews(reviews_doc)
        return review

    def mark_end_failed(self, session_id: str, error: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        session["status"] = "end_failed"
        session["updated_at"] = utc_now()
        session["end_error"] = error
        self.save_session(session)
        return session

    def mark_completed(
        self,
        *,
        session_id: str,
        final_review: dict[str, Any] | None = None,
        profile_update: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        now = utc_now()
        session["status"] = "completed"
        session["updated_at"] = now
        session["ended_at"] = now
        session["end_error"] = None
        session["final_review"] = final_review
        session["profile_update"] = profile_update
        self.save_session(session)
        return session

    def session_path(self, session_id: str) -> Path:
        safe_id = slugify(session_id)
        month = safe_id[:6] if re.match(r"^\d{6}", safe_id) else datetime.now().strftime("%Y-%m")
        if re.match(r"^\d{6}", month):
            month = f"{month[:4]}-{month[4:6]}"
        return self.root / month / f"{safe_id}.json"

    def reviews_path(self, session_id: str) -> Path:
        path = self.session_path(session_id)
        return path.with_name(path.stem + ".reviews.json")

    def _next_message_id(self, messages: list[dict[str, Any]]) -> str:
        return f"msg-{len(messages) + 1:04d}"

    def _find_message(self, session: dict[str, Any], message_id: str, *, role: str | None = None) -> dict[str, Any]:
        for message in session.get("messages", []):
            if message.get("id") == message_id and (role is None or message.get("role") == role):
                return message
        raise ValueError(f"message not found: {message_id}")

    def _find_review(
        self,
        reviews: list[dict[str, Any]],
        *,
        user_message_id: str,
        assistant_message_id: str,
    ) -> dict[str, Any] | None:
        for review in reviews:
            if review.get("user_message_id") == user_message_id and review.get("assistant_message_id") == assistant_message_id:
                return review
        return None

    def _update_session_context(
        self,
        session: dict[str, Any],
        *,
        interview_plan: dict[str, Any] | None = None,
        interview_state: dict[str, Any] | None = None,
        source_note_paths: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if interview_plan is not None:
            session["interview_plan"] = interview_plan
        if interview_state is not None:
            session["interview_state"] = interview_state
        if source_note_paths:
            context = session.setdefault("context", {})
            existing = list(context.get("source_note_paths") or [])
            for path in source_note_paths:
                text = str(path).strip()
                if text and text not in existing:
                    existing.append(text)
            context["source_note_paths"] = existing
            context["source_note_count"] = len(existing)

    def _infer_topic_label(self, session: dict[str, Any]) -> str:
        state = session.get("interview_state") or {}
        if state.get("current_topic"):
            return str(state["current_topic"])
        plan = session.get("interview_plan") or {}
        topics = plan.get("topics") if isinstance(plan, dict) else None
        if topics and isinstance(topics, list) and topics[0].get("name"):
            return str(topics[0]["name"])
        context = session.get("context") or {}
        return str(context.get("source_value") or "")
