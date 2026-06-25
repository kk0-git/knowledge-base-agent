from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.workflows.conversation_schema import (
    apply_assistant_completion,
    apply_assistant_failure,
    build_pending_assistant_message,
    build_user_message,
    find_message as schema_find_message,
    next_message_id,
    next_turn_id,
    normalize_session_messages,
)


SESSION_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str | None, fallback: str = "answer") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\\/:*?\"<>|\s]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:48] or fallback


def make_session_id(scope_value: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    slug = slugify(scope_value, fallback="answer")
    return f"{timestamp}-{slug}-{suffix}"


class AnswerSessionStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        *,
        scope_type: str = "all",
        scope_value: str | None = None,
        scope_paths: list[str] | None = None,
        strict_evidence: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        session_id = make_session_id(scope_value)
        session = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": session_id,
            "kind": "answer",
            "mode": "answer",
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "archived_at": "",
            "agent": {"skill": "librarian", "runtime_version": ""},
            "domain": {},
            "context": {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "scope_paths": [str(path) for path in (scope_paths or []) if str(path).strip()],
                "strict_evidence": bool(strict_evidence),
                "extra": extra or {},
            },
            "messages": [],
        }
        self.save_session(session)
        return session

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.root.glob("*/*.json"):
            try:
                session = self.load_session(path.stem)
            except Exception:
                continue
            messages = session.get("messages") or []
            first_user = next((msg for msg in messages if msg.get("role") == "user"), None)
            title = str((first_user or {}).get("content") or "").strip()[:80] or "问答对话"
            context = session.get("context") or {}
            sessions.append(
                {
                    "session_id": session.get("session_id"),
                    "status": session.get("status"),
                    "created_at": session.get("created_at"),
                    "updated_at": session.get("updated_at"),
                    "archived_at": session.get("archived_at"),
                    "title": title,
                    "scope_type": context.get("scope_type"),
                    "scope_value": context.get("scope_value"),
                    "message_count": len(messages),
                }
            )
        sessions.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return sessions[:limit]

    def load_session_bundle(self, session_id: str) -> dict[str, Any]:
        return {"session": self.load_session(session_id)}

    def load_session(self, session_id: str) -> dict[str, Any]:
        path = self.session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"answer session not found: {session_id}")
        session = json.loads(path.read_text(encoding="utf-8-sig"))
        return normalize_session_messages(session)

    def save_session(self, session: dict[str, Any]) -> None:
        session_id = str(session["session_id"])
        path = self.session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")

    def archive_session(self, session_id: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        now = utc_now()
        session["status"] = "archived"
        session["archived_at"] = now
        session["updated_at"] = now
        self.save_session(session)
        return session

    def append_pending_turn(
        self,
        *,
        session_id: str,
        user_content: str,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        if session.get("status") not in {"active"}:
            raise ValueError(f"cannot append to session with status {session.get('status')}")
        now = utc_now()
        messages = session.setdefault("messages", [])
        turn_id = next_turn_id(messages)
        user_message = build_user_message(
            message_id=next_message_id(messages),
            turn_id=turn_id,
            content=user_content,
            created_at=now,
        )
        assistant_message = build_pending_assistant_message(
            message_id=next_message_id([*messages, user_message]),
            turn_id=turn_id,
            created_at=now,
        )
        messages.extend([user_message, assistant_message])
        session["updated_at"] = now
        self.save_session(session)
        return {"user_message": user_message, "assistant_message": assistant_message, "session": session}

    def complete_assistant_message(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        assistant_content: str,
        agent_actions: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
        process_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        message = schema_find_message(session, assistant_message_id, role="assistant")
        now = utc_now()
        apply_assistant_completion(
            message,
            assistant_content=assistant_content,
            updated_at=now,
            agent_actions=agent_actions,
            citations=citations,
            process_summary=process_summary,
        )
        session["updated_at"] = now
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
        message = schema_find_message(session, assistant_message_id, role="assistant")
        now = utc_now()
        apply_assistant_failure(
            message,
            updated_at=now,
            assistant_content=assistant_content,
            error_type=error_type,
            error_message=error_message,
            retryable=retryable,
        )
        session["updated_at"] = now
        self.save_session(session)
        return {"assistant_message": message, "session": session}

    def session_path(self, session_id: str) -> Path:
        safe_id = slugify(session_id, fallback=session_id)
        month = safe_id[:6] if re.match(r"^\d{6}", safe_id) else datetime.now().strftime("%Y-%m")
        if re.match(r"^\d{6}", month):
            month = f"{month[:4]}-{month[4:6]}"
        return self.root / month / f"{safe_id}.json"

    def _find_message(self, session: dict[str, Any], message_id: str, *, role: str | None = None) -> dict[str, Any]:
        return schema_find_message(session, message_id, role=role)
