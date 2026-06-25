from __future__ import annotations

from typing import Any

CONVERSATION_SCHEMA_VERSION = 2

MESSAGE_ROLES = frozenset({"user", "assistant", "system"})
MESSAGE_STATUSES = frozenset({"pending", "completed", "failed"})

KIND_SKILLS = {
    "interview": "interviewer",
    "answer": "librarian",
    "review_dialogue": "reviewer",
}


def next_message_id(messages: list[dict[str, Any]]) -> str:
    return f"msg-{len(messages) + 1:04d}"


def next_turn_id(messages: list[dict[str, Any]]) -> str:
    turn_ids = [str(msg.get("turn_id") or "") for msg in messages if str(msg.get("turn_id") or "").strip()]
    if not turn_ids:
        return "turn-0001"
    last = turn_ids[-1]
    if last.startswith("turn-") and last[5:].isdigit():
        return f"turn-{int(last[5:]) + 1:04d}"
    return f"turn-{len(turn_ids) + 1:04d}"


def empty_error() -> dict[str, Any]:
    return {"type": "", "message": "", "retryable": False}


def sync_error_fields(message: dict[str, Any]) -> None:
    error = message.get("error")
    if not isinstance(error, dict):
        error = empty_error()
        message["error"] = error
    error_type = str(message.get("error_type") or error.get("type") or "")
    error_message = str(message.get("error_message") or error.get("message") or "")
    retryable = bool(message.get("retryable", error.get("retryable", False)))
    error["type"] = error_type
    error["message"] = error_message
    error["retryable"] = retryable
    message["error_type"] = error_type
    message["error_message"] = error_message
    message["retryable"] = retryable


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    msg = dict(message or {})
    role = str(msg.get("role") or "").strip()
    if role not in MESSAGE_ROLES:
        msg["role"] = role or "user"
    status = str(msg.get("status") or "").strip()
    if status not in MESSAGE_STATUSES:
        msg["status"] = "completed" if role == "user" else status or "completed"
    msg.setdefault("citations", [])
    msg.setdefault("agent_actions", [])
    msg.setdefault("meta", {})
    sync_error_fields(msg)
    return msg


def normalize_session_envelope(session: dict[str, Any]) -> dict[str, Any]:
    data = dict(session or {})
    mode = str(data.get("mode") or data.get("kind") or "").strip()
    if mode:
        data["mode"] = mode
        data["kind"] = str(data.get("kind") or mode)
    agent = dict(data.get("agent") or {})
    kind = str(data.get("kind") or "")
    if kind and not agent.get("skill"):
        agent["skill"] = KIND_SKILLS.get(kind, "")
    agent.setdefault("runtime_version", "")
    data["agent"] = agent
    data.setdefault("domain", {})
    return data


def normalize_session_messages(session: dict[str, Any]) -> dict[str, Any]:
    data = normalize_session_envelope(session)
    messages = [normalize_message(msg) for msg in data.get("messages") or [] if isinstance(msg, dict)]
    pending_turn_id = ""
    turn_counter = 0
    for msg in messages:
        if str(msg.get("turn_id") or "").strip():
            pending_turn_id = ""
            continue
        role = str(msg.get("role") or "")
        if role == "user":
            turn_counter += 1
            pending_turn_id = f"turn-{turn_counter:04d}"
            msg["turn_id"] = pending_turn_id
        elif role == "assistant" and pending_turn_id:
            msg["turn_id"] = pending_turn_id
            pending_turn_id = ""
        elif role == "assistant":
            turn_counter += 1
            msg["turn_id"] = f"turn-{turn_counter:04d}"
    data["messages"] = messages
    return data


def build_user_message(
    *,
    message_id: str,
    turn_id: str,
    content: str,
    created_at: str,
    status: str = "completed",
) -> dict[str, Any]:
    return normalize_message(
        {
            "id": message_id,
            "turn_id": turn_id,
            "role": "user",
            "content": content,
            "status": status,
            "created_at": created_at,
            "updated_at": created_at,
            "citations": [],
            "agent_actions": [],
            "meta": {},
            "error_type": "",
            "error_message": "",
            "retryable": False,
        }
    )


def build_pending_assistant_message(
    *,
    message_id: str,
    turn_id: str,
    created_at: str,
) -> dict[str, Any]:
    return normalize_message(
        {
            "id": message_id,
            "turn_id": turn_id,
            "role": "assistant",
            "content": "",
            "status": "pending",
            "created_at": created_at,
            "updated_at": created_at,
            "citations": [],
            "agent_actions": [],
            "meta": {},
            "error_type": "",
            "error_message": "",
            "retryable": False,
        }
    )


def build_completed_assistant_message(
    *,
    message_id: str,
    turn_id: str,
    content: str,
    created_at: str,
    agent_actions: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    msg = normalize_message(
        {
            "id": message_id,
            "turn_id": turn_id,
            "role": "assistant",
            "content": content,
            "status": "completed",
            "created_at": created_at,
            "updated_at": created_at,
            "citations": list(citations or []),
            "agent_actions": list(agent_actions or []),
            "meta": {},
            "error_type": "",
            "error_message": "",
            "retryable": False,
        }
    )
    return msg


def apply_assistant_completion(
    message: dict[str, Any],
    *,
    assistant_content: str,
    updated_at: str,
    agent_actions: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    process_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message["content"] = assistant_content
    message["status"] = "completed"
    message["updated_at"] = updated_at
    message.setdefault("citations", [])
    message.setdefault("agent_actions", [])
    message.setdefault("meta", {})
    if agent_actions is not None:
        message["agent_actions"] = list(agent_actions)
    if citations is not None:
        message["citations"] = list(citations)
    if process_summary is not None:
        meta = dict(message.get("meta") or {})
        meta["process_summary"] = process_summary
        message["meta"] = meta
        message["process_summary"] = process_summary
    message["error_type"] = ""
    message["error_message"] = ""
    message["retryable"] = False
    sync_error_fields(message)
    return message


def apply_assistant_failure(
    message: dict[str, Any],
    *,
    updated_at: str,
    assistant_content: str = "",
    error_type: str = "Error",
    error_message: str = "",
    retryable: bool = True,
) -> dict[str, Any]:
    message["content"] = assistant_content or message.get("content", "")
    message["status"] = "failed"
    message["updated_at"] = updated_at
    message["error_type"] = error_type
    message["error_message"] = error_message
    message["retryable"] = bool(retryable)
    sync_error_fields(message)
    return message


def find_message(
    session: dict[str, Any],
    message_id: str,
    *,
    role: str | None = None,
) -> dict[str, Any]:
    for message in session.get("messages", []):
        if message.get("id") == message_id and (role is None or message.get("role") == role):
            return message
    raise ValueError(f"message not found: {message_id}")


def turn_id_for_message_pair(
    session: dict[str, Any],
    *,
    user_message_id: str,
    assistant_message_id: str,
) -> str:
    session = normalize_session_messages(session)
    for message in session.get("messages", []):
        if message.get("id") in {user_message_id, assistant_message_id} and str(message.get("turn_id") or "").strip():
            return str(message["turn_id"])
    return ""


def project_dialogue_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if str(message.get("status") or "") != "completed":
            continue
        role = str(message.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        history.append({"role": role, "content": content})
    return history


def messages_for_agent_context(
    messages: list[dict[str, Any]],
    *,
    before_message_id: str | None = None,
    limit: int = 12,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if before_message_id and str(message.get("id") or "") == before_message_id:
            break
        if str(message.get("status") or "") != "completed":
            continue
        role = str(message.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "")
        result.append({"role": role, "content": content})
    return result[-limit:]


def migrate_history_to_messages(history: list[dict[str, Any]], *, created_at: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_user: str | None = None
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if role == "user":
            pending_user = content
            continue
        if role != "assistant" or pending_user is None:
            continue
        turn_id = next_turn_id(messages)
        messages.append(
            build_user_message(
                message_id=next_message_id(messages),
                turn_id=turn_id,
                content=pending_user,
                created_at=created_at,
            )
        )
        messages.append(
            build_completed_assistant_message(
                message_id=next_message_id(messages),
                turn_id=turn_id,
                content=content,
                created_at=created_at,
            )
        )
        pending_user = None
    if pending_user is not None:
        turn_id = next_turn_id(messages)
        messages.append(
            build_user_message(
                message_id=next_message_id(messages),
                turn_id=turn_id,
                content=pending_user,
                created_at=created_at,
            )
        )
    return normalize_session_messages({"messages": messages})["messages"]


def extract_suggested_commits(agent_actions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    for action in agent_actions or []:
        if not isinstance(action, dict):
            continue
        name = str(action.get("tool") or action.get("name") or "").strip()
        if name != "suggest_review_commit":
            continue
        output = action.get("output") or action.get("result") or {}
        if not isinstance(output, dict):
            continue
        weak_point_id = str(output.get("weak_point_id") or "").strip()
        if not weak_point_id:
            continue
        commits.append(
            {
                "weak_point_id": weak_point_id,
                "action": str(output.get("suggested_action") or output.get("action") or "retry").strip().lower(),
            }
        )
    return commits