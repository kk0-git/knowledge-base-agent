from __future__ import annotations

import json
import time
from typing import Any

from services.tasks.pipeline_task import PipelineTaskManager


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def classify_runtime_error(exc: BaseException | str) -> dict[str, Any]:
    raw = str(exc or "")
    lower = raw.lower()
    error_type = type(exc).__name__ if isinstance(exc, BaseException) else "Error"
    retryable = True
    category = "unknown"
    if "ssl" in lower or "certificate" in lower:
        category = "ssl"
    elif "timeout" in lower or "timed out" in lower:
        category = "timeout"
    elif "connection" in lower or "network" in lower or "dns" in lower or "temporary failure" in lower:
        category = "network"
    elif "429" in lower or "rate limit" in lower or "too many requests" in lower:
        category = "rate_limit"
    elif "500" in lower or "502" in lower or "503" in lower or "504" in lower:
        category = "server_error"
    elif "400" in lower or "401" in lower or "403" in lower or "unauthorized" in lower or "forbidden" in lower:
        category = "request_error"
        retryable = False
    return {
        "message": raw,
        "error_type": error_type,
        "category": category,
        "retryable": retryable,
    }


def stream_task_events(task_manager: PipelineTaskManager, task_id: str):
    last_seq = -1
    while True:
        task = task_manager.get(task_id)
        if not task:
            yield sse_event("error", {"message": f"task not found: {task_id}"})
            return
        for item in task_manager.list_events(task_id, after_seq=last_seq):
            last_seq = int(item.get("seq", last_seq))
            event_name = str(item.get("event") or "message")
            if event_name == "started":
                continue
            yield sse_event(event_name, item.get("payload") or {})
        status = str(task.get("status") or "")
        if status in {"succeeded", "failed"}:
            if status == "failed":
                has_error = any(str(entry.get("event") or "") == "error" for entry in task.get("events") or [])
                if not has_error:
                    yield sse_event("error", classify_runtime_error(task.get("error") or "task failed"))
            return
        time.sleep(0.2)
