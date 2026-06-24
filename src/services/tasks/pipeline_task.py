from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineTask:
    task_id: str
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    current_step: str | None = None
    events: list[dict[str, Any]] | None = None
    result: Any = None
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_step": self.current_step,
            "events": list(self.events or []),
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
        }


class PipelineTaskContext:
    def __init__(self, manager: "PipelineTaskManager", task_id: str) -> None:
        self.manager = manager
        self.task_id = task_id

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.manager.emit(self.task_id, event, payload or {})


class PipelineTaskManager:
    def __init__(self, *, max_workers: int = 1, max_events: int = 200) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline-task")
        self.max_events = max_events
        self.lock = threading.Lock()
        self.tasks: dict[str, PipelineTask] = {}

    def submit(self, kind: str, fn: Callable[[PipelineTaskContext], Any]) -> PipelineTask:
        task_id = uuid.uuid4().hex
        task = PipelineTask(
            task_id=task_id,
            kind=kind,
            status="queued",
            created_at=utc_now_iso(),
            events=[],
        )
        with self.lock:
            self.tasks[task_id] = task
        self.executor.submit(self._run, task_id, fn)
        return task

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self.lock:
            task = self.tasks.get(task_id)
            return task.to_dict() if task else None

    def get_task(self, task_id: str) -> PipelineTask | None:
        with self.lock:
            return self.tasks.get(task_id)

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.lock:
            tasks = sorted(self.tasks.values(), key=lambda task: task.created_at, reverse=True)
            return [task.to_dict() for task in tasks[:limit]]

    def emit(self, task_id: str, event: str, payload: dict[str, Any]) -> None:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            task.current_step = event
            task.events = task.events or []
            task.events.append(
                {
                    "seq": len(task.events),
                    "at": utc_now_iso(),
                    "event": event,
                    "payload": payload,
                }
            )
            if len(task.events) > self.max_events:
                task.events = task.events[-self.max_events :]
                for index, item in enumerate(task.events):
                    item["seq"] = index

    def list_events(self, task_id: str, *, after_seq: int = -1) -> list[dict[str, Any]]:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return []
            events = list(task.events or [])
        return [item for item in events if int(item.get("seq", -1)) > after_seq]

    def _run(self, task_id: str, fn: Callable[[PipelineTaskContext], Any]) -> None:
        with self.lock:
            task = self.tasks[task_id]
            task.status = "running"
            task.started_at = utc_now_iso()
            task.current_step = "started"
        context = PipelineTaskContext(self, task_id)
        context.emit("started", {})
        try:
            result = fn(context)
            with self.lock:
                task = self.tasks[task_id]
                task.status = "succeeded"
                task.result = result
                task.finished_at = utc_now_iso()
                task.current_step = "succeeded"
        except Exception as exc:
            with self.lock:
                task = self.tasks[task_id]
                task.status = "failed"
                task.error_type = type(exc).__name__
                task.error = f"{type(exc).__name__}: {exc}"
                task.result = {"traceback": traceback.format_exc()}
                task.finished_at = utc_now_iso()
                task.current_step = "failed"
            context.emit("error", {"message": str(exc), "error_type": type(exc).__name__})
