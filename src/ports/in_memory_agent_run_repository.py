from __future__ import annotations

from typing import Any, Callable

from ports.agent_run_repository import AgentRunRepository
from services.tasks.pipeline_task import PipelineTask, PipelineTaskContext, PipelineTaskManager


class InMemoryAgentRunRepository:
    def __init__(self, manager: PipelineTaskManager) -> None:
        self._manager = manager

    def submit(self, kind: str, fn: Callable[[PipelineTaskContext], Any]) -> PipelineTask:
        return self._manager.submit(kind, fn)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self._manager.get(run_id)

    def list_events(self, run_id: str, *, after_seq: int = -1) -> list[dict[str, Any]]:
        return self._manager.list_events(run_id, after_seq=after_seq)
