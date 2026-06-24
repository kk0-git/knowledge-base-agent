from __future__ import annotations

from typing import Any, Callable, Protocol

from services.tasks.pipeline_task import PipelineTask, PipelineTaskContext


class AgentRunRepository(Protocol):
    def submit(self, kind: str, fn: Callable[[PipelineTaskContext], Any]) -> PipelineTask: ...

    def get(self, run_id: str) -> dict[str, Any] | None: ...

    def list_events(self, run_id: str, *, after_seq: int = -1) -> list[dict[str, Any]]: ...
