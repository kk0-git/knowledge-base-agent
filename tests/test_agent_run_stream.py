from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.tasks.pipeline_task import PipelineTaskContext, PipelineTaskManager
from services.tasks.stream import stream_task_events


class AgentRunStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task_manager = PipelineTaskManager(max_workers=1)

    def tearDown(self) -> None:
        self.task_manager.executor.shutdown(wait=False, cancel_futures=True)

    def test_stream_replays_events_until_done(self) -> None:
        def run(ctx: PipelineTaskContext) -> dict[str, str]:
            ctx.emit("status", {"stage": "started"})
            ctx.emit("answer_delta", {"text": "hi"})
            return {"answer": "hi"}

        task = self.task_manager.submit("agent_turn:answer", run)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if (self.task_manager.get(task.task_id) or {}).get("status") == "succeeded":
                break
            time.sleep(0.05)

        chunks = list(stream_task_events(self.task_manager, task.task_id))
        joined = "".join(chunks)
        self.assertIn("event: status", joined)
        self.assertIn("event: answer_delta", joined)
        self.assertIn('"text": "hi"', joined)

    def test_stream_missing_task_emits_error(self) -> None:
        chunks = list(stream_task_events(self.task_manager, "missing-task"))
        self.assertTrue(any("task not found" in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
