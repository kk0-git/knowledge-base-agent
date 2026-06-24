from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ports.file_answer_session_repository import FileAnswerSessionRepository
from ports.file_session_repository import FileSessionRepository
from ports.in_memory_agent_run_repository import InMemoryAgentRunRepository
from services.agent_turns.schema import AgentTurnInput, AgentTurnResult
from services.agent_turns.service import AgentTurnService
from services.tasks.pipeline_task import PipelineTaskContext, PipelineTaskManager
from services.workflows.answer_sessions import AnswerSessionStore
from services.workflows.interview_sessions import InterviewSessionStore


class FakeRunner:
    def run_interview(self, ctx: PipelineTaskContext, request: AgentTurnInput) -> AgentTurnResult:
        ctx.emit("answer_delta", {"text": "ok"})
        ctx.emit("done", {"telemetry": {"interview_state": {"current_topic": "MCP"}}})
        return AgentTurnResult(
            answer_text="ok",
            interview_state={"current_topic": "MCP"},
            interview_plan={"topics": []},
        )

    def run_answer(self, ctx: PipelineTaskContext, request: AgentTurnInput) -> AgentTurnResult:
        ctx.emit("answer", {"answer": "answer ok"})
        return AgentTurnResult(answer_text="answer ok", citations=[{"path": "note.md"}])


class AgentTurnServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = InterviewSessionStore(Path(self.tmp.name))
        self.session_repo = FileSessionRepository(self.store)
        self.task_manager = PipelineTaskManager(max_workers=1)
        self.run_repo = InMemoryAgentRunRepository(self.task_manager)
        self.service = AgentTurnService(
            session_repo=self.session_repo,
            run_repo=self.run_repo,
            runner_factory=lambda _client: FakeRunner(),
            classify_error=lambda exc: {"message": str(exc), "error_type": "Error", "retryable": True},
        )
        self.session = self.store.create_session(source_type="folder", source_value="demo")

    def tearDown(self) -> None:
        self.tmp.cleanup()
        self.task_manager.executor.shutdown(wait=False, cancel_futures=True)

    def _wait_task(self, task_id: str, timeout: float = 5.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            task = self.run_repo.get(task_id)
            if task and task.get("status") in {"succeeded", "failed"}:
                return task
            time.sleep(0.05)
        raise AssertionError(f"task did not finish: {task_id}")

    def test_interview_run_completes_pending_message(self) -> None:
        payload = self.service.start_run(
            AgentTurnInput(
                query="test",
                chat_mode="interview",
                session_id=self.session["session_id"],
            ),
            llm_client=object(),
        )
        self._wait_task(payload["task_id"])
        session = self.session_repo.load_session(self.session["session_id"])
        assistant = session["messages"][-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["status"], "completed")
        self.assertEqual(assistant["content"], "ok")

    def test_answer_run_returns_result(self) -> None:
        payload = self.service.start_run(
            AgentTurnInput(query="what is mcp", chat_mode="answer"),
            llm_client=object(),
        )
        task = self._wait_task(payload["task_id"])
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(task["result"]["answer"], "answer ok")

    def test_answer_run_completes_pending_message(self) -> None:
        answer_store = AnswerSessionStore(Path(self.tmp.name) / "answer-sessions")
        answer_repo = FileAnswerSessionRepository(answer_store)
        answer_session = answer_store.create_session(scope_type="all", scope_value="demo")
        service = AgentTurnService(
            session_repo=self.session_repo,
            answer_session_repo=answer_repo,
            run_repo=self.run_repo,
            runner_factory=lambda _client: FakeRunner(),
            classify_error=lambda exc: {"message": str(exc), "error_type": "Error", "retryable": True},
        )
        payload = service.start_run(
            AgentTurnInput(
                query="what is mcp",
                chat_mode="answer",
                session_id=answer_session["session_id"],
            ),
            llm_client=object(),
        )
        self._wait_task(payload["task_id"])
        session = answer_repo.load_session(answer_session["session_id"])
        assistant = session["messages"][-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["status"], "completed")
        self.assertEqual(assistant["content"], "answer ok")
        self.assertEqual(assistant["citations"][0]["path"], "note.md")


if __name__ == "__main__":
    unittest.main()
