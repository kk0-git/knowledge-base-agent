from __future__ import annotations

from typing import Any, Callable

from ports.agent_run_repository import AgentRunRepository
from ports.answer_session_repository import AnswerSessionRepository
from ports.session_repository import SessionRepository
from services.agent_turns.runner import AgentTurnRunner
from services.agent_turns.schema import AgentTurnInput, AgentTurnResult
from services.tasks.pipeline_task import PipelineTaskContext


class AgentTurnService:
    def __init__(
        self,
        *,
        session_repo: SessionRepository,
        run_repo: AgentRunRepository,
        runner_factory: Callable[[Any], AgentTurnRunner],
        classify_error: Callable[[BaseException | str], dict[str, Any]],
        answer_session_repo: AnswerSessionRepository | None = None,
    ) -> None:
        self.session_repo = session_repo
        self.answer_session_repo = answer_session_repo
        self.run_repo = run_repo
        self.runner_factory = runner_factory
        self.classify_error = classify_error

    def start_run(self, request: AgentTurnInput, *, llm_client: Any) -> dict[str, Any]:
        if not str(request.query or "").strip():
            raise ValueError("query is required")

        turn_ids: dict[str, str] | None = None
        if request.chat_mode == "interview":
            if not str(request.session_id or "").strip():
                raise ValueError("session_id is required for interview runs")
            turn_ids = self._ensure_interview_pending_turn(request)
        elif request.chat_mode == "answer" and self.answer_session_repo is not None:
            if not str(request.session_id or "").strip():
                raise ValueError("session_id is required for answer runs")
            turn_ids = self._ensure_answer_pending_turn(request)

        metadata = {
            "chat_mode": request.chat_mode,
            "session_id": request.session_id,
            "user_message_id": turn_ids["user"] if turn_ids else None,
            "assistant_message_id": turn_ids["assistant"] if turn_ids else None,
        }

        def execute(ctx: PipelineTaskContext) -> dict[str, Any]:
            runner = self.runner_factory(llm_client)
            try:
                if request.chat_mode == "interview":
                    result = runner.run_interview(ctx, request)
                    if turn_ids:
                        self._complete_interview_turn(request, turn_ids, result)
                elif request.chat_mode == "answer":
                    result = runner.run_answer(ctx, request)
                    if turn_ids and self.answer_session_repo is not None:
                        self._complete_answer_turn(request, turn_ids, result)
                else:
                    raise ValueError(f"unsupported chat_mode for agent runs: {request.chat_mode}")
                return self._result_payload(result)
            except Exception as exc:
                if turn_ids:
                    error = self.classify_error(exc)
                    if request.chat_mode == "interview":
                        self.session_repo.fail_assistant(
                            session_id=str(request.session_id),
                            assistant_message_id=turn_ids["assistant"],
                            assistant_content="",
                            error_type=str(error.get("error_type") or "Error"),
                            error_message=str(error.get("message") or exc),
                            retryable=bool(error.get("retryable", True)),
                        )
                    elif request.chat_mode == "answer" and self.answer_session_repo is not None:
                        self.answer_session_repo.fail_assistant(
                            session_id=str(request.session_id),
                            assistant_message_id=turn_ids["assistant"],
                            assistant_content="",
                            error_type=str(error.get("error_type") or "Error"),
                            error_message=str(error.get("message") or exc),
                            retryable=bool(error.get("retryable", True)),
                        )
                error = self.classify_error(exc)
                ctx.emit("error", error)
                raise

        kind = f"agent_turn:{request.chat_mode}"
        task = self.run_repo.submit(kind, execute)
        return {
            "task_id": task.task_id,
            "status": task.status,
            **metadata,
        }

    def _ensure_interview_pending_turn(self, request: AgentTurnInput) -> dict[str, str]:
        session_id = str(request.session_id)
        if request.assistant_message_id:
            session = self.session_repo.load_session(session_id)
            assistant = self._find_message(session, request.assistant_message_id, role="assistant")
            user = self._find_pair_user(session, assistant)
            if assistant.get("status") == "pending":
                return {"user": str(user.get("id") or ""), "assistant": str(assistant.get("id") or "")}
            raise ValueError("assistant_message_id is not a pending message")

        pending = self.session_repo.append_pending_turn(
            session_id=session_id,
            user_content=request.query,
            interview_plan=request.interview_plan,
            interview_state=request.interview_state,
            source_note_paths=request.source_note_paths,
        )
        return {
            "user": str(pending["user_message"]["id"]),
            "assistant": str(pending["assistant_message"]["id"]),
        }

    def _ensure_answer_pending_turn(self, request: AgentTurnInput) -> dict[str, str]:
        if self.answer_session_repo is None:
            raise ValueError("answer session repository is not configured")
        session_id = str(request.session_id)
        if request.assistant_message_id:
            session = self.answer_session_repo.load_session(session_id)
            assistant = self._find_message(session, request.assistant_message_id, role="assistant")
            user = self._find_pair_user(session, assistant)
            if assistant.get("status") == "pending":
                return {"user": str(user.get("id") or ""), "assistant": str(assistant.get("id") or "")}
            raise ValueError("assistant_message_id is not a pending message")

        pending = self.answer_session_repo.append_pending_turn(
            session_id=session_id,
            user_content=request.query,
        )
        return {
            "user": str(pending["user_message"]["id"]),
            "assistant": str(pending["assistant_message"]["id"]),
        }

    def _complete_interview_turn(
        self,
        request: AgentTurnInput,
        turn_ids: dict[str, str],
        result: AgentTurnResult,
    ) -> None:
        self.session_repo.complete_assistant(
            session_id=str(request.session_id),
            assistant_message_id=turn_ids["assistant"],
            assistant_content=result.answer_text,
            interview_plan=result.interview_plan or request.interview_plan,
            interview_state=result.interview_state or request.interview_state,
            source_note_paths=result.source_note_paths or request.source_note_paths,
            agent_actions=result.agent_actions,
            citations=result.citations,
        )

    def _complete_answer_turn(
        self,
        request: AgentTurnInput,
        turn_ids: dict[str, str],
        result: AgentTurnResult,
    ) -> None:
        if self.answer_session_repo is None:
            return
        self.answer_session_repo.complete_assistant(
            session_id=str(request.session_id),
            assistant_message_id=turn_ids["assistant"],
            assistant_content=result.answer_text,
            agent_actions=result.agent_actions,
            citations=result.citations,
        )

    @staticmethod
    def _result_payload(result: AgentTurnResult) -> dict[str, Any]:
        return {
            "answer": result.answer_text,
            "citations": result.citations,
            "agent_actions": result.agent_actions,
            "interview_plan": result.interview_plan,
            "interview_state": result.interview_state,
            "telemetry": result.telemetry,
        }

    @staticmethod
    def _find_message(session: dict[str, Any], message_id: str, *, role: str) -> dict[str, Any]:
        for message in session.get("messages") or []:
            if str(message.get("id") or "") == str(message_id) and str(message.get("role") or "") == role:
                return message
        raise ValueError(f"{role} message not found: {message_id}")

    @staticmethod
    def _find_pair_user(session: dict[str, Any], assistant: dict[str, Any]) -> dict[str, Any]:
        messages = list(session.get("messages") or [])
        assistant_id = str(assistant.get("id") or "")
        for index, message in enumerate(messages):
            if str(message.get("id") or "") == assistant_id and index > 0:
                previous = messages[index - 1]
                if str(previous.get("role") or "") == "user":
                    return previous
        raise ValueError("paired user message not found")
