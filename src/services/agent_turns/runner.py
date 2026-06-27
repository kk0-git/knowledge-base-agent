from __future__ import annotations

import time
from typing import Any

from agent.apps import InterviewInterviewerApp, InterviewTurnRequest, LibrarianApp, LibrarianRequest
from agent.interview.state import build_interview_state_machine
from services.rag.online_search import OnlineSearchClient
from services.tasks.pipeline_task import PipelineTaskContext
from services.wiki.state_store import WikiStateStore
from services.workflows.context_builder import ContextBuilder
from services.workflows.interview import (
    deterministic_interview_plan,
    format_interview_opening_message,
    interview_plan_from_dict,
    interview_plan_to_dict,
    prepare_interview_plan,
)
from services.workflows.schema import ScopeSpec
from services.workflows.scope_resolver import ScopeResolver

from services.agent_turns.schema import AgentTurnInput, AgentTurnResult, AgentTurnRunnerDeps


def _elapsed_since(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _current_time() -> float:
    return time.perf_counter()


class AgentTurnRunner:
    def __init__(self, deps: AgentTurnRunnerDeps, *, llm_client: Any) -> None:
        self.deps = deps
        self.llm_client = llm_client

    def run_interview(self, ctx: PipelineTaskContext, request: AgentTurnInput) -> AgentTurnResult:
        if self.deps.vault_path is None:
            raise ValueError("vault path is required for interview agent")
        if self.deps.wiki_state_path is None:
            raise ValueError("wiki state path is required for interview scope")

        ctx.emit("status", {"stage": "agent_v2_started", "message": "starting interview agent runtime"})
        rag_manager = self.deps.build_interview_rag_manager(request) if request.scope_type == "search" else None
        rag_manager_factory = lambda: self.deps.build_interview_rag_manager(request)
        scope = ScopeResolver(
            vault_root=self.deps.vault_path,
            wiki_state_store=WikiStateStore(self.deps.wiki_state_path),
            wiki_dir=self.deps.wiki_dir,
            rag_manager=rag_manager,
            overview_note_threshold=self.deps.overview_note_threshold,
        ).resolve(
            ScopeSpec(
                type=request.scope_type,
                value=request.scope_value,
                paths=tuple(request.scope_paths),
                top_k=request.notes_top_k,
            )
        )
        context = ContextBuilder(
            vault_root=self.deps.vault_path,
            max_chars_per_note=request.interview_max_chars_per_note,
            max_context_chars=request.interview_max_context_chars,
        ).build(scope, mode="interview_context")
        source_note_paths = list(
            dict.fromkeys(
                [
                    *request.source_note_paths,
                    *[
                        str(item.get("path") or item.get("relative_path") or "")
                        for item in context.items
                        if str(item.get("path") or item.get("relative_path") or "").strip()
                    ],
                ]
            )
        )
        ctx.emit("context", {"items": context.items, "stats": context.stats, "scope": scope.metadata})

        plan = None
        plan_error: str | None = None
        plan_source: str | None = None
        plan_started_at = _current_time()
        plan_ms: int | None = None
        if request.interview_plan:
            try:
                plan = interview_plan_from_dict(request.interview_plan, context=context)
                plan_source = "client_reused"
                plan_ms = _elapsed_since(plan_started_at)
            except Exception as exc:
                plan_error = f"client plan rejected: {exc}"
        if plan is None:
            try:
                plan = prepare_interview_plan(
                    context=context,
                    llm_client=self.llm_client,
                    model=self.deps.llm_model,
                    temperature=min(self.deps.llm_temperature, 0.2),
                )
                plan_source = "generated"
                plan_ms = _elapsed_since(plan_started_at)
            except Exception as exc:
                plan_error = str(exc) if not plan_error else f"{plan_error}; generated plan failed: {exc}"
                plan = deterministic_interview_plan(context)
                plan_source = "deterministic_fallback"
                plan_ms = _elapsed_since(plan_started_at)
        ctx.emit(
            "interview_plan",
            {
                "available": True,
                "fallback_used": plan_source == "deterministic_fallback",
                "source": plan_source,
                "plan": interview_plan_to_dict(plan),
                "error": plan_error,
                "latency_ms": plan_ms,
            },
        )

        server_interview_state = request.interview_state
        if request.session_id and self.deps.load_session_state:
            try:
                session = self.deps.load_session_state(str(request.session_id)) or {}
                server_interview_state = session.get("interview_state") or request.interview_state
            except Exception:
                server_interview_state = request.interview_state

        state_machine = build_interview_state_machine(
            plan=plan,
            state_payload=server_interview_state,
            session_id=str(request.session_id or ""),
        )
        if state_machine.snapshot().get("topic_phase") == "awaiting_selection":
            state_snapshot = state_machine.snapshot()
            selection_prompt = format_interview_opening_message(plan)
            ctx.emit("state_updated", state_snapshot)
            ctx.emit("answer_delta", {"text": selection_prompt})
            ctx.emit("answer", {"answer": selection_prompt, "model": self.deps.llm_model})
            telemetry = {
                "command": "InterviewAgentV2",
                "agent_v2": True,
                "awaiting_topic_selection": True,
                "interview_state": state_snapshot,
            }
            ctx.emit("done", {"telemetry": telemetry})
            if request.scope_type in {"folder", "selected_notes"} and self.deps.prewarm_interview_rag:
                self.deps.prewarm_interview_rag(request)
            return AgentTurnResult(
                answer_text=selection_prompt,
                interview_plan=interview_plan_to_dict(plan),
                interview_state=state_snapshot,
                source_note_paths=source_note_paths,
                telemetry=telemetry,
            )

        app_runner = InterviewInterviewerApp(self.deps.build_agent_runtime(self.llm_client))
        turn_request = InterviewTurnRequest(
            query=request.query,
            session_id=str(request.session_id or ""),
            chat_history=list(request.chat_history or []),
            interview_plan=plan,
            interview_state=server_interview_state,
            vault_root=self.deps.vault_path,
            rag_manager=rag_manager,
            rag_manager_factory=rag_manager_factory,
            scope_note_paths=tuple(source_note_paths),
            scope_type=request.scope_type,
            scope_value=request.scope_value or "",
            session_store=self.deps.interview_session_store,
            profile_store=self.deps.interview_profile_store,
            model=self.deps.llm_model,
            tool_mode="auto",
            trace_path=str(self.deps.project_root / "eval-results" / "agent-debug" / "traces"),
            max_steps=6,
            max_tool_calls_per_step=4,
            temperature=self.deps.llm_temperature,
        )

        answer_text = ""
        interview_state: dict[str, Any] | None = None
        citations: list[dict[str, Any]] = []
        agent_actions: list[dict[str, Any]] = []
        telemetry: dict[str, Any] = {}
        for event in app_runner.run_turn_stream(turn_request):
            event_type = str(event.get("type") or "")
            payload = event.get("payload") or {}
            ctx.emit(event_type, payload)
            if event_type == "answer_delta":
                answer_text += str(payload.get("text") or "")
            if event_type == "answer":
                answer_text = str(payload.get("answer") or answer_text)
            if event_type == "state_updated":
                interview_state = payload
            if event_type == "done":
                telemetry = dict(payload.get("telemetry") or {})
                interview_state = telemetry.get("interview_state") or interview_state
                citations = list(telemetry.get("citations") or [])
                agent_actions = list(telemetry.get("agent_actions") or payload.get("agent_actions") or [])

        if request.scope_type in {"folder", "selected_notes"} and self.deps.prewarm_interview_rag:
            self.deps.prewarm_interview_rag(request)

        return AgentTurnResult(
            answer_text=answer_text.strip(),
            interview_plan=interview_plan_to_dict(plan),
            interview_state=interview_state,
            citations=citations,
            agent_actions=agent_actions,
            source_note_paths=source_note_paths,
            telemetry=telemetry,
        )

    def run_answer(self, ctx: PipelineTaskContext, request: AgentTurnInput) -> AgentTurnResult:
        if self.deps.vault_path is None:
            raise ValueError("vault path is required for librarian agent")

        ctx.emit("status", {"stage": "agent_v2_started", "message": "starting librarian agent runtime"})
        scope, scope_note_paths, scope_metadata = self.deps.resolve_librarian_scope(request)
        source_note_paths = list(scope_note_paths)
        ctx.emit(
            "context",
            {
                "mode": "answer",
                "strict_evidence": bool(request.strict_evidence),
                "scope": {
                    "type": request.scope_type,
                    "value": request.scope_value,
                    "paths": request.scope_paths,
                    "metadata": scope_metadata,
                    "note_count": len(scope_note_paths),
                },
                "items": list(scope.notes) if scope is not None else [],
                "stats": {"context_items": len(scope_note_paths), "agent_v2": True},
            },
        )

        online_enabled = self.deps.librarian_online_enabled(request)
        online_client = OnlineSearchClient(provider=request.online_provider) if online_enabled else None
        rag_manager_factory = lambda: self.deps.build_librarian_rag_manager(request)
        learner_memory_context = ""
        profile_store = getattr(self.deps, "interview_profile_store", None)
        if profile_store is not None:
            from services.memory.injection import render_librarian_memory_context

            model = profile_store.ensure_derived_fresh()
            learner_memory_context = render_librarian_memory_context(
                model=model,
                scope_note_paths=tuple(scope_note_paths),
                scope_value=str(request.scope_value or ""),
            )
        app_runner = LibrarianApp(self.deps.build_agent_runtime(self.llm_client))
        answer_text = ""
        citations: list[dict[str, Any]] = []
        telemetry: dict[str, Any] = {}
        for event in app_runner.run_stream(
            LibrarianRequest(
                query=request.query,
                scope_type=request.scope_type,
                scope_value=request.scope_value or "",
                scope_note_paths=tuple(scope_note_paths),
                selected_note_paths=tuple(request.scope_paths) if request.scope_type == "selected_notes" else (),
                chat_history=list(request.chat_history or []),
                vault_root=self.deps.vault_path,
                rag_manager=self.deps.build_librarian_rag_manager(request),
                rag_manager_factory=rag_manager_factory,
                online_search_client=online_client,
                online_enabled=online_enabled,
                strict_evidence=bool(request.strict_evidence),
                model=self.deps.llm_model,
                tool_mode="auto",
                trace_path=str(self.deps.project_root / "eval-results" / "agent-debug" / "traces"),
                temperature=self.deps.llm_temperature,
                learner_memory_context=learner_memory_context,
            )
        ):
            event_type = str(event.get("type") or "")
            payload = event.get("payload") or {}
            ctx.emit(event_type, payload)
            if event_type == "answer_delta":
                answer_text += str(payload.get("text") or "")
            if event_type == "answer":
                answer_text = str(payload.get("answer") or answer_text)
            if event_type == "done":
                telemetry = dict(payload.get("telemetry") or payload)
                citations = list(telemetry.get("citations") or payload.get("citations") or [])

        return AgentTurnResult(
            answer_text=answer_text.strip(),
            citations=citations,
            source_note_paths=source_note_paths,
            telemetry=telemetry,
        )
