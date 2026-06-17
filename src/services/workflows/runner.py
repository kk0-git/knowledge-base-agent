from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from services.rag.agent_answer import AgentAnswerPipeline, agent_run_result_to_dict
from services.wiki.manager import WikiManager
from services.wiki.state_store import WikiStateStore
from services.workflows.audit import run_deterministic_audit
from services.workflows.context_builder import ContextBuilder
from services.workflows.review import run_organize_suggestions
from services.workflows.schema import ContextPack, ScopeResult, TaskResult, WorkflowSpec
from services.workflows.scope_resolver import ScopeResolver
from knowledge_base_agent.llm.client import LLMClient


class WorkflowRunner:
    def __init__(
        self,
        *,
        scope_resolver: ScopeResolver,
        context_builder: ContextBuilder,
        answer_pipeline: AgentAnswerPipeline | None = None,
        wiki_manager: WikiManager | None = None,
        vault_root: Path | None = None,
        wiki_state_store: WikiStateStore | None = None,
        wiki_dir: Path | None = None,
        overview_note_threshold: int = 30,
        llm_client: LLMClient | None = None,
        llm_model: str | None = None,
        llm_temperature: float = 0.2,
    ) -> None:
        self.scope_resolver = scope_resolver
        self.context_builder = context_builder
        self.answer_pipeline = answer_pipeline
        self.wiki_manager = wiki_manager
        self.vault_root = vault_root
        self.wiki_state_store = wiki_state_store
        self.wiki_dir = wiki_dir
        self.overview_note_threshold = overview_note_threshold
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

    def run(self, spec: WorkflowSpec) -> TaskResult:
        if spec.task_type == "answer":
            return self.run_answer(spec)
        if spec.task_type == "synthesize_wiki":
            return self.run_synthesize_wiki(spec)
        if spec.task_type == "audit":
            return self.run_audit(spec)
        if spec.task_type == "organize":
            return self.run_organize(spec)
        if spec.task_type in {"organize_suggestions", "generate_review"}:
            return self.run_organize_suggestions(spec)
        raise ValueError(f"Unsupported workflow task_type: {spec.task_type}")

    def run_answer(self, spec: WorkflowSpec) -> TaskResult:
        if self.answer_pipeline is None:
            raise ValueError("answer_pipeline is required for answer workflow")
        query = spec.user_request or spec.scope.value or ""
        if not query.strip():
            raise ValueError("answer workflow requires user_request or search scope value")

        started_at = time.perf_counter()
        result = self.answer_pipeline.run(query)
        raw = agent_run_result_to_dict(result)
        scope = ScopeResult(
            scope=spec.scope,
            chunks=tuple(raw["context_items"]),
            metadata={
                "query": query,
                "command": raw["command"],
                "router_decision": raw["router_decision"],
            },
        )
        context = ContextPack(
            mode=spec.context_mode or "answer_context",
            scope_result=scope,
            context_text=raw["context_text"],
            items=tuple(raw["context_items"]),
            citations=tuple(raw["context_items"]),
            stats=raw["telemetry"].get("context", {}),
        )
        return TaskResult(
            task_type="answer",
            scope=scope,
            context=context,
            output={
                "answer": raw["answer"],
                "retrieval": raw["retrieval"],
                "tool_errors": raw["tool_errors"],
            },
            writeback={"type": spec.writeback.type},
            timing={"workflow_ms": elapsed_ms(started_at), **raw["timing"]},
            telemetry=raw["telemetry"],
        )

    def run_audit(self, spec: WorkflowSpec) -> TaskResult:
        if self.vault_root is None:
            raise ValueError("vault_root is required for audit workflow")

        started_at = time.perf_counter()
        scope = self.scope_resolver.resolve(spec.scope)
        context = self.context_builder.build(scope, mode=spec.context_mode or "audit_context")
        audit = run_deterministic_audit(
            vault_root=self.vault_root,
            scope_result=scope,
            wiki_state_store=self.wiki_state_store,
            wiki_dir=self.wiki_dir,
            overview_note_threshold=self.overview_note_threshold,
            min_note_chars=int(spec.options.get("min_note_chars", 80)),
            max_issues=int(spec.options.get("max_issues", 200)),
        )
        return TaskResult(
            task_type="audit",
            scope=scope,
            context=context,
            output={"audit": audit},
            writeback={"type": spec.writeback.type, "path": spec.writeback.path},
            timing={"workflow_ms": elapsed_ms(started_at)},
            telemetry={
                "scope": scope.metadata,
                "context": context.stats,
                "audit": audit["summary"],
            },
        )

    def run_synthesize_wiki(self, spec: WorkflowSpec) -> TaskResult:
        if self.wiki_manager is None:
            raise ValueError("wiki_manager is required for synthesize_wiki workflow")
        if spec.scope.type != "tag":
            raise ValueError("synthesize_wiki currently requires tag scope")
        tag = str(spec.scope.value or "").strip()
        if not tag:
            raise ValueError("synthesize_wiki requires scope.value tag")

        started_at = time.perf_counter()
        scope = self.scope_resolver.resolve(spec.scope)
        context = self.context_builder.build(scope, mode=spec.context_mode or "wiki_context")
        result = self.wiki_manager.synthesize_tag(
            tag,
            force=bool(spec.options.get("force", True)),
        )
        return TaskResult(
            task_type="synthesize_wiki",
            scope=scope,
            context=context,
            output={"synthesize": result},
            writeback={"type": spec.writeback.type, "path": spec.writeback.path},
            timing={"workflow_ms": elapsed_ms(started_at)},
            telemetry={
                "scope": scope.metadata,
                "context": context.stats,
            },
        )

    def run_organize_suggestions(self, spec: WorkflowSpec) -> TaskResult:
        if self.vault_root is None:
            raise ValueError("vault_root is required for organize_suggestions workflow")
        if self.llm_client is None or not self.llm_model:
            raise ValueError("llm_client and llm_model are required for organize_suggestions workflow")

        started_at = time.perf_counter()
        scope = self.scope_resolver.resolve(spec.scope)
        context = self.context_builder.build(scope, mode=spec.context_mode or "suggestion_context")
        review = run_organize_suggestions(
            vault_root=self.vault_root,
            scope_result=scope,
            llm_client=self.llm_client,
            llm_model=self.llm_model,
            temperature=float(spec.options.get("temperature", self.llm_temperature)),
            max_notes=int(spec.options.get("max_notes", 12)),
            max_chars_per_note=int(spec.options.get("max_chars_per_note", 1800)),
            review_mode=str(spec.options.get("review_mode", "auto")),
        )
        return TaskResult(
            task_type=spec.task_type,
            scope=scope,
            context=context,
            output={"organize_suggestions": review},
            writeback={"type": spec.writeback.type, "path": spec.writeback.path},
            timing={"workflow_ms": elapsed_ms(started_at)},
            telemetry={
                "scope": scope.metadata,
                "context": context.stats,
                "review": {
                    "notes": len(review["packet"].get("notes", [])),
                    "internal_edges": len(review["packet"].get("internal_edges", [])),
                    "peer_hints": len(review["packet"].get("peer_hints", [])),
                },
            },
        )

    def run_organize(self, spec: WorkflowSpec) -> TaskResult:
        if self.vault_root is None:
            raise ValueError("vault_root is required for organize workflow")
        if self.llm_client is None or not self.llm_model:
            raise ValueError("llm_client and llm_model are required for organize workflow")

        started_at = time.perf_counter()
        scope = self.scope_resolver.resolve(spec.scope)
        context = self.context_builder.build(scope, mode=spec.context_mode or "organize_context")
        audit = run_deterministic_audit(
            vault_root=self.vault_root,
            scope_result=scope,
            wiki_state_store=self.wiki_state_store,
            wiki_dir=self.wiki_dir,
            overview_note_threshold=self.overview_note_threshold,
            min_note_chars=int(spec.options.get("min_note_chars", 80)),
            max_issues=int(spec.options.get("max_issues", 200)),
        )
        review = run_organize_suggestions(
            vault_root=self.vault_root,
            scope_result=scope,
            llm_client=self.llm_client,
            llm_model=self.llm_model,
            temperature=float(spec.options.get("temperature", self.llm_temperature)),
            max_notes=int(spec.options.get("max_notes", 12)),
            max_chars_per_note=int(spec.options.get("max_chars_per_note", 1800)),
            review_mode=str(spec.options.get("review_mode", "auto")),
        )
        summary = {
            "notes": len(scope.notes),
            "issues": audit["summary"].get("issues", 0),
            "errors": audit["summary"].get("errors", 0),
            "warnings": audit["summary"].get("warnings", 0),
            "review_mode": review.get("review_mode"),
            "review_notes": len(review["packet"].get("notes", [])),
            "validation_warnings": review.get("validation", {}).get("warning_count", 0),
            "validation_corrections": review.get("validation", {}).get("correction_count", 0),
        }
        return TaskResult(
            task_type="organize",
            scope=scope,
            context=context,
            output={
                "organize": {
                    "summary": summary,
                    "audit": audit,
                    "review": review,
                }
            },
            writeback={"type": spec.writeback.type, "path": spec.writeback.path},
            timing={"workflow_ms": elapsed_ms(started_at)},
            telemetry={
                "scope": scope.metadata,
                "context": context.stats,
                "audit": audit["summary"],
                "review": {
                    "notes": len(review["packet"].get("notes", [])),
                    "internal_edges": len(review["packet"].get("internal_edges", [])),
                    "peer_hints": len(review["packet"].get("peer_hints", [])),
                    "validation": review.get("validation", {}),
                },
            },
        )


def task_result_to_dict(result: TaskResult) -> dict[str, Any]:
    return asdict(result)


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
