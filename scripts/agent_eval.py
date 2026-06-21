from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agent.apps import (
    CoachTurnRequest,
    InterviewCoachApp,
    InterviewInterviewerApp,
    InterviewTurnRequest,
    LibrarianApp,
    LibrarianRequest,
)
from agent.interview.state import build_interview_state_machine
from agent.runtime import AgentRuntime
from agent.skill_loader import SkillLoader
from agent.tool_registry import ToolRegistry
from agent.tools import register_debug_tools, register_interview_tools, register_profile_tools
from agent.tools.vault import register_vault_tools
from agent.trace import TraceRecorder
from agent_debug import DeterministicDebugLLM, FileScanRAGManager
from services.workflows.interview import InterviewPlan, TopicCard
from services.workflows.interview_memory_commit import commit_interview_memory
from services.workflows.interview_profile import InterviewProfileStore
from services.workflows.interview_sessions import InterviewSessionStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent golden-session evals")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "agent_eval" / "interview_golden.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "eval-results" / "agent-golden"))
    parser.add_argument("--llm-mode", choices=["fake"], default="fake")
    args = parser.parse_args()
    summary = run_eval(cases_path=Path(args.cases), out_dir=Path(args.out))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed_count"] == summary["case_count"] else 1


def run_eval(*, cases_path: Path, out_dir: Path) -> dict[str, Any]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_case(case=case, out_dir=out_dir) for case in cases]
    summary = summarize_results(results)
    summary["schema_version"] = "agent_eval_summary_v1"
    summary["results"] = results
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_case(*, case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    case_type = str(case.get("type") or "")
    case_id = str(case.get("id") or case_type or "case")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            if case_type == "awaiting_topic":
                actual = eval_awaiting_topic()
            elif case_type == "select_topic":
                actual = eval_select_topic()
            elif case_type == "interviewer":
                actual = eval_interviewer(case=case, root=root, out_dir=out_dir)
            elif case_type == "coach":
                actual = eval_coach(case=case, root=root, out_dir=out_dir)
            elif case_type == "memory_commit":
                actual = eval_memory_commit(root=root)
            elif case_type == "librarian":
                actual = eval_librarian(case=case, root=root, out_dir=out_dir)
            else:
                raise ValueError(f"unsupported case type: {case_type}")
            passed, failures = match_expectations(actual, case.get("expect") or {})
            return {"id": case_id, "type": case_type, "passed": passed, "failures": failures, "actual": actual}
        except Exception as exc:
            return {"id": case_id, "type": case_type, "passed": False, "failures": [str(exc)], "actual": {}}


def eval_awaiting_topic() -> dict[str, Any]:
    machine = build_interview_state_machine(plan=sample_plan(), session_id="eval")
    snapshot = machine.snapshot()
    return {"topic_phase": snapshot.get("topic_phase"), "agent_steps": 0}


def eval_select_topic() -> dict[str, Any]:
    machine = build_interview_state_machine(plan=sample_plan(), session_id="eval")
    result = machine.select_topic(name="MCP 协议", reason="eval topic selection", source="eval")
    snapshot = machine.snapshot()
    return {
        "topic_phase": snapshot.get("topic_phase"),
        "current_topic": snapshot.get("current_topic"),
        "topic_selected": bool(result.get("selected")),
    }


def eval_interviewer(*, case: dict[str, Any], root: Path, out_dir: Path) -> dict[str, Any]:
    vault_root = root / "vault"
    vault_root.mkdir()
    (vault_root / "mcp.md").write_text("# MCP\nHost Client Server", encoding="utf-8")
    runtime = build_runtime(trace_dir=out_dir / "traces")
    app = InterviewInterviewerApp(runtime)
    result, _machine = app.run_turn(
        InterviewTurnRequest(
            query=str(case.get("input") or ""),
            session_id=f"eval-{case.get('id')}",
            interview_plan=sample_plan(),
            interview_state=active_state(),
            vault_root=vault_root,
            rag_manager=FileScanRAGManager(vault_root),
            scope_note_paths=("mcp.md",),
            model="debug-fake",
            tool_mode="json",
            trace_path=str(out_dir / "traces"),
        )
    )
    tool_sequence = [call.name for step in result.steps for call in step.tool_calls]
    metrics = result.state.working.extra.get("derived_metrics") or {}
    return {
        **metrics,
        "tool_call_count": len(tool_sequence),
        "tool_sequence": tool_sequence,
        "trace_path": result.trace_path,
        "stopped_reason": result.stopped_reason,
    }


def eval_librarian(*, case: dict[str, Any], root: Path, out_dir: Path) -> dict[str, Any]:
    vault_root = root / "vault"
    vault_root.mkdir()
    (vault_root / "redis.md").write_text("# Redis Stream\nUse XREADGROUP for consumer groups.", encoding="utf-8")
    (vault_root / "mcp.md").write_text("# MCP\nHost, client, and server are separate roles.", encoding="utf-8")
    runtime = build_runtime(trace_dir=out_dir / "traces")
    app = LibrarianApp(runtime)
    scope_type = str(case.get("scope_type") or "all_vault")
    scope_note_paths = tuple(case.get("scope_note_paths") or ())
    selected_note_paths = tuple(case.get("selected_note_paths") or ())
    result = app.run(
        LibrarianRequest(
            query=str(case.get("input") or "How does Redis Stream work?"),
            scope_type=scope_type,
            scope_note_paths=scope_note_paths,
            selected_note_paths=selected_note_paths,
            vault_root=vault_root,
            rag_manager=FileScanRAGManager(vault_root),
            model="debug-fake",
            tool_mode="json",
            trace_path=str(out_dir / "traces"),
        )
    )
    metrics = result.state.working.extra.get("derived_metrics") or {}
    return {
        **metrics,
        "trace_path": result.trace_path,
        "stopped_reason": result.stopped_reason,
        "schema_valid": bool(result.final_answer),
    }


def eval_coach(*, case: dict[str, Any], root: Path, out_dir: Path) -> dict[str, Any]:
    runtime = build_runtime(trace_dir=out_dir / "traces")
    session_store = InterviewSessionStore(root / "sessions")
    session = session_store.create_session(source_type="folder", source_value="eval", interview_plan=plan_dict(), interview_state=active_state())
    pending = session_store.append_pending_turn(session_id=session["session_id"], user_content=str(case.get("input") or ""))
    profile_store = InterviewProfileStore(root / "profile.json")
    profile_store.save({"schema_version": 2, "weak_points": [], "strong_points": [], "topic_mastery": {}, "communication": {"style": "", "suggestions": []}})
    app = InterviewCoachApp(runtime)
    summary, _result = app.run_review(
        CoachTurnRequest(
            session_id=session["session_id"],
            user_message_id=pending["user_message"]["id"],
            assistant_message_id=pending["assistant_message"]["id"],
            previous_interviewer_question="MCP 是什么？",
            latest_user_answer=str(case.get("input") or ""),
            interviewer_followup="请区分 Host、Client、Server。",
            interview_plan=sample_plan(),
            interview_state=active_state(),
            context_note_paths=("mcp.md",),
            session_store=session_store,
            profile_store=profile_store,
            model="debug-fake",
            tool_mode="json",
            trace_path=str(out_dir / "traces"),
        )
    )
    return {
        "schema_valid": review_schema_valid(summary),
        "signal_count": len(summary.get("profile_signals") or []),
    }


def eval_memory_commit(*, root: Path) -> dict[str, Any]:
    session_store = InterviewSessionStore(root / "sessions")
    profile_store = InterviewProfileStore(root / "profile.json")
    profile_store.save({"schema_version": 2, "weak_points": [], "strong_points": [], "topic_mastery": {}, "communication": {"style": "", "suggestions": []}})
    session = session_store.create_session(source_type="folder", source_value="eval", interview_plan=plan_dict(), interview_state=active_state())
    session_store.append_memory_signal(
        session_id=session["session_id"],
        signal={
            "type": "possible_weak_point",
            "point": "MCP 角色边界不清",
            "topic": "MCP 协议",
            "category": "knowledge_gap",
            "scope_suggestion": "domain",
            "evidence": "用户没有区分 Host、Client、Server。",
            "context_note_paths": ["mcp.md"],
        },
    )
    session_store.append_observation_draft(
        session_id=session["session_id"],
        draft={
            "point": "回答结构缺少角色分层",
            "topic": "MCP 协议",
            "category": "answer_structure",
            "scope": "domain",
            "evidence": "用户只给出一句泛化定义。",
            "context_note_paths": ["mcp.md"],
        },
    )
    _final_review, update = commit_interview_memory(
        session_store=session_store,
        profile_store=profile_store,
        session=session_store.load_session(session["session_id"]),
        reviews=[],
    )
    bridge = update.get("commit_bridge") or {}
    return {
        "consumed_signal_count": bridge.get("consumed_signal_count", 0),
        "consumed_draft_count": bridge.get("consumed_draft_count", 0),
        "schema_valid": update.get("source") == "commit_bridge",
    }


def build_runtime(*, trace_dir: Path) -> AgentRuntime:
    registry = ToolRegistry()
    register_debug_tools(registry)
    register_vault_tools(registry)
    register_interview_tools(registry)
    register_profile_tools(registry)
    return AgentRuntime(
        llm_client=DeterministicDebugLLM(),
        skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
        tool_registry=registry,
        trace_recorder=TraceRecorder(trace_dir),
    )


def sample_plan() -> InterviewPlan:
    return InterviewPlan(
        topics=(TopicCard(name="MCP 协议", coverage=("概念边界", "角色分工", "调用链路"), source_note_paths=("mcp.md",)),),
        suggested_order=("MCP 协议",),
    )


def plan_dict() -> dict[str, Any]:
    return {"topics": [{"name": "MCP 协议", "coverage": ["概念边界", "角色分工", "调用链路"], "source_note_paths": ["mcp.md"]}], "suggested_order": ["MCP 协议"]}


def active_state() -> dict[str, Any]:
    machine = build_interview_state_machine(plan=sample_plan(), session_id="eval")
    return machine.select_topic(name="MCP 协议", reason="eval setup", source="eval")["state"]


def review_schema_valid(summary: dict[str, Any]) -> bool:
    feedback = summary.get("feedback") if isinstance(summary.get("feedback"), dict) else {}
    required_feedback = ["question_requires", "coach_note", "covered", "gaps", "thinking_framework", "interviewer_followup_note"]
    return all(key in feedback for key in required_feedback) and "expression_example" in summary and "profile_signals" in summary


def match_expectations(actual: dict[str, Any], expect: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, expected in expect.items():
        observed = actual.get(key)
        if observed != expected:
            failures.append(f"{key}: expected {expected!r}, got {observed!r}")
    return not failures, failures


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result.get("passed")),
        "routine_state_fetch": 0,
        "tool_call_count": 0,
        "notes_read": 0,
        "profile_recalled": 0,
        "layer_advanced": 0,
        "topic_selected": 0,
        "over_search": 0,
        "signal_count": 0,
        "schema_valid": 0,
        "online_used": 0,
        "search_count": 0,
    }
    for result in results:
        actual = result.get("actual") or {}
        for key in ["routine_state_fetch", "tool_call_count", "notes_read", "profile_recalled", "signal_count", "search_count"]:
            totals[key] += int(actual.get(key) or 0)
        for key in ["layer_advanced", "topic_selected", "over_search", "schema_valid", "online_used"]:
            totals[key] += 1 if actual.get(key) else 0
    return totals


if __name__ == "__main__":
    raise SystemExit(main())
