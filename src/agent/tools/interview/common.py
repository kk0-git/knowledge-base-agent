from __future__ import annotations

from typing import Any

from agent.interview.state import InterviewStateMachine, build_interview_state_machine, plan_topics_summary
from agent.schema import WorkingMemory


def get_state_machine(ctx: Any) -> InterviewStateMachine:
    machine = getattr(ctx, "state_machine", None)
    if machine is None:
        raise ValueError("state_machine is required for interview state tools")
    return machine


def sync_working_from_machine(working: WorkingMemory, machine: InterviewStateMachine) -> None:
    snapshot = machine.snapshot()
    working.session_id = snapshot.get("session_id") or working.session_id
    working.current_topic = snapshot.get("current_topic")
    working.current_layer_index = int(snapshot.get("current_layer_index") or 0)
    working.follow_up_count = int(snapshot.get("follow_up_count") or 0)
    working.plan_topic_names = [str(item.get("name") or "") for item in snapshot.get("plan_topics", []) if item.get("name")]
    working.extra["interview_state"] = snapshot


def ensure_state_machine(ctx: Any) -> InterviewStateMachine:
    machine = getattr(ctx, "state_machine", None)
    if machine is not None:
        return machine
    machine = build_interview_state_machine(
        plan=getattr(ctx, "interview_plan", None),
        state_payload=getattr(ctx, "interview_state", None),
        session_id=getattr(ctx, "session_id", ""),
    )
    ctx.state_machine = machine
    return machine


def topics_payload(plan: Any, *, include_sources: bool) -> dict[str, Any]:
    topics = plan_topics_summary(plan, include_sources=include_sources)
    suggested = plan.get("suggested_order", ()) if isinstance(plan, dict) else getattr(plan, "suggested_order", ())
    return {
        "topic_count": len(topics),
        "topics": topics,
        "suggested_order": list(suggested or []),
    }
