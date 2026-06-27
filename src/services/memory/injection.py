"""Prompt injection helpers for learner memory."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .bridge import learner_model_to_profile_view


_FACET_PROBE_HINTS: dict[str, str] = {
    "knowledge": "probe definition and underlying mechanism",
    "behavior": "probe how they organize, frame, and express the answer",
}

# §8: Per-reader injection configuration — single rendering function, config-driven differences.
READER_CONFIG: dict[str, dict[str, Any]] = {
    "interviewer": {
        "belief_budget": 5,
        "procedure_budget": 2,
        "due_mode": "inline",
        "derived": True,
        "derived_header": "### Derived domain summary",
        "commitment": True,
        "rich_belief_format": True,
        "layer_boost": True,
        "instruction": "Private background from prior sessions. Use quietly to shape probes; do not mention memory, weak points, or review history unless the user asks.",
        "boundary": "Let the current layer objective and the user's live answer drive the next probe.\nDo not read these notes aloud or turn them into a checklist.\nDomain weak-point bodies for the current layer are still available via recall_profile when profile_layer_counts indicate matches.",
    },
    "reviewer": {
        "belief_budget": 5,
        "procedure_budget": 0,
        "due_mode": "inline",
        "derived": True,
        "derived_header": "### Derived domain summary",
        "commitment": True,
        "rich_belief_format": True,
        "layer_boost": False,
        "instruction": "Private background from prior sessions. Use quietly to shape review questions; do not mention memory or weak points unless the user asks.",
        "boundary": "Let the review plan and the user's live answers drive the dialogue.\nDo not read these notes aloud or turn them into a checklist.",
    },
    "librarian": {
        "belief_budget": 3,
        "procedure_budget": 2,
        "due_mode": "marker_2",
        "derived": True,
        "derived_header": "",
        "commitment": False,
        "rich_belief_format": False,
        "layer_boost": False,
        "instruction": "Private background from prior sessions. Use quietly; do not mention memory, weak points, or review history unless the user asks.",
        "boundary": "Let the user's current question and note evidence drive the answer.\nDo not force these notes into the reply or treat them as a checklist.",
    },
    "coach": {
        "belief_budget": 0,
        "procedure_budget": 0,
        "due_mode": "none",
        "derived": False,
        "derived_header": "",
        "commitment": False,
        "rich_belief_format": False,
        "layer_boost": False,
        "instruction": "",
        "boundary": "",
    },
}


def render_memory_context(
    *,
    model: dict[str, Any],
    reader: str,
    scope_note_paths: tuple[str, ...] = (),
    scope_value: str = "",
    current_topic: str = "",
    planned_layer: str = "",
) -> str:
    """Unified memory injection for all reader types (§8)."""
    config = READER_CONFIG.get(reader)
    if config is None:
        return ""

    # Collect and select beliefs
    active_candidates = _collect_scoped_active_beliefs(
        model,
        scope_note_paths=scope_note_paths,
        scope_value=scope_value or current_topic,
        current_topic=current_topic,
    )
    ranked = _sort_weak_points_for_prompt(active_candidates)
    if config["layer_boost"] and planned_layer:
        ranked = _boost_layer_matches(ranked, planned_layer)

    due_mode = config["due_mode"]
    if due_mode == "inline":
        max_due_markers = None
    elif due_mode.startswith("marker_"):
        max_due_markers = int(due_mode.split("_")[1])
    else:
        max_due_markers = 0

    selected = _select_beliefs_from_ranked(
        ranked, max_beliefs=config["belief_budget"], max_due_markers=max_due_markers
    )

    procedures = (
        _active_procedures(model, limit=config["procedure_budget"])
        if config["procedure_budget"] > 0
        else []
    )
    derived = model.get("derived") or {}
    blurb = (
        _pick_derived_blurb(derived, scope_value or current_topic, scope_note_paths)
        if config["derived"]
        else ""
    )
    commitments = _commitments_for_prompt(model) if config["commitment"] else []

    if not selected and not procedures and not blurb and not commitments:
        return ""

    # Render sections
    lines: list[str] = []
    if config["instruction"]:
        lines.extend(["## Learner Memory Background", config["instruction"], ""])

    if blurb:
        if config["derived_header"]:
            lines.extend([config["derived_header"], blurb, ""])
        else:
            lines.extend([blurb, ""])

    if selected:
        if config["rich_belief_format"]:
            lines.append("### Active beliefs (probe-oriented)")
            for weak, is_due in selected:
                lines.append(_format_interviewer_belief_block(weak, is_due=is_due))
        else:
            lines.append("Active beliefs relevant to this scope:")
            for weak, is_due in selected:
                suffix = " [due]" if is_due else ""
                lines.append(f"- {_format_belief_for_prompt(weak)}{suffix}")
        lines.append("")

    if commitments:
        lines.append("### User commitments")
        for item in commitments:
            lines.append(f"- {item}")
        lines.append("")

    if procedures:
        lines.append("### Interaction preferences")
        for proc in procedures:
            title = str(proc.get("title") or proc.get("point") or "").strip()
            if title:
                lines.append(f"- {title}")
        lines.append("")

    if config["boundary"]:
        lines.extend(["## Memory Use Boundary", config["boundary"]])

    return "\n".join(lines)


def render_librarian_memory_context(
    *,
    model: dict[str, Any],
    scope_note_paths: tuple[str, ...] = (),
    scope_value: str = "",
) -> str:
    return render_memory_context(
        model=model,
        reader="librarian",
        scope_note_paths=scope_note_paths,
        scope_value=scope_value,
    )


def render_interviewer_memory_context(
    *,
    model: dict[str, Any],
    current_topic: str = "",
    planned_layer: str = "",
    scope_note_paths: tuple[str, ...] = (),
) -> str:
    return render_memory_context(
        model=model,
        reader="interviewer",
        scope_note_paths=scope_note_paths,
        scope_value=current_topic,
        current_topic=current_topic,
        planned_layer=planned_layer,
    )


def _collect_scoped_active_beliefs(
    model: dict[str, Any],
    *,
    scope_note_paths: tuple[str, ...],
    scope_value: str,
    current_topic: str = "",
) -> list[dict[str, Any]]:
    from services.workflows.interview_profile import (
        domain_relevance_for_current,
        is_injectable_weak_point,
    )

    profile = learner_model_to_profile_view(model)
    weak_points = profile.get("weak_points") or []
    topic_card = _synthetic_topic_card(scope_note_paths, scope_value or current_topic)
    topic = current_topic or scope_value

    active_candidates: list[dict[str, Any]] = []
    for weak in weak_points:
        if not is_injectable_weak_point(weak):
            continue
        if weak.get("scope") == "universal":
            active_candidates.append(weak)
            continue
        relevance = domain_relevance_for_current(
            weak,
            current_topic_card=topic_card,
            current_topic=topic,
        )
        if relevance in {"strong", "medium", "weak"}:
            active_candidates.append(weak)
    return active_candidates


def _sort_weak_points_for_prompt(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from services.workflows.interview_profile import sort_weak_points_for_prompt

    return sort_weak_points_for_prompt(items)


def _select_beliefs_from_ranked(
    ranked: list[dict[str, Any]],
    *,
    max_beliefs: int,
    max_due_markers: int | None,
) -> list[tuple[dict[str, Any], bool]]:
    today = date.today().isoformat()
    selected: list[tuple[dict[str, Any], bool]] = []
    due_marked = 0
    for weak in ranked:
        if len(selected) >= max_beliefs:
            break
        sr = weak.get("sr") or {}
        is_due = str(sr.get("next_review") or "2000-01-01") <= today
        if is_due and max_due_markers is not None and due_marked >= max_due_markers:
            continue
        if is_due:
            due_marked += 1
        selected.append((weak, is_due))
    return selected


def _boost_layer_matches(
    ranked: list[dict[str, Any]],
    planned_layer: str,
) -> list[dict[str, Any]]:
    layer_key = str(planned_layer or "").strip().lower()
    if not layer_key:
        return ranked
    matched = [weak for weak in ranked if str(weak.get("planned_layer") or "").strip().lower() == layer_key]
    others = [weak for weak in ranked if str(weak.get("planned_layer") or "").strip().lower() != layer_key]
    return matched + others


def _active_procedures(model: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    return [
        item
        for item in (model.get("assistant_items") or model.get("procedures") or [])
        if isinstance(item, dict) and str(item.get("lifecycle") or "") == "active"
    ][:limit]


def _commitments_for_prompt(model: dict[str, Any], *, max_items: int = 3) -> list[str]:
    lines: list[str] = []
    for item in reversed(model.get("commitments") or []):
        if not isinstance(item, dict):
            continue
        note = str(item.get("note") or "").strip()
        action = str(item.get("action") or "").strip()
        if not note and not action:
            continue
        label = note or action.replace("_", " ")
        if label and label not in lines:
            lines.append(label)
        if len(lines) >= max_items:
            break
    return lines


def _format_belief_for_prompt(weak: dict[str, Any]) -> str:
    if str(weak.get("kind") or "") == "confusion_pair":
        left = str(weak.get("left") or "").strip()
        right = str(weak.get("right") or "").strip()
        distinction = str(weak.get("distinction") or "").strip()
        label = f"{left} vs {right}" if left and right else str(weak.get("point") or "").strip()
        return f"{label}: {distinction}" if distinction else label
    return str(weak.get("point") or "").strip()


def _probe_hint_for_weak(weak: dict[str, Any]) -> str:
    explicit = str(weak.get("probe_hint") or "").strip()
    if explicit:
        return explicit
    facet = str(weak.get("facet") or weak.get("category") or "").strip()
    if facet in _FACET_PROBE_HINTS:
        return _FACET_PROBE_HINTS[facet]
    return f"probe {facet}" if facet else "probe for signal on this layer"


def _latest_evidence_summary(weak: dict[str, Any]) -> str:
    refs = weak.get("evidence_refs") or []
    if refs and isinstance(refs[-1], dict):
        return str(refs[-1].get("summary") or "").strip()
    legacy = weak.get("evidence") or []
    if isinstance(legacy, list) and legacy:
        return str(legacy[-1] or "").strip()
    if isinstance(legacy, str):
        return legacy.strip()
    return ""


def _format_interviewer_belief_block(weak: dict[str, Any], *, is_due: bool) -> str:
    point = _format_belief_for_prompt(weak)
    due_suffix = " [due]" if is_due else ""
    layer = str(weak.get("planned_layer") or "").strip()
    probe_hint = _probe_hint_for_weak(weak)
    meta_parts = [f"probe: {probe_hint}"]
    if layer:
        meta_parts.append(f"layer: {layer}")
    lines = [f"- {point}{due_suffix} ({'; '.join(meta_parts)})"]
    evidence = _latest_evidence_summary(weak)
    if evidence:
        lines.append(f"  latest evidence: {evidence}")
    return "\n".join(lines)


def _synthetic_topic_card(scope_note_paths: tuple[str, ...], scope_value: str) -> dict[str, Any]:
    paths = [str(path).replace("\\", "/") for path in scope_note_paths if str(path).strip()]
    return {
        "name": scope_value,
        "source_note_paths": paths[:12],
    }


def _topic_from_paths(scope_note_paths: tuple[str, ...]) -> str:
    if not scope_note_paths:
        return ""
    return str(Path(scope_note_paths[0]).parent).replace("\\", "/")


def _pick_derived_blurb(
    derived: dict[str, Any],
    scope_value: str,
    scope_note_paths: tuple[str, ...],
) -> str:
    blurbs = derived.get("inject_blurbs") or {}
    if not isinstance(blurbs, dict):
        return ""
    if scope_value and blurbs.get(scope_value):
        return str(blurbs[scope_value])
    for path in scope_note_paths:
        parent = str(Path(path).parent).replace("\\", "/")
        if blurbs.get(parent):
            return str(blurbs[parent])
    if blurbs:
        return str(next(iter(blurbs.values())))
    domains = derived.get("domains") or []
    if domains and isinstance(domains[0], dict):
        topic = str(domains[0].get("plan_topic") or domains[0].get("scope_path") or "")
        active_count = len(domains[0].get("active_belief_ids") or [])
        due_count = len(domains[0].get("due_belief_ids") or [])
        if topic:
            return f"{topic} 域：{active_count} 处 active 弱项" + (f"，{due_count} 处 due。" if due_count else "。")
    return ""
