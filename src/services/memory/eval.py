"""Lightweight learner memory quality metrics."""

from __future__ import annotations

from typing import Any

from .schema import normalize_learner_model
from .types import today_iso


def memory_metrics(model: dict[str, Any], *, today: str | None = None) -> dict[str, Any]:
    day = today or today_iso()
    normalized = normalize_learner_model(model)
    beliefs = [item for item in (normalized.get("learner_items") or normalized.get("beliefs") or []) if isinstance(item, dict)]
    procedures = [item for item in (normalized.get("assistant_items") or normalized.get("procedures") or []) if isinstance(item, dict)]

    active_beliefs = [item for item in beliefs if item.get("lifecycle") == "active" and not item.get("improved")]
    due_active_beliefs = [
        item
        for item in active_beliefs
        if str((item.get("sr") or {}).get("next_review") or "2000-01-01") <= day
    ]
    active_procedures = [item for item in procedures if item.get("lifecycle") == "active"]
    derived = normalized.get("derived") or {}
    revision = int(normalized.get("canonical_revision") or 0)
    generation = int(derived.get("generation") or 0)
    belief_counts = {
        "candidate": _count_lifecycle(beliefs, "candidate"),
        "active": _count_lifecycle(beliefs, "active"),
        "archived": _count_lifecycle(beliefs, "archived"),
        "confusion_pair": sum(1 for item in beliefs if item.get("kind") == "confusion_pair"),
        "due_active": len(due_active_beliefs),
    }
    procedure_counts = {
        "candidate": _count_lifecycle(procedures, "candidate"),
        "active": _count_lifecycle(procedures, "active"),
        "archived": _count_lifecycle(procedures, "archived"),
    }
    has_derived_blurb = bool((derived.get("inject_blurbs") or {}) or derived.get("domains"))
    candidate_total = belief_counts["candidate"] + procedure_counts["candidate"]
    archived_total = belief_counts["archived"] + procedure_counts["archived"]
    generation_mismatch = generation != revision
    derived_stale = bool(derived.get("stale", True))

    # §10 outcome metrics — evidence-based signals for scoring calibration
    outcome = _outcome_metrics(normalized, beliefs, day)

    return {
        "canonical_revision": revision,
        "beliefs": belief_counts,
        "procedures": procedure_counts,
        "backlog": {
            "candidate_total": candidate_total,
            "archived_total": archived_total,
            "due_active_beliefs": len(due_active_beliefs),
        },
        "injection_preview": {
            "librarian_active_beliefs": min(len(active_beliefs), 3),
            "librarian_due_markers": min(len(due_active_beliefs), 2),
            "librarian_procedures": min(len(active_procedures), 2),
            "interviewer_active_beliefs": min(len(active_beliefs), 5),
            "interviewer_due_markers": min(len(due_active_beliefs), 5),
            "interviewer_procedures": min(len(active_procedures), 2),
            "has_derived_blurb": has_derived_blurb,
        },
        "derived": {
            "stale": derived_stale,
            "generation": generation,
            "generation_mismatch": generation_mismatch,
        },
        "health": {
            "candidate_backlog": candidate_total,
            "archived_total": archived_total,
            "derived_stale": derived_stale,
            "generation_mismatch": generation_mismatch,
        },
        "outcome": outcome,
    }


def _outcome_metrics(
    model: dict[str, Any],
    beliefs: list[dict[str, Any]],
    today: str,
) -> dict[str, Any]:
    """§10: Outcome signals for calibrating scoring weights and thresholds."""
    commitments = model.get("commitments") or []

    # candidate_confirm_rate: ratio of confirmed vs denied candidates
    confirm_count = 0
    deny_count = 0
    mis_merge_count = 0
    mis_merge_notes: list[str] = []

    for commit in commitments:
        if not isinstance(commit, dict):
            continue
        action = str(commit.get("action") or "").strip()
        note = str(commit.get("note") or "").strip().lower()

        if action == "confirm_candidate":
            confirm_count += 1
        elif action == "deny_belief":
            deny_count += 1
            # Detect mis-merge: user says two items were incorrectly merged
            if _is_mis_merge_signal(note):
                mis_merge_count += 1
                if note:
                    mis_merge_notes.append(note[:120])

    total_confirms = confirm_count + deny_count
    candidate_confirm_rate = round(confirm_count / total_confirms, 3) if total_confirms > 0 else None

    # injected_belief_hit: requires interview turn-level instrumentation (not yet wired).
    # Tracked here as a placeholder; actual collection needs hook in interview turn review.
    injected_belief_hit = None  # TBD: wire via interview turn review → model.derived.outcome

    return {
        "candidate_confirm_rate": candidate_confirm_rate,
        "candidate_confirms": confirm_count,
        "candidate_denies": deny_count,
        "mis_merge_flag": mis_merge_count,
        "mis_merge_notes": mis_merge_notes[:5],
        "injected_belief_hit": injected_belief_hit,
    }


_MIS_MERGE_KEYWORDS = [
    "不是一回事", "不一样", "不同概念", "合并错了",
    "not the same", "different", "wrong merge", "should not merge",
    "重复", "duplicate",
]


def _is_mis_merge_signal(note: str) -> bool:
    return any(keyword in note for keyword in _MIS_MERGE_KEYWORDS)


def _count_lifecycle(items: list[dict[str, Any]], lifecycle: str) -> int:
    return sum(1 for item in items if item.get("lifecycle") == lifecycle)
