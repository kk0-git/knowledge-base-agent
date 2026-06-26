"""Lightweight learner memory quality metrics."""

from __future__ import annotations

from typing import Any

from .schema import normalize_learner_model
from .types import today_iso


def memory_metrics(model: dict[str, Any], *, today: str | None = None) -> dict[str, Any]:
    day = today or today_iso()
    normalized = normalize_learner_model(model)
    beliefs = [item for item in normalized.get("beliefs") or [] if isinstance(item, dict)]
    procedures = [item for item in normalized.get("procedures") or [] if isinstance(item, dict)]

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
    }


def _count_lifecycle(items: list[dict[str, Any]], lifecycle: str) -> int:
    return sum(1 for item in items if item.get("lifecycle") == lifecycle)
