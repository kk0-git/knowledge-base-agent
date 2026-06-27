"""Learner memory v4 schema helpers."""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any

from .types import (
    default_sr,
    ensure_list,
    normalize_belief_kind,
    normalize_facet,
    normalize_float,
    normalize_int,
    normalize_lifecycle,
    normalize_scope,
    normalize_source_kind,
    unique_strings,
)


SCHEMA_VERSION = 5


def default_learner_model() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_revision": 0,
        "updated_at": "",
        "learner_items": [],
        "assistant_items": [],
        "strong_points": [],
        "commitments": [],
        "derived": {
            "stale": True,
            "updated_at": "",
            "domains": [],
        },
        "legacy": {
            "communication": {},
            "topic_mastery": {},
        },
    }


def normalize_domain_anchor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    scope_path = value.get("scope_path")
    if isinstance(scope_path, str):
        scope_path = [scope_path]
    return {
        "topic": str(value.get("topic") or "").strip(),
        "scope_path": unique_strings(scope_path or []),
        "source_note_paths": unique_strings(value.get("source_note_paths") or []),
        "evidence_terms": unique_strings(value.get("evidence_terms") or []),
    }


def normalize_sr(value: Any) -> dict[str, Any]:
    base = default_sr()
    if isinstance(value, dict):
        base.update(value)
    base["interval_days"] = normalize_int(base.get("interval_days"), 1, minimum=1)
    base["ease_factor"] = normalize_float(base.get("ease_factor"), 2.5, minimum=1.3)
    base["repetitions"] = normalize_int(base.get("repetitions"), 0, minimum=0)
    base["last_reviewed"] = str(base.get("last_reviewed") or "")
    base["last_outcome"] = str(base.get("last_outcome") or "")
    base["next_review"] = str(base.get("next_review") or "")
    return base


def normalize_evidence_refs(value: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in ensure_list(value):
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            refs.append(
                {
                    "source_kind": "interview",
                    "session_id": "",
                    "turn_id": "",
                    "review_run_id": "",
                    "card_id": "",
                    "at": "",
                    "summary": text,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        refs.append(
            {
                "source_kind": normalize_source_kind(item.get("source_kind")),
                "session_id": str(item.get("session_id") or ""),
                "turn_id": str(item.get("turn_id") or ""),
                "review_run_id": str(item.get("review_run_id") or ""),
                "card_id": str(item.get("card_id") or ""),
                "at": str(item.get("at") or ""),
                "summary": str(item.get("summary") or item.get("evidence") or "").strip(),
            }
        )
    return refs


def normalize_belief(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    belief = dict(value)
    kind = normalize_belief_kind(belief.get("kind"))
    left = str(belief.get("left") or "").strip()
    right = str(belief.get("right") or "").strip()
    distinction = str(belief.get("distinction") or "").strip()
    point = str(belief.get("point") or "").strip()
    if kind == "confusion_pair" and not point and left and right:
        point = f"{left} vs {right}"

    normalized = dict(belief)
    normalized.update(
        {
            "id": str(belief.get("id") or f"wp-{uuid.uuid4().hex[:12]}"),
            "kind": kind,
            "lifecycle": normalize_lifecycle(belief.get("lifecycle")),
            "point": point,
            "left": left,
            "right": right,
            "distinction": distinction,
            "category": normalize_facet(belief.get("category") or belief.get("facet")),
            "scope": normalize_scope(belief.get("scope")),
            "topic": str(belief.get("topic") or "").strip(),
            "planned_layer": str(belief.get("planned_layer") or "").strip(),
            "domain_anchor": normalize_domain_anchor(belief.get("domain_anchor")),
            "source_note_paths": unique_strings(belief.get("source_note_paths") or []),
            "source_session_ids": unique_strings(belief.get("source_session_ids") or []),
            "source_kinds": [normalize_source_kind(v) for v in unique_strings(belief.get("source_kinds") or [])]
            or ["interview"],
            "evidence_refs": normalize_evidence_refs(
                belief.get("evidence_refs") if "evidence_refs" in belief else belief.get("evidence")
            ),
            "times_seen": normalize_int(belief.get("times_seen"), 1, minimum=0),
            "first_seen": str(belief.get("first_seen") or ""),
            "last_seen": str(belief.get("last_seen") or ""),
            "improved": bool(belief.get("improved", False)),
            "improved_at": str(belief.get("improved_at") or ""),
            "sr": normalize_sr(belief.get("sr")),
        }
    )
    return normalized


def normalize_procedure(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    procedure = dict(value)
    procedure.setdefault("id", f"proc-{uuid.uuid4().hex[:12]}")
    procedure.setdefault("lifecycle", "candidate")
    procedure["lifecycle"] = normalize_lifecycle(procedure.get("lifecycle"))
    key = str(procedure.get("procedure_key") or procedure.get("key") or "").strip()
    procedure["procedure_key"] = key
    procedure["key"] = key
    procedure["scope"] = normalize_scope(procedure.get("scope"), default="universal")
    procedure["title"] = str(procedure.get("title") or procedure.get("point") or "").strip()
    procedure["description"] = str(procedure.get("description") or "").strip()
    procedure["steps"] = [str(step) for step in ensure_list(procedure.get("steps")) if str(step).strip()]
    procedure["evidence_refs"] = normalize_evidence_refs(procedure.get("evidence_refs") or [])
    procedure["source_kinds"] = [normalize_source_kind(v) for v in unique_strings(procedure.get("source_kinds") or [])]
    procedure["source_session_ids"] = unique_strings(procedure.get("source_session_ids") or [])
    procedure["times_seen"] = normalize_int(procedure.get("times_seen"), 1, minimum=0)
    procedure["first_seen"] = str(procedure.get("first_seen") or "")
    procedure["last_seen"] = str(procedure.get("last_seen") or "")
    return procedure


def normalize_commitment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    commitment = dict(value)
    commitment.setdefault("id", f"commit-{uuid.uuid4().hex[:12]}")
    commitment.setdefault("action", "")
    commitment.setdefault("belief_id", "")
    commitment.setdefault("target_id", "")
    commitment.setdefault("at", "")
    commitment.setdefault("note", "")
    return commitment


def normalize_learner_model(model: Any) -> dict[str, Any]:
    if not isinstance(model, dict):
        model = {}
    defaults = default_learner_model()
    normalized = deepcopy(defaults)
    normalized.update(model)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["canonical_revision"] = normalize_int(normalized.get("canonical_revision"), 0, minimum=0)
    normalized["updated_at"] = str(normalized.get("updated_at") or "")
    # Read v5 keys first, fall back to v4 keys for in-flight migration
    normalized["learner_items"] = [
        normalize_belief(item)
        for item in ensure_list(normalized.get("learner_items") or normalized.pop("beliefs", []))
    ]
    normalized["assistant_items"] = [
        normalize_procedure(item)
        for item in ensure_list(normalized.get("assistant_items") or normalized.pop("procedures", []))
    ]
    normalized["strong_points"] = [dict(item) for item in ensure_list(normalized.get("strong_points")) if isinstance(item, dict)]
    normalized["commitments"] = [normalize_commitment(item) for item in ensure_list(normalized.get("commitments"))]

    derived = normalized.get("derived") if isinstance(normalized.get("derived"), dict) else {}
    merged_derived = deepcopy(defaults["derived"])
    merged_derived.update(derived)
    merged_derived["stale"] = bool(merged_derived.get("stale", True))
    normalized["derived"] = merged_derived

    legacy = normalized.get("legacy") if isinstance(normalized.get("legacy"), dict) else {}
    merged_legacy = deepcopy(defaults["legacy"])
    merged_legacy.update(legacy)
    normalized["legacy"] = merged_legacy
    return normalized
