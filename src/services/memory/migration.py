"""Migration from legacy interview_profile v3 to learner memory v4."""

from __future__ import annotations

import hashlib
from typing import Any

from .schema import normalize_learner_model


def stable_weak_point_id(weak_point: dict[str, Any]) -> str:
    for key in ("id", "weak_id", "uid"):
        value = str(weak_point.get(key) or "").strip()
        if value:
            return value
    seed = "|".join(
        [
            str(weak_point.get("topic") or ""),
            str(weak_point.get("planned_layer") or ""),
            str(weak_point.get("point") or ""),
        ]
    )
    return "weak-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _legacy_evidence_refs(weak_point: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    session_ids = weak_point.get("source_session_ids") or weak_point.get("sessions") or []
    if isinstance(session_ids, str):
        session_ids = [session_ids]
    session_ids = [str(item) for item in session_ids if str(item).strip()]

    evidence = weak_point.get("evidence")
    evidence_items = evidence if isinstance(evidence, list) else [evidence]
    for item in evidence_items:
        if isinstance(item, dict):
            summary = str(item.get("summary") or item.get("evidence") or "").strip()
            session_id = str(item.get("session_id") or (session_ids[0] if session_ids else ""))
            at = str(item.get("at") or weak_point.get("last_seen") or "")
        else:
            summary = str(item or "").strip()
            session_id = session_ids[0] if session_ids else ""
            at = str(weak_point.get("last_seen") or "")
        if not summary and not session_id:
            continue
        refs.append(
            {
                "source_kind": "interview",
                "session_id": session_id,
                "turn_id": "",
                "review_run_id": "",
                "card_id": "",
                "at": at,
                "summary": summary,
            }
        )
    for session_id in session_ids:
        if not any(ref.get("session_id") == session_id for ref in refs):
            refs.append(
                {
                    "source_kind": "interview",
                    "session_id": session_id,
                    "turn_id": "",
                    "review_run_id": "",
                    "card_id": "",
                    "at": str(weak_point.get("last_seen") or ""),
                    "summary": "",
                }
            )
    return refs


def migrate_v3_profile_to_v4(profile_v3: dict[str, Any] | None) -> dict[str, Any]:
    profile_v3 = profile_v3 or {}
    weak_points = profile_v3.get("weak_points") or []
    beliefs: list[dict[str, Any]] = []
    for index, weak_point in enumerate(weak_points):
        if not isinstance(weak_point, dict):
            continue
        belief = {
            "id": stable_weak_point_id(weak_point),
            "kind": "standard",
            "lifecycle": "active",
            "point": str(weak_point.get("point") or weak_point.get("text") or "").strip(),
            "category": weak_point.get("category") or "knowledge_gap",
            "scope": weak_point.get("scope") or "domain",
            "topic": weak_point.get("topic") or "",
            "planned_layer": weak_point.get("planned_layer") or "",
            "domain_anchor": weak_point.get("domain_anchor") or {},
            "source_note_paths": weak_point.get("source_note_paths") or [],
            "source_session_ids": weak_point.get("source_session_ids") or [],
            "source_kinds": ["interview"],
            "evidence_refs": _legacy_evidence_refs(weak_point),
            "times_seen": weak_point.get("times_seen", 1),
            "first_seen": weak_point.get("first_seen") or "",
            "last_seen": weak_point.get("last_seen") or "",
            "improved": bool(weak_point.get("improved", False)),
            "improved_at": weak_point.get("improved_at") or "",
            "sr": weak_point.get("sr") or {},
        }
        beliefs.append(belief)

    model = {
        "schema_version": 4,
        "canonical_revision": 1,
        "updated_at": str(profile_v3.get("updated_at") or ""),
        "beliefs": beliefs,
        "procedures": [],
        "strong_points": profile_v3.get("strong_points") or [],
        "commitments": [],
        "derived": {
            "stale": True,
            "updated_at": "",
            "domains": [],
        },
        "legacy": {
            "communication": profile_v3.get("communication") or {},
            "topic_mastery": profile_v3.get("topic_mastery") or {},
        },
    }
    return normalize_learner_model(model)
