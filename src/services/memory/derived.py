"""Derived cache rebuild for learner memory v4."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schema import normalize_learner_model
from .types import today_iso, utc_now_iso


def rebuild_derived(model: dict[str, Any], *, today: str | None = None) -> dict[str, Any]:
    day = today or today_iso()
    updated = normalize_learner_model(deepcopy(model))
    revision = int(updated.get("canonical_revision") or 0)
    groups: dict[str, dict[str, Any]] = {}

    for belief in (updated.get("learner_items") or updated.get("beliefs") or []):
        if not isinstance(belief, dict):
            continue
        if str(belief.get("lifecycle") or "") != "active":
            continue
        if belief.get("improved"):
            continue
        key, domain_entry = _domain_key_and_entry(belief)
        bucket = groups.setdefault(
            key,
            {
                **domain_entry,
                "active_belief_ids": [],
                "due_belief_ids": [],
            },
        )
        belief_id = str(belief.get("id") or "").strip()
        if not belief_id:
            continue
        bucket["active_belief_ids"].append(belief_id)
        sr = belief.get("sr") or {}
        next_review = str(sr.get("next_review") or "2000-01-01")
        if next_review <= day:
            bucket["due_belief_ids"].append(belief_id)

    domains: list[dict[str, Any]] = []
    inject_blurbs: dict[str, str] = {}
    for key, bucket in sorted(groups.items(), key=lambda item: item[0]):
        active_count = len(bucket["active_belief_ids"])
        due_count = len(bucket["due_belief_ids"])
        coverage = "partial" if active_count <= 2 else "broad"
        confidence = round(max(0.35, min(0.95, 1.0 - active_count * 0.12 - due_count * 0.08)), 2)
        domain = {
            "scope_path": bucket.get("scope_path") or key,
            "plan_topic": bucket.get("plan_topic") or "",
            "coverage": coverage,
            "confidence": confidence,
            "known_facets": [],
            "unknown_facets": [],
            "active_belief_ids": list(bucket["active_belief_ids"]),
            "due_belief_ids": list(bucket["due_belief_ids"]),
            "user_override": None,
        }
        domains.append(domain)
        topic_label = str(domain.get("plan_topic") or domain.get("scope_path") or key)
        inject_blurbs[key] = (
            f"{topic_label} 域：{active_count} 处 active 弱项"
            + (f"，{due_count} 处 due" if due_count else "")
            + "。"
        )

    updated["derived"] = {
        "schema_version": 1,
        "generation": revision,
        "updated_at": utc_now_iso(),
        "stale": False,
        "domains": domains,
        "inject_blurbs": inject_blurbs,
    }
    return updated


def _domain_key_and_entry(belief: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    anchor = belief.get("domain_anchor") or {}
    scope_path = anchor.get("scope_path")
    if isinstance(scope_path, list):
        scope_key = "/".join(str(item) for item in scope_path if str(item).strip())
    else:
        scope_key = str(scope_path or "").strip()
    topic = str(anchor.get("topic") or belief.get("topic") or "").strip()
    if not scope_key and topic:
        scope_key = topic
    if not scope_key:
        scope_key = "universal"
    return scope_key, {
        "scope_path": scope_key,
        "plan_topic": topic,
    }
