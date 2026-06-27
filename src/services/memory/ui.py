"""UI helpers for learner memory pages."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote


def enrich_memory_item_for_ui(item: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(item)
    refs = enriched.get("evidence_refs") or []
    enriched["evidence_links"] = evidence_links_from_refs(refs if isinstance(refs, list) else [])
    return enriched


def evidence_links_from_refs(refs: list[dict[str, Any]]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        summary = str(ref.get("summary") or "").strip()
        source_kind = str(ref.get("source_kind") or "interview").strip() or "interview"
        session_id = str(ref.get("session_id") or "").strip()
        review_run_id = str(ref.get("review_run_id") or "").strip()
        turn_id = str(ref.get("turn_id") or "").strip()
        card_id = str(ref.get("card_id") or "").strip()
        href = ""
        label = ""
        if source_kind == "answer" and session_id:
            href = f"/?mode=answer&session_id={quote(session_id)}"
            label = f"问答 {session_id}"
        elif session_id:
            href = f"/?mode=interview&session_id={quote(session_id)}"
            label = f"面试 {session_id}"
        elif review_run_id:
            href = f"/review?run={quote(review_run_id)}"
            label = f"复习 {review_run_id}"
        elif summary:
            label = source_kind
        if not label and not summary:
            continue
        links.append(
            {
                "summary": summary,
                "href": href,
                "label": label,
                "source_kind": source_kind,
                "session_id": session_id,
                "turn_id": turn_id,
                "review_run_id": review_run_id,
                "card_id": card_id,
            }
        )
    return links
