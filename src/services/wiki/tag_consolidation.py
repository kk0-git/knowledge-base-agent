from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.wiki.schema import WikiState, WikiTagRecord


TAG_CONSOLIDATION_PROMPT = """You are reviewing the tag taxonomy of a personal knowledge base.

Tags are used for navigation and wiki synthesis. Your task is to propose safe cleanup actions.

Important:
- This is proposal-only. Do not assume changes are already applied.
- Be conservative. Related tags are not necessarily duplicates.
- Do not merge a focused knowledge tag into an overview/navigation parent tag.
- Prefer stable 2-level tags: existing_root/specific_tag.
- If a tag has no natural parent, suggest deleting it rather than forcing it.

You may propose:
- merge: source tag should be merged into target tag.
- delete: tag should be removed from generated tag state.
- rename: tag should be renamed to a clearer path.
- mark_overview: tag is too broad for detailed wiki pages and should be overview.
- mark_skip: tag is low-value and should not generate wiki.

Return only JSON:
{
  "proposals": [
    {
      "action": "merge",
      "source_tag": "web框架/jsp",
      "target_tag": "java/jsp",
      "reason": "short reason",
      "confidence": 0.74
    }
  ]
}
"""


@dataclass(frozen=True)
class TagCleanupProposal:
    action: str
    source_tag: str
    target_tag: str | None = None
    reason: str = ""
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "llm"


class LLMTagConsolidator:
    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def propose_cleanup(
        self,
        *,
        state: WikiState,
        vault_root: Path,
        max_notes_per_tag: int = 5,
    ) -> list[TagCleanupProposal]:
        response = self.client.complete(
            LLMRequest(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    LLMMessage(role="system", content=TAG_CONSOLIDATION_PROMPT),
                    LLMMessage(
                        role="user",
                        content=build_consolidation_user_prompt(
                            state=state,
                            vault_root=vault_root,
                            max_notes_per_tag=max_notes_per_tag,
                        ),
                    ),
                ],
            )
        )
        payload = parse_json_object(response.content)
        raw_proposals = payload.get("proposals", [])
        if not isinstance(raw_proposals, list):
            return []

        proposals: list[TagCleanupProposal] = []
        existing_tags = set(state.tags)
        for item in raw_proposals:
            if not isinstance(item, dict):
                continue
            action = normalize_action(str(item.get("action", "")))
            source_tag = str(item.get("source_tag", "")).strip()
            target_tag = normalize_optional_tag(item.get("target_tag"))
            if not action or not source_tag or source_tag not in existing_tags:
                continue
            if action in {"merge", "rename"} and (not target_tag or source_tag == target_tag):
                continue
            proposals.append(
                TagCleanupProposal(
                    action=action,
                    source_tag=source_tag,
                    target_tag=target_tag,
                    reason=str(item.get("reason", "")),
                    confidence=float_or_default(item.get("confidence"), 0.0),
                    evidence={},
                    source="llm",
                )
            )
        return proposals


def propose_deterministic_cleanup(state: WikiState) -> list[TagCleanupProposal]:
    proposals: list[TagCleanupProposal] = []
    tags = state.tags
    tag_items = sorted(tags.items())

    for tag, record in tag_items:
        if tag_depth(tag) == 1 and not record.wiki_path:
            proposals.append(
                TagCleanupProposal(
                    action="mark_skip",
                    source_tag=tag,
                    reason="Flat generated tag is not suitable as a wiki topic.",
                    confidence=0.95,
                    evidence={"note_count": len(record.source_paths)},
                    source="deterministic",
                )
            )

    for index, (left_tag, left_record) in enumerate(tag_items):
        for right_tag, right_record in tag_items[index + 1 :]:
            if left_tag == right_tag:
                continue

            same_leaf = tag_leaf(left_tag).lower() == tag_leaf(right_tag).lower()
            same_parent = tag_parent(left_tag) == tag_parent(right_tag)
            left_sources = set(left_record.source_paths)
            right_sources = set(right_record.source_paths)
            source_overlap = sorted(left_sources & right_sources)
            subset_relation = bool(left_sources and right_sources) and (
                left_sources <= right_sources or right_sources <= left_sources
            )

            if same_leaf and same_parent and tag_leaf(left_tag) != tag_leaf(right_tag):
                source_tag, target_tag = choose_case_merge_direction(left_tag, left_record, right_tag, right_record)
                proposals.append(
                    TagCleanupProposal(
                        action="merge",
                        source_tag=source_tag,
                        target_tag=target_tag,
                        reason="Tags differ only by case under the same parent; merge into the canonical spelling.",
                        confidence=0.92,
                        evidence={
                            "same_parent": tag_parent(left_tag),
                            "left_sources": sorted(left_sources),
                            "right_sources": sorted(right_sources),
                        },
                        source="deterministic",
                    )
                )
                continue

            if same_leaf and not same_parent:
                source_tag, target_tag = choose_merge_direction(left_tag, left_record, right_tag, right_record)
                proposals.append(
                    TagCleanupProposal(
                        action="review_merge",
                        source_tag=source_tag,
                        target_tag=target_tag,
                        reason="Tags share the same leaf name under different parents; review whether they are duplicate taxonomy entries.",
                        confidence=0.6 if not source_overlap else 0.75,
                        evidence={
                            "same_leaf": tag_leaf(left_tag),
                            "source_overlap": source_overlap,
                            "left_sources": sorted(left_sources),
                            "right_sources": sorted(right_sources),
                        },
                        source="deterministic",
                    )
                )
                continue

            if subset_relation and source_overlap and related_leaf(left_tag, right_tag):
                source_tag, target_tag = choose_merge_direction(left_tag, left_record, right_tag, right_record)
                target_record = tags[target_tag]
                if target_record.wiki_policy != "overview":
                    proposals.append(
                        TagCleanupProposal(
                            action="merge",
                            source_tag=source_tag,
                            target_tag=target_tag,
                            reason="Source paths are a subset and tag names are similar; likely duplicate or redundant tags.",
                            confidence=0.78,
                            evidence={
                                "source_overlap": source_overlap,
                                "source_paths_subset": True,
                            },
                            source="deterministic",
                        )
                    )

    return dedupe_proposals(proposals)


def build_consolidation_user_prompt(
    *,
    state: WikiState,
    vault_root: Path,
    max_notes_per_tag: int,
) -> str:
    lines: list[str] = []
    lines.append("Existing tag tree:")
    for tag in sorted(state.tags):
        record = state.tags[tag]
        evidence_counts = {
            source: len(paths)
            for source, paths in sorted(record.evidence.items())
        }
        lines.append(
            f"- {tag} notes={len(record.source_paths)} evidence={evidence_counts} policy={record.wiki_policy} wiki={record.wiki_path or '(none)'}"
        )

    lines.append("")
    lines.append("Tags with source notes:")
    for tag, record in sorted(state.tags.items()):
        lines.append("")
        lines.append(f"## {tag}")
        lines.append(f"policy: {record.wiki_policy}")
        lines.append(f"candidate_kind: {record.candidate_kind}")
        lines.append(f"evidence: {record.evidence}")
        lines.append(f"source_paths: {', '.join(record.source_paths) or '(none)'}")
        for source_path in record.source_paths[:max_notes_per_tag]:
            note = state.files.get(source_path)
            file_path = vault_root / source_path
            title = note.title if note else file_path.stem
            candidate_tags = note.candidate_tags if note else []
            llm_tags = note.llm_tags if note else []
            user_tags = note.user_tags if note else []
            lines.append(f"- note: {source_path}")
            lines.append(f"  title: {title}")
            lines.append(f"  user_tags: {user_tags}")
            lines.append(f"  llm_tags: {llm_tags}")
            lines.append(f"  candidate_tags: {candidate_tags}")

    lines.append("")
    lines.append("Propose safe cleanup actions. Do not propose changes for tags that should remain distinct.")
    return "\n".join(lines)


def choose_merge_direction(
    left_tag: str,
    left_record: WikiTagRecord,
    right_tag: str,
    right_record: WikiTagRecord,
) -> tuple[str, str]:
    if left_record.wiki_policy == "overview" and right_record.wiki_policy != "overview":
        return right_tag, left_tag
    if right_record.wiki_policy == "overview" and left_record.wiki_policy != "overview":
        return left_tag, right_tag
    if len(left_record.source_paths) < len(right_record.source_paths):
        return left_tag, right_tag
    if len(right_record.source_paths) < len(left_record.source_paths):
        return right_tag, left_tag
    return sorted([left_tag, right_tag])[0], sorted([left_tag, right_tag])[1]


def choose_case_merge_direction(
    left_tag: str,
    left_record: WikiTagRecord,
    right_tag: str,
    right_record: WikiTagRecord,
) -> tuple[str, str]:
    if left_record.wiki_path and not right_record.wiki_path:
        return right_tag, left_tag
    if right_record.wiki_path and not left_record.wiki_path:
        return left_tag, right_tag
    if left_record.wiki_policy_source == "manual" and right_record.wiki_policy_source != "manual":
        return right_tag, left_tag
    if right_record.wiki_policy_source == "manual" and left_record.wiki_policy_source != "manual":
        return left_tag, right_tag
    if len(left_record.source_paths) > len(right_record.source_paths):
        return right_tag, left_tag
    if len(right_record.source_paths) > len(left_record.source_paths):
        return left_tag, right_tag
    if preferred_case_score(tag_leaf(left_tag)) >= preferred_case_score(tag_leaf(right_tag)):
        return right_tag, left_tag
    return left_tag, right_tag


def preferred_case_score(value: str) -> int:
    if any(char.isupper() for char in value) and not value.isupper():
        return 2
    if value.islower():
        return 1
    return 0


def related_leaf(left_tag: str, right_tag: str) -> bool:
    left = tag_leaf(left_tag).lower()
    right = tag_leaf(right_tag).lower()
    return left == right or left in right or right in left


def tag_leaf(tag: str) -> str:
    parts = tag_parts(tag)
    return parts[-1] if parts else tag


def tag_parent(tag: str) -> str:
    parts = tag_parts(tag)
    return "/".join(parts[:-1])


def tag_depth(tag: str) -> int:
    return len(tag_parts(tag))


def tag_parts(tag: str) -> list[str]:
    return [part for part in tag.split("/") if part.strip()]


def normalize_action(action: str) -> str:
    normalized = action.strip()
    allowed = {"merge", "delete", "rename", "mark_overview", "mark_skip", "review_merge"}
    return normalized if normalized in allowed else ""


def normalize_optional_tag(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null", "n/a"}:
        return None
    return normalized


def dedupe_proposals(proposals: list[TagCleanupProposal]) -> list[TagCleanupProposal]:
    seen: set[tuple[str, str, str | None]] = set()
    result: list[TagCleanupProposal] = []
    for proposal in proposals:
        key = (proposal.action, proposal.source_tag, proposal.target_tag)
        if key in seen:
            continue
        seen.add(key)
        result.append(proposal)
    return result


def tag_cleanup_proposal_to_dict(proposal: TagCleanupProposal) -> dict[str, Any]:
    return asdict(proposal)


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
