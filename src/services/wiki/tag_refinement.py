from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.wiki.schema import WikiState


TAG_REFINEMENT_SYSTEM_PROMPT = """You are a tag taxonomy refinement assistant for a personal knowledge base.

You are given one existing tag and the notes currently assigned to it.
Your task is to decide whether this tag is too broad for wiki synthesis.

If the tag is too broad, propose more focused child or sibling tags and assign each source note to the proposed tags.
If the tag is already coherent, keep it.

Important:
- This is a proposal only. Do not rewrite source notes.
- Do not split a tag merely because notes have minor differences.
- Prefer stable, navigable tags that could become useful wiki pages.
- Keep a broad parent tag if it is useful for navigation, but mark it as unsuitable for direct wiki synthesis when appropriate.

Return only JSON:
{
  "tag": "web/backend",
  "decision": "split" | "keep",
  "problem": "too_broad" | "mixed_topics" | "coherent",
  "keep_parent_tag": true,
  "parent_wiki_policy": "skip" | "overview" | "generate",
  "suggested_tags": [
    {
      "tag": "java/servlet",
      "source_paths": ["path/to/source.md"],
      "reason": "short reason"
    }
  ],
  "reason": "short overall reason"
}
"""


@dataclass(frozen=True)
class SuggestedTagAssignment:
    tag: str
    source_paths: list[str]
    reason: str


@dataclass(frozen=True)
class TagRefinementProposal:
    tag: str
    decision: str
    problem: str
    keep_parent_tag: bool
    parent_wiki_policy: str
    suggested_tags: list[SuggestedTagAssignment]
    reason: str
    raw_response: str


class LLMTagRefiner:
    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.0,
        max_chars_per_note: int = 1800,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_chars_per_note = max_chars_per_note

    def refine_tag(
        self,
        *,
        state: WikiState,
        vault_root: Path,
        tag: str,
    ) -> TagRefinementProposal:
        tag_record = state.tags.get(tag)
        if tag_record is None:
            raise ValueError(f"Tag not found in wiki state: {tag}")

        user_prompt = build_refinement_user_prompt(
            state=state,
            vault_root=vault_root,
            tag=tag,
            source_paths=tag_record.source_paths,
            max_chars_per_note=self.max_chars_per_note,
        )
        response = self.client.complete(
            LLMRequest(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    LLMMessage(role="system", content=TAG_REFINEMENT_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=user_prompt),
                ],
            )
        )
        payload = parse_json_object(response.content)
        return proposal_from_payload(tag, payload, response.content)


def build_refinement_user_prompt(
    *,
    state: WikiState,
    vault_root: Path,
    tag: str,
    source_paths: list[str],
    max_chars_per_note: int,
) -> str:
    existing_tags = sorted(state.tags)
    parts: list[str] = []
    parts.append(f"Current tag: {tag}")
    parts.append("")
    parts.append("Existing tag tree:")
    for existing_tag in existing_tags[:300]:
        parts.append(f"- {existing_tag}")
    parts.append("")
    parts.append("Notes under current tag:")

    for index, source_path in enumerate(source_paths, start=1):
        record = state.files.get(source_path)
        file_path = vault_root / source_path
        text = ""
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        excerpt = text[:max_chars_per_note]
        title = record.title if record else file_path.stem
        tags = sorted(set((record.user_tags if record else []) + (record.llm_tags if record else [])))
        parts.append("")
        parts.append(f"[N{index}] {source_path}")
        parts.append(f"Title: {title}")
        parts.append(f"Current note tags: {', '.join(tags) or '(none)'}")
        parts.append("---")
        parts.append(excerpt)
        parts.append("---")

    parts.append("")
    parts.append("Decide whether the current tag should be kept as a direct wiki topic or split/refined.")
    return "\n".join(parts)


def proposal_from_payload(tag: str, payload: dict[str, Any], raw_response: str) -> TagRefinementProposal:
    raw_suggested = payload.get("suggested_tags", [])
    if not isinstance(raw_suggested, list):
        raw_suggested = []

    suggested: list[SuggestedTagAssignment] = []
    for item in raw_suggested:
        if not isinstance(item, dict):
            continue
        suggested_tag = str(item.get("tag", "")).strip()
        source_paths = item.get("source_paths", [])
        if not isinstance(source_paths, list):
            source_paths = []
        if not suggested_tag:
            continue
        suggested.append(
            SuggestedTagAssignment(
                tag=suggested_tag,
                source_paths=[str(path) for path in source_paths],
                reason=str(item.get("reason", "")),
            )
        )

    return TagRefinementProposal(
        tag=str(payload.get("tag", tag)),
        decision=normalize_choice(str(payload.get("decision", "keep")), {"split", "keep"}, "keep"),
        problem=normalize_choice(
            str(payload.get("problem", "coherent")),
            {"too_broad", "mixed_topics", "coherent"},
            "coherent",
        ),
        keep_parent_tag=bool(payload.get("keep_parent_tag", True)),
        parent_wiki_policy=normalize_choice(
            str(payload.get("parent_wiki_policy", "generate")),
            {"skip", "overview", "generate"},
            "generate",
        ),
        suggested_tags=suggested,
        reason=str(payload.get("reason", "")),
        raw_response=raw_response,
    )


def normalize_choice(value: str, allowed: set[str], default: str) -> str:
    normalized = value.strip()
    return normalized if normalized in allowed else default


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


def tag_refinement_proposal_to_dict(proposal: TagRefinementProposal) -> dict[str, Any]:
    return asdict(proposal)
