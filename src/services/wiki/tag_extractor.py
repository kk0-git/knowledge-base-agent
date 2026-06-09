from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.wiki.schema import TagExtractionResult


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
OBSIDIAN_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_\-\u4e00-\u9fff]+(?:/[A-Za-z0-9_\-\u4e00-\u9fff]+)*)")


TAG_SYSTEM_PROMPT = """You are a knowledge management assistant that categorizes one Markdown note with navigation tags.

Tags are used to browse, filter, and synthesize wiki articles from related notes.
Only add a tag if the note strongly belongs under that category.

Use the existing tag tree as the primary vocabulary.
Tags must use the current path format: existing_root/specific_tag, for example java/servlet.
The first path segment must be one of the existing top-level roots shown below.
Do not invent a new top-level category.
Do not return flat tags such as javaweb or jsp.
If none of the existing top-level categories naturally fit this note, return an empty tag list.
New tags should be level-2 or deeper under an existing category.
Prefer broad, stable tags over overly specific one-off tags.
Avoid tags that are only keywords, commands, temporary states, or formatting labels.

Return only a JSON object like:
{
  "tags": ["web框架/fastapi", "python/异步"],
  "confidence": 0.82,
  "reason": "short reason"
}
"""


@dataclass(frozen=True)
class NoteTagInput:
    note_path: str
    title: str
    headings: list[str]
    user_tags: list[str]
    excerpt: str


class LLMTagExtractor:
    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.0,
        max_tags: int = 5,
        allow_new_roots: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tags = max_tags
        self.allow_new_roots = allow_new_roots

    def extract_tags(
        self,
        note: NoteTagInput,
        existing_tag_tree: list[str],
    ) -> TagExtractionResult:
        roots = sorted({tag.split("/", 1)[0] for tag in existing_tag_tree if tag})
        user_prompt = build_tag_user_prompt(
            note=note,
            existing_tag_tree=existing_tag_tree,
            allowed_roots=roots,
            max_tags=self.max_tags,
        )

        response = self.client.complete(
            LLMRequest(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    LLMMessage(role="system", content=TAG_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=user_prompt),
                ],
            )
        )

        payload = parse_json_object(response.content)
        raw_tags = payload.get("tags", [])
        if not isinstance(raw_tags, list):
            raw_tags = []

        normalized, candidate_tags = normalize_tags(
            raw_tags,
            allowed_roots=roots,
            allow_new_roots=self.allow_new_roots,
            max_tags=self.max_tags,
        )
        return TagExtractionResult(
            tags=normalized,
            candidate_tags=candidate_tags,
            confidence=float_or_default(payload.get("confidence"), 0.0),
            reason=str(payload.get("reason", "")),
            raw_response=response.content,
        )


def build_note_tag_input(vault_root: Path, file_path: Path, max_excerpt_chars: int = 3000) -> NoteTagInput:
    markdown = file_path.read_text(encoding="utf-8", errors="replace")
    note_path = file_path.relative_to(vault_root).as_posix()
    headings = extract_headings(markdown, limit=30)
    user_tags = extract_user_tags(markdown)
    return NoteTagInput(
        note_path=note_path,
        title=file_path.stem,
        headings=headings,
        user_tags=user_tags,
        excerpt=sample_note_excerpt(markdown, max_chars=max_excerpt_chars),
    )


def build_tag_user_prompt(
    *,
    note: NoteTagInput,
    existing_tag_tree: list[str],
    allowed_roots: list[str],
    max_tags: int,
) -> str:
    tag_tree_text = "\n".join(f"- {tag}" for tag in existing_tag_tree[:300]) or "(empty)"
    roots_text = ", ".join(allowed_roots) if allowed_roots else "(none)"
    headings_text = "\n".join(f"- {heading}" for heading in note.headings) or "(none)"
    user_tags_text = ", ".join(note.user_tags) or "(none)"
    return f"""Existing tag tree:
{tag_tree_text}

Allowed top-level roots:
{roots_text}

Max tags: {max_tags}

Note path: {note.note_path}
Title: {note.title}
Existing user tags: {user_tags_text}
Headings:
{headings_text}

Markdown excerpt:
---
{note.excerpt}
---

Choose navigation tags for this note."""


def extract_headings(markdown: str, limit: int = 50) -> list[str]:
    headings: list[str] = []
    for match in HEADING_RE.finditer(markdown):
        heading = match.group(2).strip()
        if heading:
            headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def extract_user_tags(markdown: str) -> list[str]:
    tags = set()
    for match in OBSIDIAN_TAG_RE.finditer(markdown):
        tag = normalize_tag(match.group(1))
        if is_wiki_user_tag(tag):
            tags.add(tag)
    return sorted(tag for tag in tags if tag)


def sample_note_excerpt(markdown: str, max_chars: int) -> str:
    stripped = markdown.strip()
    if len(stripped) <= max_chars:
        return stripped

    lines = stripped.splitlines()
    selected: list[str] = []
    total = 0
    for line in lines:
        if HEADING_RE.match(line) or line.strip():
            selected.append(line)
            total += len(line) + 1
        if total >= max_chars:
            break
    return "\n".join(selected)[:max_chars]


def normalize_tags(
    tags: list[Any],
    *,
    allowed_roots: list[str],
    allow_new_roots: bool,
    max_tags: int,
) -> tuple[list[str], list[str]]:
    result: list[str] = []
    candidate_tags: list[str] = []
    allowed_root_set = set(allowed_roots)

    for raw_tag in tags:
        tag = normalize_tag(str(raw_tag))
        if not tag:
            continue

        root = tag.split("/", 1)[0]
        if allowed_root_set and not allow_new_roots and root not in allowed_root_set:
            if tag not in candidate_tags:
                candidate_tags.append(tag)
            continue
        if tag_depth(tag) < 2:
            if tag not in candidate_tags:
                candidate_tags.append(tag)
            continue

        if tag not in result:
            result.append(tag)

        if len(result) >= max_tags:
            continue

    return result, candidate_tags


def normalize_tag(tag: str) -> str:
    normalized = tag.strip().strip("#").strip()
    if normalized.startswith("tags/"):
        normalized = normalized.removeprefix("tags/")
    normalized = re.sub(r"/+", "/", normalized)
    normalized = normalized.strip("/")
    return normalized


def is_wiki_user_tag(tag: str) -> bool:
    return tag_depth(tag) >= 2


def tag_depth(tag: str) -> int:
    return len([part for part in tag.split("/") if part.strip()])


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}

    return payload if isinstance(payload, dict) else {}


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
