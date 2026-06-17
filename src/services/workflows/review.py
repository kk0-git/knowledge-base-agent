from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.workflows.schema import ScopeResult


ReviewMode = Literal["topic", "notes"]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_/\-\u4e00-\u9fff]+)")
WIKILINK_RE = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]*)?\]\]")


BASE_REVIEW_PROMPT = """You are a knowledge-base organization advisor for a personal Obsidian vault.

Your role is closer to an editor, curator, and study coach than an executor. You read a bounded set of source notes and produce review suggestions for a human to consider.

Audience: Chinese user. Write all user-facing prose in Simplified Chinese. Keep enum values and source paths exactly in the requested schema.

Role boundaries:
- Evidence comes from the ReviewPacket: source paths, headings, tags, wikilinks, relationship hints, and excerpts.
- Path fidelity is part of your role. Refer only to source paths that appear in the packet. If a relationship is plausible but uncertain, label it as review_needed.
- Preservation-first editing is part of your role. Favor add_links, add_tags, add_summary, merge_review_needed, split_review_needed, and review_needed over destructive advice.
- Good suggestions are concrete: name the note, explain why, and describe the next human action.
- A useful review distinguishes core notes, edge notes, missing summaries, missing links, and candidate review questions.
- The output contract is one JSON object matching this shape. Do not wrap it in Markdown.

{
  "summary": "一两句话概括这组笔记的整理状态",
  "topic_structure": {
    "coverage": "主题或笔记组覆盖了什么",
    "core_notes": ["relative/path.md"],
    "edge_notes": ["relative/path.md"],
    "missing_parts": ["缺口"]
  },
  "note_reviews": [
    {
      "path": "relative/path.md",
      "role": "concept|summary|troubleshooting|reference|temporary|index|unknown",
      "recommended_action": "keep|add_links|add_tags|add_summary|merge_review_needed|split_review_needed|review_needed",
      "reason": "原因",
      "suggested_tags": ["tag/path"],
      "suggested_links": ["relative/path.md"],
      "risk": "low|medium|high"
    }
  ],
  "relationship_suggestions": [
    {
      "source": "relative/path.md",
      "target": "relative/path.md",
      "relationship": "duplicate|overlap|complementary|prerequisite|related|unknown",
      "recommended_action": "add_link|merge_review_needed|keep_separate|create_topic_wiki|review_needed",
      "reason": "原因"
    }
  ],
  "topic_suggestions": [
    {
      "topic": "主题名",
      "notes": ["relative/path.md"],
      "suggested_output": "wiki|review_questions|summary_note|none",
      "reason": "原因"
    }
  ],
  "review_questions": [
    {
      "question": "复习或追问问题",
      "source_notes": ["relative/path.md"],
      "reason": "为什么这个问题有价值"
    }
  ],
  "next_actions": ["下一步动作"]
}
"""


TOPIC_REVIEW_PROMPT = """Current mode: Topic Review.

Role: knowledge topic curator.

A topic curator looks at the shape of a knowledge topic rather than grading every note line by line. Your review should help the user decide whether this topic is ready for a wiki, needs more links, has duplicated material, or lacks a summary.

Focus on:
- What this topic already covers.
- Which notes are core evidence and which notes are peripheral.
- Which notes should link to each other.
- Whether the topic has duplication, overlap, fragmentation, or missing synthesis.
- Whether a topic wiki should be generated or refreshed.
- How the topic wiki should be organized.
- Which review questions would help the user consolidate this topic.

Keep note-level comments selective. Include only the notes where a concrete action is useful.
"""


NOTES_REVIEW_PROMPT = """Current mode: Notes Review.

Role: note editor and local knowledge-base maintainer.

A note editor looks at concrete notes and local relationships. Your review should help the user improve the selected notes without turning the output into a broad topic wiki plan.

Focus on:
- Whether each note has a clear purpose.
- Whether headings and structure are useful.
- Whether the note needs a short summary, tags, or wikilinks.
- Whether selected notes duplicate, overlap, complement, or should reference each other.
- Whether a note should be split or merged only after human review.
- Which review questions can be generated from these notes.

Keep topic-wiki suggestions secondary unless the selected notes clearly form a coherent topic.
"""


@dataclass(frozen=True)
class ReviewNote:
    path: str
    title: str
    chars: int
    headings: tuple[str, ...]
    tags: tuple[str, ...]
    links: tuple[str, ...]
    signals: tuple[str, ...]
    excerpt: str


@dataclass(frozen=True)
class ReviewEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class ReviewPacket:
    scope: dict[str, Any]
    review_mode: ReviewMode
    notes: tuple[ReviewNote, ...]
    internal_edges: tuple[ReviewEdge, ...] = ()
    tag_groups: dict[str, list[str]] = field(default_factory=dict)
    peer_hints: tuple[dict[str, Any], ...] = ()


def run_organize_suggestions(
    *,
    vault_root: Path,
    scope_result: ScopeResult,
    llm_client: LLMClient,
    llm_model: str,
    temperature: float = 0.2,
    max_notes: int = 12,
    max_chars_per_note: int = 1800,
    review_mode: str = "auto",
) -> dict[str, Any]:
    resolved_mode = resolve_review_mode(scope_result=scope_result, review_mode=review_mode)
    packet = build_review_packet(
        vault_root=vault_root,
        scope_result=scope_result,
        max_notes=max_notes,
        max_chars_per_note=max_chars_per_note,
        review_mode=resolved_mode,
    )
    user_prompt = "\n".join(
        [
            "请以当前角色审阅这组笔记，并输出整理建议。",
            "",
            "ReviewPacket JSON:",
            json.dumps(review_packet_to_dict(packet), ensure_ascii=False, indent=2),
        ]
    )
    response = llm_client.complete(
        LLMRequest(
            model=llm_model,
            messages=[
                LLMMessage(role="system", content=build_system_prompt(resolved_mode)),
                LLMMessage(role="user", content=user_prompt),
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    )
    parsed = normalize_and_validate_suggestions(
        suggestions=parse_json_object(response.content),
        packet=packet,
        vault_root=vault_root,
    )
    return {
        "review_mode": resolved_mode,
        "packet": review_packet_to_dict(packet),
        "suggestions": parsed,
        "validation": parsed.get("_validation", {}),
        "raw_response": response.content,
    }


def resolve_review_mode(*, scope_result: ScopeResult, review_mode: str) -> ReviewMode:
    if review_mode in {"topic", "notes"}:
        return review_mode  # type: ignore[return-value]
    if review_mode and review_mode != "auto":
        raise ValueError(f"Unsupported review_mode: {review_mode}")
    return "topic" if scope_result.scope.type == "tag" else "notes"


def build_system_prompt(review_mode: ReviewMode) -> str:
    mode_prompt = TOPIC_REVIEW_PROMPT if review_mode == "topic" else NOTES_REVIEW_PROMPT
    return BASE_REVIEW_PROMPT + "\n\n" + mode_prompt


def build_review_packet(
    *,
    vault_root: Path,
    scope_result: ScopeResult,
    max_notes: int,
    max_chars_per_note: int,
    review_mode: ReviewMode,
) -> ReviewPacket:
    source_notes = list(scope_result.notes[:max_notes])
    notes: list[ReviewNote] = []

    for note in source_notes:
        path = str(note.get("path", ""))
        if not path:
            continue
        full_path = vault_root / path
        if not full_path.exists():
            continue
        text = full_path.read_text(encoding="utf-8", errors="replace")
        headings = tuple(match.group(2).strip() for match in HEADING_RE.finditer(text))
        tags = tuple(sorted({match.group(1).strip("/") for match in TAG_RE.finditer(text)}))
        links = tuple(sorted({normalize_link_target(match.group(1)) for match in WIKILINK_RE.finditer(text)}))
        signals = infer_note_signals(text=text, headings=headings, tags=tags, links=links)
        notes.append(
            ReviewNote(
                path=path,
                title=str(note.get("title") or full_path.stem),
                chars=len(text.strip()),
                headings=headings[:30],
                tags=tags[:30],
                links=links[:50],
                signals=signals,
                excerpt=sample_excerpt(text, max_chars=max_chars_per_note),
            )
        )

    path_index = build_path_index(notes)
    internal_edges: list[ReviewEdge] = []
    for note in notes:
        for link in note.links:
            resolved = resolve_link(link, path_index)
            if resolved and resolved != note.path:
                internal_edges.append(ReviewEdge(source=note.path, target=resolved, relation="wikilink"))

    return ReviewPacket(
        scope={
            "type": scope_result.scope.type,
            "value": scope_result.scope.value,
            "note_count": len(notes),
            "metadata": scope_result.metadata,
        },
        review_mode=review_mode,
        notes=tuple(notes),
        internal_edges=tuple(internal_edges),
        tag_groups=build_tag_groups(notes),
        peer_hints=build_peer_hints(notes, internal_edges),
    )


def infer_note_signals(
    *,
    text: str,
    headings: tuple[str, ...],
    tags: tuple[str, ...],
    links: tuple[str, ...],
) -> tuple[str, ...]:
    stripped = text.strip()
    signals: list[str] = []
    if not stripped:
        signals.append("empty_note")
    elif len(stripped) < 200:
        signals.append("very_short")
    if not headings:
        signals.append("no_headings")
    elif len(headings) > 25:
        signals.append("many_headings")
    if not tags:
        signals.append("missing_tags")
    if not links:
        signals.append("no_wikilinks")
    lowered = stripped.lower()
    if any(keyword in lowered for keyword in ("error", "exception", "traceback", "报错", "失败")):
        signals.append("troubleshooting")
    if any(keyword in stripped for keyword in ("TODO", "待整理", "临时", "草稿")):
        signals.append("temporary_or_draft")
    return tuple(signals)


def normalize_link_target(raw: str) -> str:
    return raw.strip().replace("\\", "/").split("#", 1)[0].split("^", 1)[0].strip()


def build_path_index(notes: list[ReviewNote]) -> dict[str, str]:
    index: dict[str, str] = {}
    for note in notes:
        path = Path(note.path)
        index[note.path] = note.path
        index[note.path.removesuffix(".md")] = note.path
        index[path.name] = note.path
        index[path.stem] = note.path
    return index


def resolve_link(link: str, path_index: dict[str, str]) -> str | None:
    cleaned = link.strip().replace("\\", "/").strip("/")
    candidates = [
        cleaned,
        cleaned.removesuffix(".md"),
        f"{cleaned}.md",
        Path(cleaned).name,
        Path(cleaned).stem,
    ]
    for candidate in candidates:
        if candidate in path_index:
            return path_index[candidate]
    return None


def build_tag_groups(notes: list[ReviewNote]) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = {}
    for note in notes:
        for tag in note.tags:
            parts = tag.split("/")
            group = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
            groups.setdefault(group, set()).add(note.path)
    return {tag: sorted(paths) for tag, paths in sorted(groups.items()) if len(paths) >= 2}


def build_peer_hints(notes: list[ReviewNote], edges: list[ReviewEdge]) -> tuple[dict[str, Any], ...]:
    edge_pairs = {(edge.source, edge.target) for edge in edges}
    hints: list[dict[str, Any]] = []
    for index, left in enumerate(notes):
        for right in notes[index + 1 :]:
            shared_tags = sorted(set(left.tags) & set(right.tags))
            shared_headings = sorted(set(left.headings) & set(right.headings))
            linked = (left.path, right.path) in edge_pairs or (right.path, left.path) in edge_pairs
            if shared_tags or shared_headings or linked:
                hints.append(
                    {
                        "left": left.path,
                        "right": right.path,
                        "shared_tags": shared_tags[:8],
                        "shared_headings": shared_headings[:8],
                        "linked": linked,
                    }
                )
    return tuple(hints[:30])


def sample_excerpt(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "\n...[truncated]"


def review_packet_to_dict(packet: ReviewPacket) -> dict[str, Any]:
    return {
        "scope": packet.scope,
        "review_mode": packet.review_mode,
        "notes": [asdict(note) for note in packet.notes],
        "internal_edges": [asdict(edge) for edge in packet.internal_edges],
        "tag_groups": packet.tag_groups,
        "peer_hints": list(packet.peer_hints),
    }


def normalize_and_validate_suggestions(
    *,
    suggestions: dict[str, Any],
    packet: ReviewPacket,
    vault_root: Path,
) -> dict[str, Any]:
    suggestions = normalize_free_text(suggestions)
    note_paths = {note.path for note in packet.notes}
    path_index = build_suggestion_path_index(packet.notes)
    warnings: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []

    def normalize_path(value: Any, field: str) -> Any:
        if not isinstance(value, str) or not value.strip():
            return value
        raw = value.strip().replace("\\", "/")
        fixed = resolve_suggested_path(raw, vault_root=vault_root, note_paths=note_paths, path_index=path_index)
        if fixed and fixed != raw:
            corrections.append({"field": field, "from": raw, "to": fixed})
            return fixed
        if fixed:
            return raw
        warnings.append({"type": "invalid_suggested_path", "field": field, "path": raw})
        return raw

    topic_structure = suggestions.get("topic_structure")
    if isinstance(topic_structure, dict):
        for key in ("core_notes", "edge_notes"):
            if isinstance(topic_structure.get(key), list):
                topic_structure[key] = [
                    normalize_path(path, f"topic_structure.{key}") for path in topic_structure[key]
                ]

    for index, item in enumerate(as_list(suggestions.get("note_reviews"))):
        if not isinstance(item, dict):
            continue
        item["path"] = normalize_path(item.get("path"), f"note_reviews[{index}].path")
        if isinstance(item.get("suggested_links"), list):
            item["suggested_links"] = [
                normalize_path(path, f"note_reviews[{index}].suggested_links") for path in item["suggested_links"]
            ]
        if isinstance(item.get("suggested_tags"), list):
            item["suggested_tags"] = [normalize_tag(tag) for tag in item["suggested_tags"] if normalize_tag(tag)]
        action = str(item.get("recommended_action", ""))
        reason = str(item.get("reason", ""))
        if "delete" in action.lower() or "删除" in reason:
            warnings.append(
                {
                    "type": "delete_suggestion_downgraded",
                    "field": f"note_reviews[{index}]",
                    "path": item.get("path"),
                }
            )
            item["recommended_action"] = "review_needed"
            item["reason"] = reason.replace("删除", "归档复查")

    for index, item in enumerate(as_list(suggestions.get("relationship_suggestions"))):
        if not isinstance(item, dict):
            continue
        item["source"] = normalize_path(item.get("source"), f"relationship_suggestions[{index}].source")
        item["target"] = normalize_path(item.get("target"), f"relationship_suggestions[{index}].target")

    for index, item in enumerate(as_list(suggestions.get("topic_suggestions"))):
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("notes"), list):
            item["notes"] = [
                normalize_path(path, f"topic_suggestions[{index}].notes") for path in item["notes"]
            ]

    for index, item in enumerate(as_list(suggestions.get("review_questions"))):
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("source_notes"), list):
            item["source_notes"] = [
                normalize_path(path, f"review_questions[{index}].source_notes") for path in item["source_notes"]
            ]

    suggestions["_validation"] = {
        "warnings": warnings,
        "corrections": corrections,
        "warning_count": len(warnings),
        "correction_count": len(corrections),
    }
    return suggestions


def normalize_free_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("tags/", "").replace("Tags/", "")
    if isinstance(value, list):
        return [normalize_free_text(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_free_text(item) for key, item in value.items()}
    return value


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_suggestion_path_index(notes: tuple[ReviewNote, ...]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for note in notes:
        path = Path(note.path)
        keys = {
            note.path,
            note.path.removesuffix(".md"),
            path.name,
            path.stem,
        }
        for key in keys:
            index.setdefault(key, []).append(note.path)
    return index


def resolve_suggested_path(
    raw: str,
    *,
    vault_root: Path,
    note_paths: set[str],
    path_index: dict[str, list[str]],
) -> str | None:
    cleaned = raw.strip().strip("`").replace("\\", "/").strip("/")
    if not cleaned:
        return None
    if cleaned in note_paths and (vault_root / cleaned).exists():
        return cleaned
    if (vault_root / cleaned).exists():
        return cleaned
    candidates = [
        cleaned,
        cleaned.removesuffix(".md"),
        f"{cleaned}.md",
        Path(cleaned).name,
        Path(cleaned).stem,
    ]
    matches: set[str] = set()
    for candidate in candidates:
        matches.update(path_index.get(candidate, []))
    existing_matches = {path for path in matches if (vault_root / path).exists()}
    if len(existing_matches) == 1:
        return next(iter(existing_matches))
    return None


def normalize_tag(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    tag = value.strip().strip("#").strip("/")
    while tag.lower().startswith("tags/"):
        tag = tag[5:].strip("/")
    return tag


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM review response must be a JSON object")
    return payload
