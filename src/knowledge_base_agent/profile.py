from __future__ import annotations
import re

from dataclasses import asdict, dataclass, field
from typing import Any

from knowledge_base_agent.parser import ParsedNote

@dataclass(frozen=True)
class NoteProfile:
    path: str
    title: str
    note_type: str
    tags: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    links_out: list[str] = field(default_factory=list)
    external_links: list[str] = field(default_factory=list)
    has_frontmatter: bool = False
    code_block_count: int = 0
    char_count: int = 0
    word_count: int = 0
    size_bytes: int = 0
    modified_time: float = 0.0
    excerpt: str = ""
    semantic_excerpt: str = ""

def build_note_profile(parsed_note: ParsedNote) -> NoteProfile:
    body = parsed_note.body

    return NoteProfile(
        path=parsed_note.relative_path,
        title=parsed_note.title,
        note_type=classify_note(parsed_note),
        tags=parsed_note.tags,
        headings=[heading.text for heading in parsed_note.headings],
        links_out=sorted({link.target for link in parsed_note.wikilinks}),
        external_links=parsed_note.external_links,
        has_frontmatter=parsed_note.frontmatter_raw is not None,
        code_block_count=len(parsed_note.code_blocks),
        char_count=len(body),
        word_count=count_words_mixed(body),
        size_bytes=parsed_note.size_bytes,
        modified_time=parsed_note.modified_time,
        excerpt=parsed_note.excerpt,
        semantic_excerpt=build_semantic_excerpt(parsed_note.body),
    )

def profile_to_dict(profile: NoteProfile) -> dict[str, Any]:
    return asdict(profile)



def classify_note(parsed_note: ParsedNote) -> str:
    path = parsed_note.relative_path.replace("\\", "/").lower()
    tags = {tag.lower() for tag in parsed_note.tags}
    title = parsed_note.title.lower()

    if has_any(path, ["daily/", "日记/", "journal/"]) or has_any(tags, ["daily", "journal"]):
        return "daily"

    if has_any(path, ["troubleshooting/", "debug/", "问题/", "问题记录/"]):
        return "troubleshooting"

    if has_any(tags, ["troubleshooting", "debug", "bugfix"]):
        return "troubleshooting"

    if has_any(path, ["project/", "projects/", "项目/"]) or has_any(tags, ["project"]):
        return "project"

    if has_any(path, ["reference/", "references/", "摘录/", "资料/"]):
        return "reference"

    if has_any(tags, ["reference", "摘录"]):
        return "reference"

    if has_any(path, ["inbox/", "temporary/", "temp/", "临时/"]):
        return "temporary"

    if has_any(tags, ["temporary", "temp", "draft"]):
        return "temporary"

    if has_any(path, ["moc/", "index/", "索引/"]):
        return "index"

    if has_any(tags, ["moc", "index"]):
        return "index"

    if "moc" in title or "index" in title or "索引" in title:
        return "index"

    if len(parsed_note.code_blocks) >= 3:
        return "code"

    if has_any(tags, ["summary", "总结"]):
        return "summary"

    if has_any(tags, ["concept", "概念"]):
        return "concept"

    return "unknown"

def has_any(value: str | set[str], needles: list[str]) -> bool:
    if isinstance(value, set):
        return any(needle in value for needle in needles)

    return any(needle in value for needle in needles)

def count_words_mixed(text: str) -> int:
    english_tokens = re.findall(r"[A-Za-z0-9_./+-]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)

    return len(english_tokens) + len(chinese_chars)


HEADING_SPLIT_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```[\w+-]*\n.*?```", re.DOTALL)


def build_semantic_excerpt(
    body: str,
    max_total_chars: int = 2400,
    max_section_chars: int = 400,
    min_section_chars: int = 160,
) -> str:
    """Build a section-balanced excerpt for semantic matching.

    The regular excerpt keeps the beginning of the note for human preview.
    This semantic excerpt samples from multiple heading sections so long notes
    are not represented only by their first chapter.
    """
    text = CODE_BLOCK_RE.sub("", body)
    sections = split_markdown_sections(text)

    if not sections:
        return normalize_excerpt_text(text)[:max_total_chars]

    per_section_chars = calculate_section_char_budget(
        section_count=len(sections),
        max_total_chars=max_total_chars,
        min_section_chars=min_section_chars,
        max_section_chars=max_section_chars,
    )

    parts: list[str] = []
    used_chars = 0

    for heading, section_body in sections:
        normalized_body = normalize_excerpt_text(section_body)
        if not normalized_body and not heading:
            continue

        section_text = ""
        if heading:
            section_text += heading.strip()
        if normalized_body:
            if section_text:
                section_text += " "
            section_text += normalized_body[:per_section_chars]

        if not section_text:
            continue

        remaining = max_total_chars - used_chars
        if remaining <= 0:
            break

        section_text = section_text[:remaining]
        parts.append(section_text)
        used_chars += len(section_text)

        if used_chars >= max_total_chars:
            break

    return "\n\n".join(parts)


def calculate_section_char_budget(
    section_count: int,
    max_total_chars: int,
    min_section_chars: int,
    max_section_chars: int,
) -> int:
    if section_count <= 0:
        return max_section_chars

    average_budget = max_total_chars // section_count
    return max(min_section_chars, min(max_section_chars, average_budget))


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING_SPLIT_RE.finditer(text))

    if not matches:
        stripped = text.strip()
        return [("", stripped)] if stripped else []

    sections: list[tuple[str, str]] = []

    preface = text[: matches[0].start()].strip()
    if preface:
        sections.append(("", preface))

    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_body = text[start:end].strip()
        sections.append((heading, section_body))

    return sections


def normalize_excerpt_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
