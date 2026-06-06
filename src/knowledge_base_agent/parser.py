from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from knowledge_base_agent.scanner import ScannedNote

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
WIKILINK_RE = re.compile(r"(?<!\!)\[\[([^\]]+)\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w\u4e00-\u9fff/-]+)")
EXTERNAL_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
CODE_BLOCK_RE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)

@dataclass(frozen=True)
class Heading:
    level: int
    text: str

@dataclass(frozen=True)
class WikiLink:
    raw: str
    target: str
    alias: str | None = None
    heading: str | None = None
    block_id: str | None = None

@dataclass(frozen=True)
class CodeBlock:
    language: str | None
    content: str


@dataclass(frozen=True)
class ParsedNote:
    path: Path
    relative_path: str
    title: str
    content: str
    body: str
    frontmatter_raw: str | None
    headings: list[Heading] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    wikilinks: list[WikiLink] = field(default_factory=list)
    external_links: list[str] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    excerpt: str = ""
    size_bytes: int = 0
    modified_time: float = 0.0

def parse_note(scanned_note: ScannedNote) -> ParsedNote:
    """从扫描结果中解析笔记内容，提取结构化信息。"""
    content = scanned_note.path.read_text(encoding="utf-8", errors="replace")
    frontmatter_raw, body = split_frontmatter(content)

    return ParsedNote(
        path=scanned_note.path,
        relative_path=scanned_note.relative_path,
        title=extract_title(scanned_note.path, body),
        content=content,
        body=body,
        frontmatter_raw=frontmatter_raw,
        headings=extract_headings(body),
        tags=extract_tags(body, frontmatter_raw),
        wikilinks=extract_wikilinks(body),
        external_links=extract_external_links(body),
        code_blocks=extract_code_blocks(body),
        excerpt=build_excerpt(body),
        size_bytes=scanned_note.size_bytes,
        modified_time=scanned_note.modified_time,
    )

def split_frontmatter(content: str) -> tuple[str | None, str]:
    match = FRONTMATTER_RE.match(content)
    if not match:
        return None, content

    # 取出 frontmatter 内容并去除两端的 --- 和空白
    frontmatter_raw = match.group(1).strip()
    body = content[match.end():]
    return frontmatter_raw, body

def extract_title(path: Path, body: str) -> str:
    headings = extract_headings(body)
    if headings and headings[0].level == 1:
        return headings[0].text

    return path.stem

def extract_headings(body: str) -> list[Heading]:
    headings: list[Heading] = []

    for match in HEADING_RE.finditer(body):
        level = len(match.group(1))
        text = match.group(2).strip()
        headings.append(Heading(level=level, text=text))

    return headings

def extract_tags(body: str, frontmatter_raw: str | None) -> list[str]:
    """从正文和 frontmatter 中提取标签，返回唯一且排序的标签列表"""
    tags: set[str] = set()

    for match in TAG_RE.finditer(body):
        tags.add(match.group(1).strip("/"))

    if frontmatter_raw:
        tags.update(extract_frontmatter_tags(frontmatter_raw))

    return sorted(tag for tag in tags if tag)

def extract_frontmatter_tags(frontmatter_raw: str) -> set[str]:
    tags: set[str] = set()
    lines = frontmatter_raw.splitlines()

    in_tags_list = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("tags:"):
            value = stripped.removeprefix("tags:").strip()
            in_tags_list = not value

            if value.startswith("[") and value.endswith("]"):
                items = value.strip("[]").split(",")
                tags.update(clean_tag(item) for item in items)
            elif value:
                tags.add(clean_tag(value))

            continue

        if in_tags_list:
            if stripped.startswith("- "):
                tags.add(clean_tag(stripped.removeprefix("- ")))
            elif stripped and not line.startswith((" ", "\t")):
                in_tags_list = False

    return {tag for tag in tags if tag}

def clean_tag(value: str) -> str:
    return value.strip().strip('"').strip("'").removeprefix("#").strip("/")

def extract_wikilinks(body: str) -> list[WikiLink]:
    links: list[WikiLink] = []

    for match in WIKILINK_RE.finditer(body):
        raw = match.group(1).strip()
        target_part, alias = split_alias(raw)
        target, heading, block_id = split_target(target_part)

        if target:
            links.append(
                WikiLink(
                    raw=raw,
                    target=target,
                    alias=alias,
                    heading=heading,
                    block_id=block_id,
                )
            )

    return links

def split_alias(raw: str) -> tuple[str, str | None]:
    if "|" not in raw:
        return raw, None

    target, alias = raw.split("|", 1)
    return target.strip(), alias.strip() or None

def split_target(raw: str) -> tuple[str, str | None, str | None]:
    block_id = None
    heading = None
    target = raw.strip()

    if "^" in target:
        target, block_id = target.split("^", 1)
        block_id = block_id.strip() or None

    if "#" in target:
        target, heading = target.split("#", 1)
        heading = heading.strip() or None

    return target.strip(), heading, block_id

def extract_external_links(body: str) -> list[str]:
    return sorted({match.group(2).strip() for match in EXTERNAL_LINK_RE.finditer(body)})


def extract_code_blocks(body: str) -> list[CodeBlock]:
    blocks: list[CodeBlock] = []

    for match in CODE_BLOCK_RE.finditer(body):
        language = match.group(1).strip() or None
        content = match.group(2).strip()
        blocks.append(CodeBlock(language=language, content=content))

    return blocks


def build_excerpt(body: str, max_chars: int = 1200) -> str:
    text = CODE_BLOCK_RE.sub("", body)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
