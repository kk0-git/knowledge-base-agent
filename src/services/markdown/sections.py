from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownSection:
    level: int
    heading_path: list[str]
    lines: list[str]
    start_line: int
    end_line: int

    @property
    def heading(self) -> str:
        return self.heading_path[-1] if self.heading_path else ""

    @property
    def content(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def char_count(self) -> int:
        return len(self.content)

    def section_id(self) -> str:
        slug = slugify_heading(self.heading or "section")
        return f"h{self.level}-{slug}-L{self.start_line}"


def normalize_heading_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def slugify_heading(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "section"


def parse_markdown_sections(markdown: str) -> list[MarkdownSection]:
    lines = markdown.splitlines()
    sections: list[MarkdownSection] = []

    heading_stack: list[tuple[int, str]] = []
    current_heading_path: list[str] = []
    current_level = 0
    current_lines: list[str] = []
    current_start = 1
    in_fenced_code = False

    for line_number, line in enumerate(lines, start=1):
        if is_fenced_code_marker(line):
            current_lines.append(line)
            in_fenced_code = not in_fenced_code
            continue

        match = None if in_fenced_code else HEADING_RE.match(line)

        if match:
            if current_lines:
                sections.append(
                    MarkdownSection(
                        level=current_level,
                        heading_path=list(current_heading_path),
                        lines=current_lines,
                        start_line=current_start,
                        end_line=line_number - 1,
                    )
                )

            level = len(match.group(1))
            heading_text = match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, heading_text))
            current_heading_path = [heading for _, heading in heading_stack]
            current_level = level
            current_lines = [line]
            current_start = line_number
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            MarkdownSection(
                level=current_level,
                heading_path=list(current_heading_path),
                lines=current_lines,
                start_line=current_start,
                end_line=len(lines),
            )
        )

    return [section for section in sections if has_content_lines(section.lines)]


split_markdown_sections = parse_markdown_sections


def is_fenced_code_marker(line: str) -> bool:
    return line.lstrip().startswith("```")


def has_content_lines(lines: list[str]) -> bool:
    return any(line.strip() for line in lines)


def section_preview(section: MarkdownSection, *, max_chars: int = 120) -> str:
    body_lines = section.lines[1:] if section.lines and HEADING_RE.match(section.lines[0]) else section.lines
    body = " ".join(line.strip() for line in body_lines if line.strip())
    if len(body) <= max_chars:
        return body
    return body[: max_chars - 3].rstrip() + "..."


def section_to_dict(section: MarkdownSection, *, preview_chars: int = 120) -> dict[str, object]:
    return {
        "id": section.section_id(),
        "level": section.level,
        "heading": section.heading,
        "heading_path": list(section.heading_path),
        "start_line": section.start_line,
        "end_line": section.end_line,
        "char_count": section.char_count,
        "preview": section_preview(section, max_chars=preview_chars),
    }


def child_sections(parent: MarkdownSection, sections: list[MarkdownSection]) -> list[MarkdownSection]:
    parent_path = parent.heading_path
    return [
        section
        for section in sections
        if len(section.heading_path) == len(parent_path) + 1
        and section.heading_path[: len(parent_path)] == parent_path
    ]


def find_sections(
    sections: list[MarkdownSection],
    *,
    heading: str = "",
    section_id: str = "",
    heading_path: list[str] | None = None,
) -> tuple[list[MarkdownSection], str | None]:
    if section_id:
        matches = [section for section in sections if section.section_id() == section_id]
        if not matches:
            return [], "not_found"
        return matches, None

    if heading_path:
        normalized_path = [normalize_heading_text(part) for part in heading_path if str(part).strip()]
        matches = [
            section
            for section in sections
            if [normalize_heading_text(part) for part in section.heading_path] == normalized_path
        ]
        if not matches:
            return [], "not_found"
        if len(matches) > 1:
            return matches, "ambiguous"
        return matches, None

    if heading:
        target = normalize_heading_text(heading)
        matches = [section for section in sections if normalize_heading_text(section.heading) == target]
        if not matches:
            return [], "not_found"
        if len(matches) > 1:
            return matches, "ambiguous"
        return matches, None

    return [], "not_found"


def build_section_map(sections: list[MarkdownSection]) -> list[dict[str, object]]:
    return [
        {
            "section_id": section.section_id(),
            "heading": section.heading,
            "heading_path": list(section.heading_path),
            "start_line": section.start_line,
            "end_line": section.end_line,
            "char_count": section.char_count,
        }
        for section in sections
    ]


def build_note_outline(
    *,
    path: str,
    title: str,
    text: str,
    preview_chars: int = 120,
    max_sections: int = 50,
) -> dict[str, object]:
    sections = parse_markdown_sections(text)
    section_dicts = [section_to_dict(section, preview_chars=preview_chars) for section in sections[:max_sections]]
    return {
        "path": path,
        "title": title,
        "char_count": len(text),
        "heading_count": len(sections),
        "sections": section_dicts,
        "truncated": len(sections) > max_sections,
        "hint": "Use read_note with heading, heading_path, or section_id to read a specific section.",
    }
