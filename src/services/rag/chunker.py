from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from services.rag.schema import TextChunk


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")


@dataclass(frozen=True)
class ChunkerConfig:
    """Heading-based, line-aware chunker configuration.

    max_chunk_chars: hard trigger for splitting an overlong section.
    target_chunk_chars: kept for CLI compatibility; v1 uses max_chunk_chars
        plus structural split points instead of equal-width character windows.
    min_chunk_chars: kept for compatibility; short heading sections are not
        merged in this Lumina-style chunker.
    chunk_overlap: overlap budget in characters for overlong-section splits.
    strip_code_blocks: kept for compatibility. Fenced code blocks are preserved
        as ordinary Markdown text, but overlong-section splitting avoids cutting
        inside fenced code blocks when possible.
    """

    max_chunk_chars: int = 1500
    target_chunk_chars: int = 900
    min_chunk_chars: int = 200
    chunk_overlap: int = 200
    strip_code_blocks: bool = False


@dataclass(frozen=True)
class MarkdownSection:
    heading_path: list[str]
    lines: list[str]
    start_line: int
    end_line: int


@dataclass(frozen=True)
class LineItem:
    line_number: int
    text: str


class HeadingChunker:
    """Markdown heading chunker with Lumina-style line-aware splitting.

    Strategy:
    1. Split Markdown by heading sections.
    2. Keep original line breaks and Markdown formatting.
    3. If a section is too long, split by line accumulation.
    4. Prefer blank-line boundary, then list-item boundary, then 75% fallback.
    5. Add overlap lines only for overlong-section secondary splits.
    6. Use stable chunk IDs: ``note_path:start_line-end_line``.
    7. Do not merge short sections.
    8. Keep fenced code blocks intact when a viable split boundary exists.
    9. Merge low-value heading-only / fence-only chunks as context.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_markdown(self, note_path: str, markdown: str) -> list[TextChunk]:
        if not markdown.strip():
            return []

        sections = split_markdown_sections(markdown)
        chunks: list[TextChunk] = []

        for section in sections:
            chunks.extend(self._chunk_section(note_path, section))

        return merge_low_value_chunks(chunks)

    def chunk_file(self, vault_root: Path, file_path: Path) -> list[TextChunk]:
        markdown = file_path.read_text(encoding="utf-8", errors="replace")
        note_path = file_path.relative_to(vault_root).as_posix()
        return self.chunk_markdown(note_path=note_path, markdown=markdown)

    def _chunk_section(self, note_path: str, section: MarkdownSection) -> list[TextChunk]:
        line_items = [
            LineItem(line_number=section.start_line + index, text=line)
            for index, line in enumerate(section.lines)
        ]

        chunks: list[TextChunk] = []
        current: list[LineItem] = []
        current_length = 0
        split_index_in_section = 0

        for item in line_items:
            current.append(item)
            current_length += len(item.text) + 1

            if current_length <= self.config.max_chunk_chars:
                continue

            raw_split_point = find_split_point([line.text for line in current])
            split_point, split_reason = adjust_split_point_for_fenced_code(
                current,
                raw_split_point,
                self.config,
            )
            first_part_too_short = (
                0 < split_point < len(current)
                and line_items_char_length(current[:split_point]) < minimum_split_chars(self.config)
            )

            if split_point <= 0 or split_point >= len(current) or first_part_too_short:
                if has_content(current):
                    chunks.append(
                        create_chunk(
                            note_path=note_path,
                            heading_path=section.heading_path,
                            line_items=current,
                            split_index_in_section=split_index_in_section,
                            split_reason=split_reason if split_reason == "oversized_code_block" else "force_split",
                        )
                    )
                    split_index_in_section += 1

                current = []
                current_length = 0
                continue

            first_part = current[:split_point]
            second_part = current[split_point:]

            if has_content(first_part):
                chunks.append(
                        create_chunk(
                            note_path=note_path,
                            heading_path=section.heading_path,
                            line_items=first_part,
                            split_index_in_section=split_index_in_section,
                            split_reason=split_reason,
                        )
                    )
                split_index_in_section += 1

            overlap_lines = get_overlap_lines(first_part, self.config.chunk_overlap)
            current = overlap_lines + second_part
            current_length = line_items_char_length(current)

        if current and has_content(current):
            chunks.append(
                create_chunk(
                    note_path=note_path,
                    heading_path=section.heading_path,
                    line_items=current,
                    split_index_in_section=split_index_in_section,
                    split_reason="section_end",
                )
            )

        return chunks


def split_markdown_sections(markdown: str) -> list[MarkdownSection]:
    lines = markdown.splitlines()
    sections: list[MarkdownSection] = []

    heading_stack: list[tuple[int, str]] = []
    current_heading_path: list[str] = []
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
                        heading_path=current_heading_path,
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
            current_lines = [line]
            current_start = line_number
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            MarkdownSection(
                heading_path=current_heading_path,
                lines=current_lines,
                start_line=current_start,
                end_line=len(lines),
            )
        )

    return [section for section in sections if has_content_lines(section.lines)]


def is_fenced_code_marker(line: str) -> bool:
    return line.lstrip().startswith("```")


def create_chunk(
    note_path: str,
    heading_path: list[str],
    line_items: list[LineItem],
    split_index_in_section: int,
    split_reason: str,
) -> TextChunk:
    start_line = line_items[0].line_number
    end_line = line_items[-1].line_number
    text = "\n".join(item.text for item in line_items).strip()

    return TextChunk(
        chunk_id=f"{note_path}:{start_line}-{end_line}",
        note_path=note_path,
        heading_path=heading_path,
        text=text,
        start_line=start_line,
        end_line=end_line,
        metadata={
            "source": "heading_line_chunker",
            "split_index_in_section": split_index_in_section,
            "split_reason": split_reason,
            "char_count": len(text),
        },
    )


def merge_low_value_chunks(chunks: list[TextChunk]) -> list[TextChunk]:
    if not chunks:
        return []

    merged: list[TextChunk] = []
    pending_prefix: list[TextChunk] = []

    for chunk in chunks:
        if is_heading_only_chunk(chunk):
            pending_prefix.append(chunk)
            continue

        if is_fence_only_chunk(chunk):
            if pending_prefix or not merged:
                pending_prefix.append(chunk)
            else:
                merged[-1] = merge_chunks(
                    [merged[-1], chunk],
                    heading_path=merged[-1].heading_path,
                    merge_reason="fence_only_to_previous",
                )
            continue

        if pending_prefix:
            chunk = merge_chunks(
                pending_prefix + [chunk],
                heading_path=chunk.heading_path,
                merge_reason="heading_only_to_next",
            )
            pending_prefix = []

        merged.append(chunk)

    if pending_prefix:
        if merged:
            merged[-1] = merge_chunks(
                [merged[-1]] + pending_prefix,
                heading_path=merged[-1].heading_path,
                merge_reason="trailing_low_value_to_previous",
            )
        else:
            merged.append(
                merge_chunks(
                    pending_prefix,
                    heading_path=pending_prefix[-1].heading_path,
                    merge_reason="only_low_value_chunks",
                )
            )

    return merged


def is_heading_only_chunk(chunk: TextChunk) -> bool:
    if len(chunk.text) > 80:
        return False

    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    if not lines:
        return False

    return all(HEADING_RE.match(line) for line in lines)


def is_fence_only_chunk(chunk: TextChunk) -> bool:
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    if not lines:
        return False

    return all(re.fullmatch(r"`{3,}\w*", line) for line in lines)


def merge_chunks(
    chunks: list[TextChunk],
    heading_path: list[str],
    merge_reason: str,
) -> TextChunk:
    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (
            chunk.start_line if chunk.start_line is not None else 0,
            chunk.end_line if chunk.end_line is not None else 0,
        ),
    )
    first_chunk = ordered_chunks[0]
    last_chunk = ordered_chunks[-1]
    start_line = first_chunk.start_line
    end_line = last_chunk.end_line
    text = "\n".join(chunk.text.strip() for chunk in ordered_chunks if chunk.text.strip()).strip()
    metadata = dict(last_chunk.metadata)
    metadata.update(
        {
            "source": "heading_line_chunker",
            "char_count": len(text),
            "low_value_merge_reason": merge_reason,
            "merged_chunk_count": len(ordered_chunks),
            "merged_from_chunk_ids": [chunk.chunk_id for chunk in ordered_chunks],
            "merged_from_split_reasons": [
                chunk.metadata.get("split_reason") for chunk in ordered_chunks
            ],
        }
    )

    return TextChunk(
        chunk_id=f"{last_chunk.note_path}:{start_line}-{end_line}",
        note_path=last_chunk.note_path,
        heading_path=heading_path,
        text=text,
        start_line=start_line,
        end_line=end_line,
        metadata=metadata,
    )


def find_split_point(lines: list[str]) -> int:
    if len(lines) <= 1:
        return len(lines)

    lower_bound = max(1, int(len(lines) * 0.5))

    # Prefer paragraph boundary, scanning backward from the end.
    for index in range(len(lines) - 1, lower_bound, -1):
        if lines[index].strip() == "":
            return index

    # Then prefer list item boundary.
    for index in range(len(lines) - 1, lower_bound, -1):
        if LIST_ITEM_RE.match(lines[index]):
            return index

    # Fallback: split at 75% of the accumulated lines.
    return max(1, int(len(lines) * 0.75))


def adjust_split_point_for_fenced_code(
    line_items: list[LineItem],
    split_point: int,
    config: ChunkerConfig,
) -> tuple[int, str]:
    if split_point <= 0 or split_point >= len(line_items):
        return split_point, "overlong_section"

    fenced_range = find_fenced_range_containing_boundary(
        [item.text for item in line_items],
        split_point,
    )
    if fenced_range is None:
        return split_point, "overlong_section"

    code_start, code_end = fenced_range
    after_code = code_end + 1
    before_code = code_start

    if after_code < len(line_items):
        after_code_part = line_items[:after_code]
        if line_items_char_length(after_code_part) <= config.max_chunk_chars:
            return after_code, "overlong_section"

    if before_code > 0:
        before_code_part = line_items[:before_code]
        if line_items_char_length(before_code_part) >= minimum_split_chars(config):
            return before_code, "overlong_section"

    if after_code == len(line_items):
        return after_code, "oversized_code_block"

    return split_point, "oversized_code_block"


def find_fenced_range_containing_boundary(lines: list[str], boundary_index: int) -> tuple[int, int] | None:
    in_code = False
    start_index = 0

    for index, line in enumerate(lines):
        if not line.lstrip().startswith("```"):
            continue

        if not in_code:
            in_code = True
            start_index = index
            continue

        if start_index < boundary_index <= index:
            return start_index, index
        in_code = False

    if in_code and start_index < boundary_index <= len(lines) - 1:
        return start_index, len(lines) - 1

    return None


def get_overlap_lines(line_items: list[LineItem], chunk_overlap: int) -> list[LineItem]:
    if chunk_overlap <= 0:
        return []

    char_count = 0
    overlap: list[LineItem] = []

    for item in reversed(line_items):
        overlap.insert(0, item)
        char_count += len(item.text) + 1
        if char_count >= chunk_overlap:
            break

    return overlap


def line_items_char_length(line_items: list[LineItem]) -> int:
    return sum(len(item.text) + 1 for item in line_items)


def minimum_split_chars(config: ChunkerConfig) -> int:
    return min(config.min_chunk_chars, max(1, int(config.max_chunk_chars * 0.4)))


def has_content(line_items: list[LineItem]) -> bool:
    return any(item.text.strip() for item in line_items)


def has_content_lines(lines: list[str]) -> bool:
    return any(line.strip() for line in lines)
