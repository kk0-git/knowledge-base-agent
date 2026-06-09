from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from services.rag.schema import SearchResult


ScoreType = Literal["cosine", "bm25", "rrf", "rerank", "unknown"]


@dataclass(frozen=True)
class PackedContextChunk:
    citation_id: int
    note_path: str
    heading: str
    lines: str
    search_score: float
    score_type: ScoreType
    text: str


@dataclass(frozen=True)
class PackedContext:
    chunks: list[PackedContextChunk]
    context_text: str
    total_chars: int
    truncated: bool


def pack_search_results(
    results: list[SearchResult],
    *,
    score_type: ScoreType,
    max_chunks: int = 5,
    max_chars_per_chunk: int = 1200,
    max_context_chars: int = 6000,
) -> PackedContext:
    chunks: list[PackedContextChunk] = []
    sections: list[str] = []
    total_chars = 0
    truncated = False

    for result in results[:max_chunks]:
        chunk = result.chunk
        raw_text = chunk.text.strip()
        text = truncate_text(raw_text, max_chars_per_chunk)
        if len(text) < len(raw_text):
            truncated = True

        packed = PackedContextChunk(
            citation_id=len(chunks) + 1,
            note_path=chunk.note_path,
            heading=" > ".join(chunk.heading_path) if chunk.heading_path else "",
            lines=line_range(chunk.start_line, chunk.end_line),
            search_score=round(float(result.score), 6),
            score_type=score_type,
            text=text,
        )
        section = render_context_section(packed)
        section_chars = len(section)

        if chunks and total_chars + section_chars > max_context_chars:
            truncated = True
            break

        chunks.append(packed)
        sections.append(section)
        total_chars += section_chars

        if total_chars >= max_context_chars:
            truncated = True
            break

    return PackedContext(
        chunks=chunks,
        context_text="\n\n".join(sections),
        total_chars=total_chars,
        truncated=truncated,
    )


def packed_context_to_dict(context: PackedContext) -> dict:
    return {
        "chunks": [asdict(chunk) for chunk in context.chunks],
        "context_text": context.context_text,
        "total_chars": context.total_chars,
        "truncated": context.truncated,
    }


def render_context_section(chunk: PackedContextChunk) -> str:
    heading = chunk.heading or "(no heading)"
    return "\n".join(
        [
            f"[{chunk.citation_id}]",
            f"source: {chunk.note_path}",
            f"heading: {heading}",
            f"lines: {chunk.lines}",
            f"score: {chunk.search_score} ({chunk.score_type})",
            "content:",
            chunk.text,
        ]
    )


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def line_range(start_line: int | None, end_line: int | None) -> str:
    if start_line is None or end_line is None:
        return ""
    return f"{start_line}-{end_line}"


def score_type_for_mode(mode: str) -> ScoreType:
    if mode == "dense":
        return "cosine"
    if mode == "bm25":
        return "bm25"
    if mode == "hybrid":
        return "rrf"
    if mode == "hybrid-rerank":
        return "rerank"
    return "unknown"
