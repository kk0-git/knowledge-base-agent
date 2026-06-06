from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from knowledge_base_agent.config import load_exclusion_patterns
from knowledge_base_agent.scanner import ExclusionFilter, scan_vault
from services.rag.chunker import ChunkerConfig, HeadingChunker, split_markdown_sections
from services.rag.schema import TextChunk


LENGTH_BUCKETS = [
    ("0-100", 0, 100),
    ("101-200", 101, 200),
    ("201-500", 201, 500),
    ("501-900", 501, 900),
    ("901-1500", 901, 1500),
    ("1501-2600", 1501, 2600),
    ("2601+", 2601, None),
]


@dataclass(frozen=True)
class NoteChunkStats:
    note_path: str
    source_type: str
    file_chars: int
    section_count: int
    chunk_count: int
    avg_chunk_chars: float
    max_chunk_chars: int
    overlong_chunk_count: int
    split_chunk_count: int
    overlap_candidate_count: int
    code_fence_count: int
    possible_code_boundary_issue: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Markdown chunk distribution without embedding.")
    parser.add_argument("--vault", required=True, help="Path to Obsidian vault")
    parser.add_argument("--out", default="./eval-results/chunk-stats", help="Output path prefix or .json/.md path")
    parser.add_argument("--max-chunk-chars", type=int, default=1500)
    parser.add_argument("--target-chunk-chars", type=int, default=900)
    parser.add_argument("--min-chunk-chars", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    vault_path = Path(args.vault).resolve()
    exclusion_filter = ExclusionFilter(load_exclusion_patterns(vault_path))
    scan_result = scan_vault(vault_path, exclusion_filter)

    config = ChunkerConfig(
        max_chunk_chars=args.max_chunk_chars,
        target_chunk_chars=args.target_chunk_chars,
        min_chunk_chars=args.min_chunk_chars,
        chunk_overlap=args.chunk_overlap,
    )
    chunker = HeadingChunker(config)

    note_stats: list[NoteChunkStats] = []
    all_chunks: list[dict[str, Any]] = []
    code_boundary_issues: list[dict[str, Any]] = []
    split_reason_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    heading_depth_counter: Counter[int] = Counter()

    for note in scan_result.notes:
        markdown = note.path.read_text(encoding="utf-8", errors="replace")
        chunks = chunker.chunk_markdown(note_path=note.relative_path, markdown=markdown)
        sections = split_markdown_sections(markdown)
        source_type = infer_source_type(note.relative_path)

        chunk_lengths = [len(chunk.text) for chunk in chunks]
        split_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("split_reason") in {"overlong_section", "force_split"}
        ]
        overlap_candidates = [
            chunk
            for chunk in chunks
            if int(chunk.metadata.get("split_index_in_section", 0)) > 0
        ]
        code_fence_count = markdown.count("```")
        note_code_boundary_issues = find_code_boundary_issues(note.relative_path, markdown, chunks)
        possible_code_boundary_issue = bool(note_code_boundary_issues)
        code_boundary_issues.extend(note_code_boundary_issues)

        note_stats.append(
            NoteChunkStats(
                note_path=note.relative_path,
                source_type=source_type,
                file_chars=len(markdown),
                section_count=len(sections),
                chunk_count=len(chunks),
                avg_chunk_chars=round(statistics.mean(chunk_lengths), 2) if chunk_lengths else 0.0,
                max_chunk_chars=max(chunk_lengths) if chunk_lengths else 0,
                overlong_chunk_count=sum(1 for length in chunk_lengths if length > args.max_chunk_chars),
                split_chunk_count=len(split_chunks),
                overlap_candidate_count=len(overlap_candidates),
                code_fence_count=code_fence_count,
                possible_code_boundary_issue=possible_code_boundary_issue,
            )
        )

        source_counter[source_type] += len(chunks)
        for chunk in chunks:
            split_reason_counter[str(chunk.metadata.get("split_reason", "unknown"))] += 1
            heading_depth_counter[len(chunk.heading_path)] += 1
            all_chunks.append(compact_chunk(chunk, source_type))

    payload = {
        "config": {
            "vault": str(vault_path),
            "max_chunk_chars": args.max_chunk_chars,
            "target_chunk_chars": args.target_chunk_chars,
            "min_chunk_chars": args.min_chunk_chars,
            "chunk_overlap": args.chunk_overlap,
        },
        "summary": build_summary(scan_result.notes, note_stats, all_chunks),
        "by_source_type": summarize_by_source(note_stats),
        "split_reason_counts": dict(sorted(split_reason_counter.items())),
        "heading_depth_counts": {str(k): v for k, v in sorted(heading_depth_counter.items())},
        "chunk_source_counts": dict(sorted(source_counter.items())),
        "chunk_length_distribution": summarize_chunk_lengths(all_chunks, args),
        "chunk_length_distribution_by_source": summarize_chunk_lengths_by_source(all_chunks, args),
        "top_notes_by_chunk_count": top_notes(note_stats, key="chunk_count", limit=args.top_n),
        "top_notes_by_max_chunk_chars": top_notes(note_stats, key="max_chunk_chars", limit=args.top_n),
        "top_shortest_chunks": sorted(all_chunks, key=lambda item: item["char_count"])[: args.top_n],
        "top_longest_chunks": sorted(all_chunks, key=lambda item: item["char_count"], reverse=True)[: args.top_n],
        "code_boundary_issues": code_boundary_issues,
        "notes_with_possible_code_boundary_issues": [
            asdict(item) for item in note_stats if item.possible_code_boundary_issue
        ],
        "notes": [asdict(item) for item in note_stats],
    }

    json_path, markdown_path = resolve_output_paths(Path(args.out))
    write_json(json_path, payload)
    write_markdown(markdown_path, payload, top_n=args.top_n)

    print(f"Markdown files: {len(scan_result.notes)}")
    print(f"Chunks: {payload['summary']['chunk_count']}")
    print(f"Split chunks: {payload['summary']['split_chunk_count']}")
    print(f"Overlap candidates: {payload['summary']['overlap_candidate_count']}")
    print(f"Code boundary issues: {len(payload['code_boundary_issues'])}")
    print(
        "Notes with possible code boundary issues: "
        f"{len(payload['notes_with_possible_code_boundary_issues'])}"
    )
    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved Markdown: {markdown_path.resolve()}")
    return 0


def infer_source_type(note_path: str) -> str:
    normalized = note_path.replace("\\", "/")
    if normalized.startswith("imported_docs/textbooks-mineru-agent/"):
        return "textbook_pdf"
    if normalized.startswith("imported_docs/"):
        return "imported_docs"
    if normalized.startswith("docs/"):
        return "official_docs"
    if normalized.startswith("papers/"):
        return "paper_or_paper_md"
    return "personal_note"


def compact_chunk(chunk: TextChunk, source_type: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "note_path": chunk.note_path,
        "source_type": source_type,
        "heading": " > ".join(chunk.heading_path),
        "heading_depth": len(chunk.heading_path),
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "char_count": len(chunk.text),
        "split_reason": chunk.metadata.get("split_reason"),
        "split_index_in_section": chunk.metadata.get("split_index_in_section"),
        "preview": chunk.text[:160],
    }


def find_code_boundary_issues(note_path: str, markdown: str, chunks: list[TextChunk]) -> list[dict[str, Any]]:
    fenced_ranges = find_fenced_code_ranges(markdown)
    if not fenced_ranges:
        return []

    boundaries: dict[int, dict[str, Any]] = {}
    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.start_line or 0, chunk.end_line or 0),
    )
    for index, chunk in enumerate(sorted_chunks):
        if chunk.start_line is not None:
            boundary_info = boundaries.setdefault(chunk.start_line, {})
            boundary_info["next_chunk"] = compact_boundary_chunk(chunk)
            boundary_info.setdefault("boundary_kinds", set()).add("chunk_start")
            if index > 0:
                boundary_info["previous_chunk"] = compact_boundary_chunk(sorted_chunks[index - 1])
        if chunk.end_line is not None:
            boundary_line = chunk.end_line + 1
            boundary_info = boundaries.setdefault(boundary_line, {})
            boundary_info["previous_chunk"] = compact_boundary_chunk(chunk)
            boundary_info.setdefault("boundary_kinds", set()).add("chunk_end_next_line")
            if index + 1 < len(sorted_chunks):
                boundary_info["next_chunk"] = compact_boundary_chunk(sorted_chunks[index + 1])

    issues: list[dict[str, Any]] = []
    for start, end in fenced_ranges:
        for boundary, boundary_info in boundaries.items():
            if start < boundary <= end:
                previous_chunk = boundary_info.get("previous_chunk")
                next_chunk = boundary_info.get("next_chunk")
                issues.append(
                    {
                        "note_path": note_path,
                        "code_start_line": start,
                        "code_end_line": end,
                        "boundary_line": boundary,
                        "boundary_kinds": sorted(boundary_info.get("boundary_kinds", [])),
                        "is_overlap_boundary": is_overlap_boundary(previous_chunk, next_chunk),
                        "previous_chunk": previous_chunk,
                        "next_chunk": next_chunk,
                    }
                )
    return issues


def compact_boundary_chunk(chunk: TextChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "heading": " > ".join(chunk.heading_path),
        "split_reason": chunk.metadata.get("split_reason"),
        "char_count": len(chunk.text),
    }


def is_overlap_boundary(previous_chunk: dict[str, Any] | None, next_chunk: dict[str, Any] | None) -> bool:
    if not previous_chunk or not next_chunk:
        return False
    previous_end = previous_chunk.get("end_line")
    next_start = next_chunk.get("start_line")
    if previous_end is None or next_start is None:
        return False
    return int(next_start) <= int(previous_end)


def find_fenced_code_ranges(markdown: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    in_code = False
    start_line = 0

    for line_number, line in enumerate(markdown.splitlines(), start=1):
        if not line.lstrip().startswith("```"):
            continue
        if not in_code:
            in_code = True
            start_line = line_number
        else:
            ranges.append((start_line, line_number))
            in_code = False

    if in_code:
        ranges.append((start_line, line_number if "line_number" in locals() else start_line))
    return ranges


def build_summary(
    notes: list[Any],
    note_stats: list[NoteChunkStats],
    all_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    chunk_lengths = [chunk["char_count"] for chunk in all_chunks]
    return {
        "note_count": len(notes),
        "chunk_count": len(all_chunks),
        "avg_chunks_per_note": round(len(all_chunks) / len(notes), 3) if notes else 0.0,
        "avg_chunk_chars": round(statistics.mean(chunk_lengths), 2) if chunk_lengths else 0.0,
        "median_chunk_chars": round(statistics.median(chunk_lengths), 2) if chunk_lengths else 0.0,
        "max_chunk_chars": max(chunk_lengths) if chunk_lengths else 0,
        "split_chunk_count": sum(item.split_chunk_count for item in note_stats),
        "overlap_candidate_count": sum(item.overlap_candidate_count for item in note_stats),
        "notes_with_possible_code_boundary_issue": sum(
            1 for item in note_stats if item.possible_code_boundary_issue
        ),
    }


def summarize_by_source(note_stats: list[NoteChunkStats]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[NoteChunkStats]] = defaultdict(list)
    for item in note_stats:
        groups[item.source_type].append(item)

    payload: dict[str, dict[str, Any]] = {}
    for source_type, items in sorted(groups.items()):
        chunk_counts = [item.chunk_count for item in items]
        max_lengths = [item.max_chunk_chars for item in items]
        payload[source_type] = {
            "note_count": len(items),
            "chunk_count": sum(chunk_counts),
            "avg_chunks_per_note": round(statistics.mean(chunk_counts), 2) if chunk_counts else 0.0,
            "max_note_chunk_count": max(chunk_counts) if chunk_counts else 0,
            "avg_max_chunk_chars": round(statistics.mean(max_lengths), 2) if max_lengths else 0.0,
            "split_chunk_count": sum(item.split_chunk_count for item in items),
            "overlap_candidate_count": sum(item.overlap_candidate_count for item in items),
            "possible_code_boundary_issue_count": sum(
                1 for item in items if item.possible_code_boundary_issue
            ),
        }
    return payload


def summarize_chunk_lengths(chunks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    lengths = [int(chunk["char_count"]) for chunk in chunks]
    total = len(lengths)
    return {
        "total_chunks": total,
        "thresholds": {
            "below_min_chunk_chars": count_with_pct(
                sum(1 for length in lengths if length < args.min_chunk_chars),
                total,
            ),
            "below_target_chunk_chars": count_with_pct(
                sum(1 for length in lengths if length < args.target_chunk_chars),
                total,
            ),
            "over_target_chunk_chars": count_with_pct(
                sum(1 for length in lengths if length > args.target_chunk_chars),
                total,
            ),
            "over_max_chunk_chars": count_with_pct(
                sum(1 for length in lengths if length > args.max_chunk_chars),
                total,
            ),
        },
        "buckets": length_bucket_counts(lengths),
    }


def summarize_chunk_lengths_by_source(
    chunks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        groups[str(chunk["source_type"])].append(chunk)

    payload: dict[str, dict[str, Any]] = {}
    for source_type, source_chunks in sorted(groups.items()):
        lengths = [int(chunk["char_count"]) for chunk in source_chunks]
        total = len(lengths)
        payload[source_type] = {
            "total_chunks": total,
            "avg_chunk_chars": round(statistics.mean(lengths), 2) if lengths else 0.0,
            "median_chunk_chars": round(statistics.median(lengths), 2) if lengths else 0.0,
            "min_chunk_chars": min(lengths) if lengths else 0,
            "max_chunk_chars": max(lengths) if lengths else 0,
            "below_min_chunk_chars": count_with_pct(
                sum(1 for length in lengths if length < args.min_chunk_chars),
                total,
            ),
            "over_max_chunk_chars": count_with_pct(
                sum(1 for length in lengths if length > args.max_chunk_chars),
                total,
            ),
            "buckets": length_bucket_counts(lengths),
        }
    return payload


def length_bucket_counts(lengths: list[int]) -> dict[str, dict[str, Any]]:
    total = len(lengths)
    counts: dict[str, dict[str, Any]] = {}
    for label, start, end in LENGTH_BUCKETS:
        if end is None:
            count = sum(1 for length in lengths if length >= start)
        else:
            count = sum(1 for length in lengths if start <= length <= end)
        counts[label] = count_with_pct(count, total)
    return counts


def count_with_pct(count: int, total: int) -> dict[str, Any]:
    return {
        "count": count,
        "pct": round(count / total, 4) if total else 0.0,
    }


def top_notes(note_stats: list[NoteChunkStats], key: str, limit: int) -> list[dict[str, Any]]:
    return [
        asdict(item)
        for item in sorted(note_stats, key=lambda item: getattr(item, key), reverse=True)[:limit]
    ]


def resolve_output_paths(out: Path) -> tuple[Path, Path]:
    if out.suffix == ".json":
        return out, out.with_suffix(".md")
    if out.suffix == ".md":
        return out.with_suffix(".json"), out
    return out.with_suffix(".json"), out.with_suffix(".md")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any], top_n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    summary = payload["summary"]

    lines.append("# Chunk Statistics")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## By Source Type")
    lines.append("")
    lines.append("| source_type | notes | chunks | avg chunks/note | split chunks | overlap candidates | code boundary issues |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for source_type, item in payload["by_source_type"].items():
        lines.append(
            f"| {source_type} "
            f"| {item['note_count']} "
            f"| {item['chunk_count']} "
            f"| {item['avg_chunks_per_note']} "
            f"| {item['split_chunk_count']} "
            f"| {item['overlap_candidate_count']} "
            f"| {item['possible_code_boundary_issue_count']} |"
        )
    lines.append("")

    write_length_distribution(lines, "Chunk Length Distribution", payload["chunk_length_distribution"])
    write_length_distribution_by_source(
        lines,
        "Chunk Length Distribution By Source",
        payload["chunk_length_distribution_by_source"],
    )

    lines.append("## Split Reasons")
    lines.append("")
    for key, value in payload["split_reason_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    write_note_table(lines, f"Top Notes By Chunk Count Top {top_n}", payload["top_notes_by_chunk_count"])
    write_note_table(lines, f"Top Notes By Max Chunk Length Top {top_n}", payload["top_notes_by_max_chunk_chars"])
    write_chunk_table(lines, f"Shortest Chunks Top {top_n}", payload["top_shortest_chunks"])
    write_chunk_table(lines, f"Longest Chunks Top {top_n}", payload["top_longest_chunks"])
    write_code_boundary_table(lines, "Code Boundary Issues", payload["code_boundary_issues"])
    write_note_table(
        lines,
        "Notes With Possible Code Boundary Issues",
        payload["notes_with_possible_code_boundary_issues"],
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_length_distribution(lines: list[str], title: str, item: dict[str, Any]) -> None:
    lines.append(f"## {title}")
    lines.append("")

    lines.append("### Thresholds")
    lines.append("")
    lines.append("| metric | count | pct |")
    lines.append("| --- | ---: | ---: |")
    for key, value in item["thresholds"].items():
        lines.append(f"| {key} | {value['count']} | {format_pct(value['pct'])} |")
    lines.append("")

    lines.append("### Buckets")
    lines.append("")
    lines.append("| bucket | count | pct |")
    lines.append("| --- | ---: | ---: |")
    for bucket, value in item["buckets"].items():
        lines.append(f"| {bucket} | {value['count']} | {format_pct(value['pct'])} |")
    lines.append("")


def write_length_distribution_by_source(
    lines: list[str],
    title: str,
    items: dict[str, dict[str, Any]],
) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "| source | chunks | avg chars | median chars | min | max | below min | over max |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for source_type, item in items.items():
        below_min = item["below_min_chunk_chars"]
        over_max = item["over_max_chunk_chars"]
        lines.append(
            f"| {source_type} "
            f"| {item['total_chunks']} "
            f"| {item['avg_chunk_chars']} "
            f"| {item['median_chunk_chars']} "
            f"| {item['min_chunk_chars']} "
            f"| {item['max_chunk_chars']} "
            f"| {below_min['count']} ({format_pct(below_min['pct'])}) "
            f"| {over_max['count']} ({format_pct(over_max['pct'])}) |"
        )
    lines.append("")

    for source_type, item in items.items():
        lines.append(f"### {source_type} Buckets")
        lines.append("")
        lines.append("| bucket | count | pct |")
        lines.append("| --- | ---: | ---: |")
        for bucket, value in item["buckets"].items():
            lines.append(f"| {bucket} | {value['count']} | {format_pct(value['pct'])} |")
        lines.append("")


def write_note_table(lines: list[str], title: str, items: list[dict[str, Any]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if not items:
        lines.append("No cases.")
        lines.append("")
        return
    lines.append("| note | source | chunks | sections | max chars | split chunks | overlap candidates | code issue |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in items:
        lines.append(
            f"| {escape_cell(item['note_path'])} "
            f"| {item['source_type']} "
            f"| {item['chunk_count']} "
            f"| {item['section_count']} "
            f"| {item['max_chunk_chars']} "
            f"| {item['split_chunk_count']} "
            f"| {item['overlap_candidate_count']} "
            f"| {item['possible_code_boundary_issue']} |"
        )
    lines.append("")


def write_chunk_table(lines: list[str], title: str, items: list[dict[str, Any]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if not items:
        lines.append("No cases.")
        lines.append("")
        return
    lines.append("| note | heading | chars | split_reason | lines |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for item in items:
        lines.append(
            f"| {escape_cell(item['note_path'])} "
            f"| {escape_cell(item['heading'])} "
            f"| {item['char_count']} "
            f"| {item['split_reason']} "
            f"| {item['start_line']}-{item['end_line']} |"
        )
    lines.append("")


def write_code_boundary_table(lines: list[str], title: str, items: list[dict[str, Any]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if not items:
        lines.append("No cases.")
        lines.append("")
        return

    lines.append(
        "| note | code range | boundary line | kinds | overlap | previous chunk | next chunk |"
    )
    lines.append("| --- | --- | ---: | --- | --- | --- | --- |")
    for item in items:
        previous_chunk = item.get("previous_chunk") or {}
        next_chunk = item.get("next_chunk") or {}
        lines.append(
            f"| {escape_cell(item['note_path'])} "
            f"| {item['code_start_line']}-{item['code_end_line']} "
            f"| {item['boundary_line']} "
            f"| {escape_cell(','.join(item.get('boundary_kinds', [])))} "
            f"| {item.get('is_overlap_boundary')} "
            f"| {format_boundary_chunk(previous_chunk)} "
            f"| {format_boundary_chunk(next_chunk)} |"
        )
    lines.append("")


def format_boundary_chunk(chunk: dict[str, Any]) -> str:
    if not chunk:
        return ""
    return escape_cell(
        f"{chunk.get('chunk_id')} "
        f"lines {chunk.get('start_line')}-{chunk.get('end_line')} "
        f"reason={chunk.get('split_reason')} "
        f"chars={chunk.get('char_count')}"
    )


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
