from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MISSING_RANK = 999_999


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze rank movement before and after reranking.")
    parser.add_argument("--before", required=True, help="Baseline eval result JSON, e.g. hybrid.")
    parser.add_argument("--after", required=True, help="Reranked eval result JSON.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output path prefix or .json/.md path. Both JSON and Markdown are written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of largest improvements/declines shown in Markdown.",
    )
    parser.add_argument(
        "--eval",
        default=None,
        help="Optional eval JSON used to enrich query_type/source_type when old result files lack them.",
    )
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    before_payload = load_payload(before_path)
    after_payload = load_payload(after_path)
    eval_metadata = load_eval_metadata(Path(args.eval)) if args.eval else {}

    movements = build_movements(before_payload, after_payload, eval_metadata=eval_metadata)
    payload = {
        "config": {
            "before": str(before_path),
            "after": str(after_path),
            "before_mode": before_payload.get("config", {}).get("mode"),
            "after_mode": after_payload.get("config", {}).get("mode"),
            "before_reranker_type": before_payload.get("config", {}).get("reranker_type"),
            "after_reranker_type": after_payload.get("config", {}).get("reranker_type"),
            "after_reranker_model": after_payload.get("config", {}).get("reranker_model"),
        },
        "summary": summarize_movements(movements),
        "by_query_type": summarize_by_field(movements, "query_type"),
        "by_source_type": summarize_by_field(movements, "source_type"),
        "movements": movements,
    }

    json_path, markdown_path = resolve_output_paths(Path(args.out))
    write_json(json_path, payload)
    write_markdown(markdown_path, payload, top_n=args.top_n)

    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved Markdown: {markdown_path.resolve()}")
    return 0


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Eval result not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "cases" not in payload or not isinstance(payload["cases"], list):
        raise ValueError(f"Eval result must contain a cases list: {path}")
    return payload


def load_eval_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval metadata file not found: {path}")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"Eval metadata file must be a JSON list: {path}")
    return {
        str(case["query"]): {
            "query_type": case.get("query_type"),
            "source_type": case.get("source_type"),
        }
        for case in cases
        if "query" in case
    }


def build_movements(
    before_payload: dict[str, Any],
    after_payload: dict[str, Any],
    eval_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    before_cases = {case["query"]: case for case in before_payload["cases"]}
    after_cases = {case["query"]: case for case in after_payload["cases"]}

    missing_after = sorted(set(before_cases) - set(after_cases))
    missing_before = sorted(set(after_cases) - set(before_cases))
    if missing_after or missing_before:
        raise ValueError(
            "Eval result query sets differ. "
            f"missing_after={missing_after}, missing_before={missing_before}"
        )

    movements: list[dict[str, Any]] = []
    for query in before_cases:
        before = before_cases[query]
        after = after_cases[query]
        metadata = eval_metadata.get(query, {})

        before_rank = normalize_rank(before.get("hit_rank"))
        after_rank = normalize_rank(after.get("hit_rank"))
        movement = before_rank - after_rank

        movements.append(
            {
                "query": query,
                "query_type": (
                    after.get("query_type")
                    or before.get("query_type")
                    or metadata.get("query_type")
                    or "unknown"
                ),
                "source_type": (
                    after.get("source_type")
                    or before.get("source_type")
                    or metadata.get("source_type")
                    or "unknown"
                ),
                "expected_notes": after.get("expected_notes") or before.get("expected_notes") or [],
                "before_rank": None if before_rank == MISSING_RANK else before_rank,
                "after_rank": None if after_rank == MISSING_RANK else after_rank,
                "movement": movement,
                "status": movement_status(before_rank, after_rank),
                "before_top1_note": top_note(before),
                "after_top1_note": top_note(after),
                "before_top1_heading": top_heading(before),
                "after_top1_heading": top_heading(after),
                "before_top_results": compact_top_results(before, limit=5),
                "after_top_results": compact_top_results(after, limit=5),
            }
        )

    return movements


def normalize_rank(rank: Any) -> int:
    if rank is None:
        return MISSING_RANK
    return int(rank)


def movement_status(before_rank: int, after_rank: int) -> str:
    if before_rank == after_rank:
        return "unchanged"
    if after_rank < before_rank:
        return "improved"
    return "declined"


def top_note(case: dict[str, Any]) -> str:
    results = case.get("top_results") or []
    return str(results[0].get("note_path", "")) if results else ""


def top_heading(case: dict[str, Any]) -> str:
    results = case.get("top_results") or []
    return str(results[0].get("heading", "")) if results else ""


def compact_top_results(case: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for result in (case.get("top_results") or [])[:limit]:
        compact.append(
            {
                "rank": result.get("rank"),
                "note_path": result.get("note_path"),
                "heading": result.get("heading"),
                "score": result.get("score"),
                "lines": result.get("lines"),
            }
        )
    return compact


def summarize_movements(movements: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(movements)
    improved = [item for item in movements if item["status"] == "improved"]
    declined = [item for item in movements if item["status"] == "declined"]
    unchanged = [item for item in movements if item["status"] == "unchanged"]
    values = [int(item["movement"]) for item in movements if abs(int(item["movement"])) < MISSING_RANK]

    return {
        "query_count": total,
        "improved_count": len(improved),
        "declined_count": len(declined),
        "unchanged_count": len(unchanged),
        "avg_movement": sum(values) / len(values) if values else 0.0,
        "max_improvement": max((item["movement"] for item in improved), default=0),
        "max_decline": min((item["movement"] for item in declined), default=0),
    }


def summarize_by_field(movements: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in movements:
        groups.setdefault(str(item.get(field) or "unknown"), []).append(item)

    return {
        group_name: summarize_movements(group_items)
        for group_name, group_items in sorted(groups.items())
    }


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

    lines.append("# Rerank Movement Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Queries: {summary['query_count']}")
    lines.append(f"- Improved: {summary['improved_count']}")
    lines.append(f"- Declined: {summary['declined_count']}")
    lines.append(f"- Unchanged: {summary['unchanged_count']}")
    lines.append(f"- Avg movement: {summary['avg_movement']:.3f}")
    lines.append(f"- Max improvement: {summary['max_improvement']}")
    lines.append(f"- Max decline: {summary['max_decline']}")
    lines.append("")

    write_group_section(lines, "By Query Type", payload["by_query_type"])
    write_group_section(lines, "By Source Type", payload["by_source_type"])

    movements = payload["movements"]
    improvements = sorted(
        [item for item in movements if item["status"] == "improved"],
        key=lambda item: item["movement"],
        reverse=True,
    )
    declines = sorted(
        [item for item in movements if item["status"] == "declined"],
        key=lambda item: item["movement"],
    )

    write_movement_table(lines, f"Largest Improvements Top {top_n}", improvements[:top_n])
    write_movement_table(lines, f"Largest Declines Top {top_n}", declines[:top_n])
    write_movement_table(lines, "All Movements", sorted(movements, key=lambda item: item["query"]))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_group_section(lines: list[str], title: str, groups: dict[str, dict[str, Any]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| group | queries | improved | declined | unchanged | avg movement |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for group_name, summary in groups.items():
        lines.append(
            f"| {escape_cell(group_name)} "
            f"| {summary['query_count']} "
            f"| {summary['improved_count']} "
            f"| {summary['declined_count']} "
            f"| {summary['unchanged_count']} "
            f"| {summary['avg_movement']:.3f} |"
        )
    lines.append("")


def write_movement_table(lines: list[str], title: str, items: list[dict[str, Any]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if not items:
        lines.append("No cases.")
        lines.append("")
        return

    lines.append(
        "| status | movement | before | after | query_type | source_type | query | before top1 | after top1 |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |")
    for item in items:
        lines.append(
            f"| {item['status']} "
            f"| {item['movement']} "
            f"| {rank_cell(item['before_rank'])} "
            f"| {rank_cell(item['after_rank'])} "
            f"| {escape_cell(item['query_type'])} "
            f"| {escape_cell(item['source_type'])} "
            f"| {escape_cell(item['query'])} "
            f"| {escape_cell(item['before_top1_note'])} "
            f"| {escape_cell(item['after_top1_note'])} |"
        )
    lines.append("")


def rank_cell(rank: int | None) -> str:
    return "miss" if rank is None else str(rank)


def escape_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
