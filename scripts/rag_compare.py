from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rag_eval
from services.rag.bm25 import BM25Index
from services.rag.chunker import HeadingChunker
from services.rag.embedder import SentenceTransformerEmbedder
from services.rag.manager import RAGManager
from services.rag.memory_vector_store import MemoryVectorStore
from services.rag.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker, DashScopeReranker


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_STRATEGIES = "dense,bm25,hybrid,local-rerank"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare multiple RAG retrieval modes")
    parser.add_argument("--index", default="./rag-index/index.json", help="Vector index JSON path")
    parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    parser.add_argument("--eval", default="./eval/rag_eval.json", help="Eval JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer embedding model")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hit-ks", default="1,3,5,10,20")
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--local-reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--dashscope-reranker-model", default="qwen3-rerank")
    parser.add_argument(
        "--strategies",
        default=DEFAULT_STRATEGIES,
        help=(
            "Comma-separated strategies: dense,bm25,hybrid,local-rerank,"
            "dashscope-rerank"
        ),
    )
    parser.add_argument(
        "--baseline",
        default="hybrid",
        help="Strategy name used for per-query delta reporting",
    )
    parser.add_argument(
        "--out",
        default="./eval-results/rag-compare",
        help="Output path prefix, or .json/.md path. Both JSON and Markdown are written.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file loaded before DashScope rerank",
    )
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    index_path = Path(args.index)
    bm25_index_path = Path(args.bm25_index) if args.bm25_index else rag_eval.derive_bm25_index_path(index_path)
    eval_cases = rag_eval.load_eval_cases(Path(args.eval))
    hit_ks = rag_eval.parse_hit_ks(args.hit_ks, args.top_k)
    strategies = parse_strategies(args.strategies)
    if args.baseline not in strategies:
        raise ValueError(
            f"baseline must be one of strategies. baseline={args.baseline}, "
            f"strategies={','.join(strategies)}"
        )

    validate_inputs(
        index_path=index_path,
        bm25_index_path=bm25_index_path,
        strategies=strategies,
    )

    shared = build_shared_resources(
        args=args,
        index_path=index_path,
        bm25_index_path=bm25_index_path,
        strategies=strategies,
    )

    strategy_payloads: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        print(f"Running strategy: {strategy}")
        manager = build_strategy_manager(strategy=strategy, args=args, shared=shared)
        results = [
            rag_eval.evaluate_case(
                manager,
                case,
                top_k=args.top_k,
                hit_ks=hit_ks,
                mode=strategy_to_eval_mode(strategy),
                dense_top_k=args.dense_top_k,
                bm25_top_k=args.bm25_top_k,
                rrf_k=args.rrf_k,
                rerank_candidates=args.rerank_candidates,
            )
            for case in eval_cases
        ]

        strategy_payloads[strategy] = {
            "config": strategy_config(args, strategy, hit_ks, index_path, bm25_index_path),
            "metrics": rag_eval.compute_metrics(results, hit_ks),
            "summary": {
                "query_count": len(results),
                "failure_count": sum(1 for result in results if result.hit_rank is None),
                "diagnosis_counts": rag_eval.count_diagnoses(results),
            },
            "cases": [asdict(result) for result in results],
        }

    payload = {
        "config": {
            "index": str(index_path),
            "bm25_index": str(bm25_index_path),
            "eval": args.eval,
            "model": args.model,
            "top_k": args.top_k,
            "hit_ks": hit_ks,
            "dense_top_k": args.dense_top_k,
            "bm25_top_k": args.bm25_top_k,
            "rrf_k": args.rrf_k,
            "rerank_candidates": args.rerank_candidates,
            "strategies": strategies,
            "baseline": args.baseline,
        },
        "strategies": strategy_payloads,
        "comparison": build_comparison(strategy_payloads, baseline=args.baseline, hit_ks=hit_ks),
        "eval_cases": eval_cases,
    }

    json_path, markdown_path = resolve_output_paths(Path(args.out))
    write_json(json_path, payload)
    write_markdown(markdown_path, payload)

    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved Markdown: {markdown_path.resolve()}")
    return 0


def parse_strategies(raw: str) -> list[str]:
    allowed = {"dense", "bm25", "hybrid", "local-rerank", "dashscope-rerank"}
    strategies = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [strategy for strategy in strategies if strategy not in allowed]
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    if not strategies:
        raise ValueError("At least one strategy is required")
    return strategies


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def validate_inputs(index_path: Path, bm25_index_path: Path, strategies: list[str]) -> None:
    needs_dense = any(strategy in {"dense", "hybrid", "local-rerank", "dashscope-rerank"} for strategy in strategies)
    needs_bm25 = any(strategy in {"bm25", "hybrid", "local-rerank", "dashscope-rerank"} for strategy in strategies)

    if needs_dense and not index_path.exists():
        raise FileNotFoundError(f"Vector index not found: {index_path}")
    if needs_bm25 and not bm25_index_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_index_path}")


def build_shared_resources(
    args: argparse.Namespace,
    index_path: Path,
    bm25_index_path: Path,
    strategies: list[str],
) -> dict[str, Any]:
    needs_dense = any(strategy in {"dense", "hybrid", "local-rerank", "dashscope-rerank"} for strategy in strategies)
    needs_bm25 = any(strategy in {"bm25", "hybrid", "local-rerank", "dashscope-rerank"} for strategy in strategies)

    embedder = SentenceTransformerEmbedder(model_name=args.model) if needs_dense else None
    vector_store = MemoryVectorStore(persist_path=index_path if index_path.exists() else None)
    bm25_index = BM25Index(persist_path=bm25_index_path) if needs_bm25 else None

    return {
        "embedder": embedder,
        "vector_store": vector_store,
        "bm25_index": bm25_index,
        "local_reranker": None,
        "dashscope_reranker": None,
    }


def build_strategy_manager(
    strategy: str,
    args: argparse.Namespace,
    shared: dict[str, Any],
) -> RAGManager:
    reranker = None
    if strategy == "local-rerank":
        if shared["local_reranker"] is None:
            shared["local_reranker"] = CrossEncoderReranker(
                model_name=args.local_reranker_model,
                batch_size=args.rerank_batch_size,
                max_length=args.rerank_max_length,
            )
        reranker = shared["local_reranker"]
    elif strategy == "dashscope-rerank":
        if shared["dashscope_reranker"] is None:
            shared["dashscope_reranker"] = DashScopeReranker(
                model_name=args.dashscope_reranker_model,
            )
        reranker = shared["dashscope_reranker"]

    return RAGManager(
        chunker=HeadingChunker(),
        embedder=shared["embedder"],
        vector_store=shared["vector_store"],
        bm25_index=shared["bm25_index"],
        reranker=reranker,
    )


def strategy_to_eval_mode(strategy: str) -> str:
    if strategy in {"local-rerank", "dashscope-rerank"}:
        return "hybrid-rerank"
    return strategy


def strategy_config(
    args: argparse.Namespace,
    strategy: str,
    hit_ks: list[int],
    index_path: Path,
    bm25_index_path: Path,
) -> dict[str, Any]:
    config = {
        "strategy": strategy,
        "index": str(index_path),
        "bm25_index": str(bm25_index_path),
        "eval": args.eval,
        "model": args.model,
        "top_k": args.top_k,
        "hit_ks": hit_ks,
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
        "rrf_k": args.rrf_k,
    }

    if strategy == "local-rerank":
        config.update({
            "reranker_type": "local",
            "reranker_model": args.local_reranker_model,
            "rerank_candidates": args.rerank_candidates,
            "rerank_batch_size": args.rerank_batch_size,
            "rerank_max_length": args.rerank_max_length,
        })
    elif strategy == "dashscope-rerank":
        config.update({
            "reranker_type": "dashscope",
            "reranker_model": args.dashscope_reranker_model,
            "rerank_candidates": args.rerank_candidates,
        })

    return config


def build_comparison(
    strategy_payloads: dict[str, dict[str, Any]],
    baseline: str,
    hit_ks: list[int],
) -> dict[str, Any]:
    metric_rows = {
        strategy: payload["metrics"]
        for strategy, payload in strategy_payloads.items()
    }

    per_query = build_per_query_comparison(strategy_payloads, baseline=baseline, hit_ks=hit_ks)
    return {
        "metrics": metric_rows,
        "per_query": per_query,
    }


def build_per_query_comparison(
    strategy_payloads: dict[str, dict[str, Any]],
    baseline: str,
    hit_ks: list[int],
) -> list[dict[str, Any]]:
    if not strategy_payloads:
        return []

    first_strategy = next(iter(strategy_payloads))
    queries = [case["query"] for case in strategy_payloads[first_strategy]["cases"]]
    cases_by_strategy = {
        strategy: {case["query"]: case for case in payload["cases"]}
        for strategy, payload in strategy_payloads.items()
    }
    baseline_cases = cases_by_strategy.get(baseline, {})
    max_k = max(hit_ks) if hit_ks else 0

    rows: list[dict[str, Any]] = []
    for query in queries:
        row: dict[str, Any] = {"query": query}
        baseline_case = baseline_cases.get(query)
        baseline_cov = get_coverage(baseline_case, max_k)
        baseline_top = get_top_note(baseline_case)

        for strategy, cases in cases_by_strategy.items():
            case = cases.get(query)
            row[strategy] = {
                "hit_rank": case.get("hit_rank") if case else None,
                f"coverage@{max_k}": get_coverage(case, max_k),
                f"unique_notes@{max_k}": get_diversity_value(case, f"unique_notes@{max_k}"),
                f"max_chunks_per_note@{max_k}": get_diversity_value(case, f"max_chunks_per_note@{max_k}"),
                "top_note": get_top_note(case),
                "top_lines": get_top_lines(case),
            }

            if strategy != baseline:
                row[strategy]["coverage_delta_vs_baseline"] = round(
                    get_coverage(case, max_k) - baseline_cov,
                    4,
                )
                row[strategy]["top_changed_vs_baseline"] = get_top_note(case) != baseline_top

        rows.append(row)

    return rows


def get_coverage(case: dict[str, Any] | None, k: int) -> float:
    if case is None:
        return 0.0
    return float(case.get("diversity", {}).get(f"expected_note_coverage@{k}", 0.0))


def get_diversity_value(case: dict[str, Any] | None, key: str) -> Any:
    if case is None:
        return None
    return case.get("diversity", {}).get(key)


def get_top_note(case: dict[str, Any] | None) -> str:
    if not case or not case.get("top_results"):
        return ""
    return str(case["top_results"][0].get("note_path", ""))


def get_top_lines(case: dict[str, Any] | None) -> str:
    if not case or not case.get("top_results"):
        return ""
    return str(case["top_results"][0].get("lines", ""))


def resolve_output_paths(out: Path) -> tuple[Path, Path]:
    if out.suffix == ".json":
        return out, out.with_suffix(".md")
    if out.suffix == ".md":
        return out.with_suffix(".json"), out
    return out.with_suffix(".json"), out.with_suffix(".md")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    strategies = config["strategies"]
    hit_ks = config["hit_ks"]
    metrics_by_strategy = payload["comparison"]["metrics"]
    per_query = payload["comparison"]["per_query"]

    lines: list[str] = []
    lines.append("# RAG Compare")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- index: `{config['index']}`")
    lines.append(f"- eval: `{config['eval']}`")
    lines.append(f"- model: `{config['model']}`")
    lines.append(f"- top_k: `{config['top_k']}`")
    lines.append(f"- hit_ks: `{','.join(str(k) for k in hit_ks)}`")
    lines.append(f"- strategies: `{','.join(strategies)}`")
    lines.append(f"- baseline: `{config['baseline']}`")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    metric_names = metric_columns(hit_ks)
    lines.append("| strategy | " + " | ".join(metric_names) + " |")
    lines.append("|---" + "|---:" * len(metric_names) + "|")
    for strategy in strategies:
        metrics = metrics_by_strategy[strategy]
        values = [format_metric(metrics.get(name)) for name in metric_names]
        lines.append(f"| {strategy} | " + " | ".join(values) + " |")
    lines.append("")

    lines.append("## Per Query")
    lines.append("")
    max_k = max(hit_ks) if hit_ks else config["top_k"]
    query_header = ["query"]
    for strategy in strategies:
        query_header.extend([
            f"{strategy} rank",
            f"{strategy} cov@{max_k}",
            f"{strategy} top",
        ])
    lines.append("| " + " | ".join(query_header) + " |")
    lines.append("|" + "|".join(["---"] * len(query_header)) + "|")

    for row in per_query:
        cells = [escape_cell(row["query"])]
        for strategy in strategies:
            item = row[strategy]
            top = item["top_note"]
            if item.get("top_lines"):
                top = f"{top}:{item['top_lines']}"
            cells.extend([
                str(item["hit_rank"]),
                format_metric(item.get(f"coverage@{max_k}")),
                escape_cell(top),
            ])
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Changes Vs Baseline")
    lines.append("")
    baseline = config["baseline"]
    for strategy in strategies:
        if strategy == baseline:
            continue

        changed = [
            row for row in per_query
            if row[strategy].get("top_changed_vs_baseline")
            or row[strategy].get("coverage_delta_vs_baseline")
        ]
        lines.append(f"### {strategy}")
        lines.append("")
        if not changed:
            lines.append("No top result or coverage changes.")
            lines.append("")
            continue

        lines.append("| query | coverage delta | top changed | top result |")
        lines.append("|---|---:|---|---|")
        for row in changed:
            item = row[strategy]
            top = item["top_note"]
            if item.get("top_lines"):
                top = f"{top}:{item['top_lines']}"
            lines.append(
                "| "
                + " | ".join([
                    escape_cell(row["query"]),
                    format_metric(item.get("coverage_delta_vs_baseline")),
                    str(item.get("top_changed_vs_baseline")),
                    escape_cell(top),
                ])
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def metric_columns(hit_ks: list[int]) -> list[str]:
    columns = ["mrr"]
    columns.extend(f"hit@{k}" for k in hit_ks)
    columns.extend(f"avg_expected_note_coverage@{k}" for k in hit_ks)
    for k in hit_ks:
        columns.append(f"avg_unique_notes@{k}")
        columns.append(f"avg_max_chunks_per_note@{k}")
    return columns


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
