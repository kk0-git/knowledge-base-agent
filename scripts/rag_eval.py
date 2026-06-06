from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from knowledge_base_agent.config import load_dotenv
from services.rag.bm25 import BM25Index
from services.rag.chunker import HeadingChunker
from services.rag.embedder import create_embedder
from services.rag.manager import RAGManager
from services.rag.memory_vector_store import MemoryVectorStore
from services.rag.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker, DashScopeReranker


DEFAULT_MODEL = "BAAI/bge-m3"


def derive_bm25_index_path(vector_index_path: Path) -> Path:
    return vector_index_path.with_suffix(".bm25.json")


@dataclass(frozen=True)
class EvalCaseResult:
    query: str
    intent: str
    query_type: str
    source_type: str
    expected_notes: list[str]
    retrieved_notes: list[str]
    hit_rank: int | None
    hits: dict[str, bool]
    mrr: float
    diversity: dict[str, Any]
    diagnosis: list[str]
    top_results: list[dict[str, Any]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG search against a small eval set")
    parser.add_argument("--index", default="./rag-index/index.json", help="Index JSON path")
    parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    parser.add_argument("--eval", default="./eval/rag_eval.json", help="Eval JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="local")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5, help="Search top K")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid", "hybrid-rerank"], default="dense")
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-type", choices=["local", "dashscope"], default="local")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument(
        "--hit-ks",
        default="1,3,5",
        help="Comma-separated K values for Hit@K, e.g. 1,3,5",
    )
    parser.add_argument(
        "--show-success",
        action="store_true",
        help="Print successful cases as well as failures",
    )
    parser.add_argument("--out", default=None, help="Optional path to save eval result JSON")
    args = parser.parse_args()

    eval_cases = load_eval_cases(Path(args.eval))
    hit_ks = parse_hit_ks(args.hit_ks, args.top_k)

    manager = build_manager(
        index_path=Path(args.index),
        bm25_index_path=Path(args.bm25_index) if args.bm25_index else None,
        model_name=args.model,
        embedding_provider=args.embedding_provider,
        embed_batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
        mode=args.mode,
        reranker_type=args.reranker_type,
        reranker_model=args.reranker_model,
        rerank_batch_size=args.rerank_batch_size,
        rerank_max_length=args.rerank_max_length,
    )

    results: list[EvalCaseResult] = []
    for case in eval_cases:
        results.append(
            evaluate_case(
                manager,
                case,
                top_k=args.top_k,
                hit_ks=hit_ks,
                mode=args.mode,
                dense_top_k=args.dense_top_k,
                bm25_top_k=args.bm25_top_k,
                rrf_k=args.rrf_k,
                rerank_candidates=args.rerank_candidates,
            )
        )

    print_summary(results, hit_ks)
    print_cases(results, show_success=args.show_success)

    if args.out:
        payload = build_eval_payload(
            args=args,
            eval_cases=eval_cases,
            results=results,
            hit_ks=hit_ks,
        )
        write_eval_payload(Path(args.out), payload)
        print(f"Saved eval result: {Path(args.out).resolve()}")

    return 0


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Eval file must be a JSON list")

    for index, case in enumerate(cases):
        if "query" not in case:
            raise ValueError(f"Eval case #{index} missing query")
        if "expected_notes" not in case:
            raise ValueError(f"Eval case #{index} missing expected_notes")

    return cases


def parse_hit_ks(raw: str, top_k: int) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    return [value for value in values if value > 0 and value <= top_k]


def build_manager(
    index_path: Path,
    bm25_index_path: Path | None,
    model_name: str,
    mode: str,
    embedding_provider: str = "local",
    embed_batch_size: int = 32,
    max_seq_length: int | None = None,
    reranker_type: str = "local",
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    rerank_batch_size: int = 16,
    rerank_max_length: int = 512,
) -> RAGManager:
    resolved_bm25_index_path = bm25_index_path or derive_bm25_index_path(index_path)

    if mode in {"dense", "hybrid", "hybrid-rerank"} and not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if mode in {"bm25", "hybrid", "hybrid-rerank"} and not resolved_bm25_index_path.exists():
        raise FileNotFoundError(f"BM25 index file not found: {resolved_bm25_index_path}")

    load_dotenv(PROJECT_ROOT / ".env")
    embedder = (
        create_embedder(
            provider=embedding_provider,
            model_name=model_name,
            batch_size=embed_batch_size,
            max_seq_length=max_seq_length,
        )
        if mode in {"dense", "hybrid", "hybrid-rerank"}
        else None
    )
    vector_store = MemoryVectorStore(persist_path=index_path if index_path.exists() else None)
    bm25_index = BM25Index(persist_path=resolved_bm25_index_path) if mode in {"bm25", "hybrid", "hybrid-rerank"} else None

    if mode == "hybrid-rerank":
        if reranker_type == "dashscope":
            reranker = DashScopeReranker(model_name=reranker_model)
        else:
            reranker = CrossEncoderReranker(
                model_name=reranker_model,
                batch_size=rerank_batch_size,
                max_length=rerank_max_length,
            )
    else:
        reranker = None

    return RAGManager(
        chunker=HeadingChunker(),
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
    )


def evaluate_case(
    manager: RAGManager,
    case: dict[str, Any],
    top_k: int,
    hit_ks: list[int],
    mode: str = "dense",
    dense_top_k: int = 50,
    bm25_top_k: int = 50,
    rrf_k: int = 60,
    rerank_candidates: int = 50,
) -> EvalCaseResult:
    query = str(case["query"])
    intent = str(case.get("intent", "focused"))
    query_type = str(case.get("query_type", intent))
    source_type = str(case.get("source_type", "unknown"))
    expected_notes = [normalize_note_path(path) for path in case.get("expected_notes", [])]
    expected_set = set(expected_notes)

    if mode == "dense":
        search_results = manager.dense_search(query=query, top_k=top_k)
    elif mode == "bm25":
        search_results = manager.bm25_search(query=query, top_k=top_k)
    elif mode == "hybrid":
        search_results = manager.hybrid_search(
            query=query,
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
        )
    elif mode == "hybrid-rerank":
        search_results = manager.hybrid_rerank_search(
            query=query,
            top_k=top_k,
            rerank_candidates=rerank_candidates,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
        )
    else:
        raise ValueError(f"Unsupported search mode: {mode}")

    retrieved_notes = [normalize_note_path(result.chunk.note_path) for result in search_results]
    hit_rank = first_hit_rank(retrieved_notes, expected_set)

    top_results = []
    for rank, result in enumerate(search_results, start=1):
        chunk = result.chunk
        top_results.append(
            {
                "rank": rank,
                "note_path": chunk.note_path,
                "heading": " > ".join(chunk.heading_path) if chunk.heading_path else "",
                "score": result.score,
                "lines": line_range(chunk.start_line, chunk.end_line),
                "chunk_id": chunk.chunk_id,
                "preview": chunk.text[:160],
            }
        )

    hits = {
        f"hit@{k}": hit_rank is not None and hit_rank <= k
        for k in hit_ks
    }
    mrr = 0.0 if hit_rank is None else 1.0 / hit_rank
    diversity = calculate_diversity_bundle(
        results=top_results,
        expected_notes=expected_notes,
        hit_ks=hit_ks,
    )
    diagnosis = diagnose_query(
        intent=intent,
        hit_at_top_k=hits.get(f"hit@{max(hit_ks)}", False) if hit_ks else hit_rank is not None,
        coverage=diversity.get(f"expected_note_coverage@{max(hit_ks)}", 0.0) if hit_ks else 0.0,
        unique_notes=diversity.get(f"unique_notes@{max(hit_ks)}", 0) if hit_ks else 0,
        max_chunks_per_note=diversity.get(f"max_chunks_per_note@{max(hit_ks)}", 0) if hit_ks else 0,
        k=max(hit_ks) if hit_ks else top_k,
    )

    return EvalCaseResult(
        query=query,
        intent=intent,
        query_type=query_type,
        source_type=source_type,
        expected_notes=expected_notes,
        retrieved_notes=retrieved_notes,
        hit_rank=hit_rank,
        hits=hits,
        mrr=mrr,
        diversity=diversity,
        diagnosis=diagnosis,
        top_results=top_results,
    )


def first_hit_rank(retrieved_notes: list[str], expected_notes: set[str]) -> int | None:
    for index, note_path in enumerate(retrieved_notes, start=1):
        if note_path in expected_notes:
            return index
    return None


def print_summary(results: list[EvalCaseResult], hit_ks: list[int]) -> None:
    total = len(results)
    if total == 0:
        print("No eval cases.")
        return

    print(f"Queries: {total}")

    for k in hit_ks:
        hits = sum(1 for result in results if result.hit_rank is not None and result.hit_rank <= k)
        print(f"Hit@{k}: {hits / total:.3f} ({hits}/{total})")

    reciprocal_ranks = [
        0.0 if result.hit_rank is None else 1.0 / result.hit_rank
        for result in results
    ]
    mrr = sum(reciprocal_ranks) / total
    print(f"MRR: {mrr:.3f}")

    diversity_metrics = compute_diversity_metrics(results, hit_ks)
    for name, value in diversity_metrics.items():
        print(f"{name}: {value:.3f}")

    print_group_summary(results, hit_ks, group_field="query_type")
    print_group_summary(results, hit_ks, group_field="source_type")

    print("")


def compute_metrics(results: list[EvalCaseResult], hit_ks: list[int]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {f"hit@{k}": 0.0 for k in hit_ks} | {"mrr": 0.0}

    metrics: dict[str, float] = {}
    for k in hit_ks:
        hits = sum(1 for result in results if result.hit_rank is not None and result.hit_rank <= k)
        metrics[f"hit@{k}"] = hits / total

    reciprocal_ranks = [
        0.0 if result.hit_rank is None else 1.0 / result.hit_rank
        for result in results
    ]
    metrics["mrr"] = sum(reciprocal_ranks) / total
    metrics.update(compute_diversity_metrics(results, hit_ks))
    return metrics


def compute_group_metrics(
    results: list[EvalCaseResult],
    hit_ks: list[int],
    group_field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvalCaseResult]] = {}
    for result in results:
        key = str(getattr(result, group_field, "unknown") or "unknown")
        grouped.setdefault(key, []).append(result)

    payload: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        group_results = grouped[key]
        payload[key] = {
            "query_count": len(group_results),
            "failure_count": sum(1 for result in group_results if result.hit_rank is None),
            "metrics": compute_metrics(group_results, hit_ks),
        }
    return payload


def print_group_summary(
    results: list[EvalCaseResult],
    hit_ks: list[int],
    group_field: str,
) -> None:
    groups = compute_group_metrics(results, hit_ks, group_field)
    if not groups:
        return

    print(f"{group_field}:")
    for group_name, payload in groups.items():
        metrics = payload["metrics"]
        hit1 = metrics.get("hit@1", 0.0)
        mrr = metrics.get("mrr", 0.0)
        print(
            f"  {group_name}: "
            f"n={payload['query_count']}, "
            f"fail={payload['failure_count']}, "
            f"hit@1={hit1:.3f}, "
            f"mrr={mrr:.3f}"
        )


def compute_diversity_metrics(
    results: list[EvalCaseResult],
    hit_ks: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for k in hit_ks:
        for metric_name in [
            f"unique_notes@{k}",
            f"max_chunks_per_note@{k}",
            f"note_entropy@{k}",
            f"expected_note_coverage@{k}",
        ]:
            metrics[f"avg_{metric_name}"] = average_diversity_metric(results, metric_name)

    return metrics


def average_diversity_metric(results: list[EvalCaseResult], metric_name: str) -> float:
    if not results:
        return 0.0

    values = [float(result.diversity.get(metric_name, 0.0)) for result in results]
    return sum(values) / len(values)


def build_eval_payload(
    args: argparse.Namespace,
    eval_cases: list[dict[str, Any]],
    results: list[EvalCaseResult],
    hit_ks: list[int],
) -> dict[str, Any]:
    return {
        "config": {
            "index": args.index,
            "bm25_index": args.bm25_index,
            "eval": args.eval,
            "model": args.model,
            "mode": args.mode,
            "top_k": args.top_k,
            "hit_ks": hit_ks,
            "dense_top_k": args.dense_top_k,
            "bm25_top_k": args.bm25_top_k,
            "rrf_k": args.rrf_k,
            "reranker_type": args.reranker_type,
            "reranker_model": args.reranker_model,
            "rerank_candidates": args.rerank_candidates,
            "rerank_batch_size": args.rerank_batch_size,
            "rerank_max_length": args.rerank_max_length,
        },
        "metrics": compute_metrics(results, hit_ks),
        "group_metrics": {
            "query_type": compute_group_metrics(results, hit_ks, "query_type"),
            "source_type": compute_group_metrics(results, hit_ks, "source_type"),
        },
        "summary": {
            "query_count": len(results),
            "failure_count": sum(1 for result in results if result.hit_rank is None),
            "diagnosis_counts": count_diagnoses(results),
        },
        "cases": [asdict(result) for result in results],
        "eval_cases": eval_cases,
    }


def count_diagnoses(results: list[EvalCaseResult]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        counter.update(result.diagnosis)
    return dict(sorted(counter.items()))


def write_eval_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_cases(results: list[EvalCaseResult], show_success: bool) -> None:
    failures = [result for result in results if result.hit_rank is None]
    successes = [result for result in results if result.hit_rank is not None]

    if failures:
        print("Failures")
        print("--------")
        for result in failures:
            print_case(result)
    else:
        print("Failures: none")
        print("")

    if show_success:
        print("Successes")
        print("---------")
        for result in successes:
            print_case(result)


def print_case(result: EvalCaseResult) -> None:
    print(f"Query: {result.query}")
    print(f"Intent: {result.intent}")
    print(f"Expected: {', '.join(result.expected_notes)}")
    if result.hit_rank is None:
        print("Hit rank: none")
    else:
        print(f"Hit rank: {result.hit_rank}")
    if result.diagnosis:
        print(f"Diagnosis: {', '.join(result.diagnosis)}")
    print(f"Diversity: {json.dumps(result.diversity, ensure_ascii=False)}")

    print("Top results:")
    for item in result.top_results:
        heading = f" > {item['heading']}" if item["heading"] else ""
        lines = f" lines:{item['lines']}" if item["lines"] else ""
        print(
            f"  {item['rank']}. {item['note_path']}{heading} "
            f"score:{item['score']}{lines}"
        )
        print(f"     {item['preview']}")
    print("")


def normalize_note_path(path: str) -> str:
    return path.replace("\\", "/")


def line_range(start_line: int | None, end_line: int | None) -> str:
    if start_line is None or end_line is None:
        return ""
    return f"{start_line}-{end_line}"


def calculate_diversity_bundle(
    results: list[dict[str, Any]],
    expected_notes: list[str],
    hit_ks: list[int],
) -> dict[str, Any]:
    diversity: dict[str, Any] = {}

    for k in hit_ks:
        diversity.update(calculate_note_diversity(results, k))
        diversity.update(calculate_expected_note_coverage(results, expected_notes, k))

    return diversity


def calculate_note_diversity(results: list[dict[str, Any]], k: int) -> dict[str, Any]:
    top_results = results[:k]
    note_counts = Counter(result["note_path"] for result in top_results)

    unique_notes = len(note_counts)
    max_chunks_per_note = max(note_counts.values(), default=0)
    total = len(top_results)

    entropy = 0.0
    if total > 0:
        for count in note_counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)

    max_entropy = math.log(unique_notes) if unique_notes > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        f"unique_notes@{k}": unique_notes,
        f"max_chunks_per_note@{k}": max_chunks_per_note,
        f"note_entropy@{k}": round(normalized_entropy, 4),
    }


def calculate_expected_note_coverage(
    results: list[dict[str, Any]],
    expected_notes: list[str],
    k: int,
) -> dict[str, Any]:
    expected = set(expected_notes)
    if not expected:
        return {
            f"expected_note_coverage@{k}": 0.0,
            f"covered_expected_notes@{k}": [],
            f"missing_expected_notes@{k}": [],
        }

    top_notes = {normalize_note_path(result["note_path"]) for result in results[:k]}
    covered = top_notes & expected

    return {
        f"expected_note_coverage@{k}": round(len(covered) / len(expected), 4),
        f"covered_expected_notes@{k}": sorted(covered),
        f"missing_expected_notes@{k}": sorted(expected - covered),
    }


def diagnose_query(
    intent: str,
    hit_at_top_k: bool,
    coverage: float,
    unique_notes: int,
    max_chunks_per_note: int,
    k: int,
) -> list[str]:
    labels: list[str] = []

    if not hit_at_top_k:
        labels.append("missed_expected_note")

    if intent == "focused":
        if unique_notes >= 4 and not hit_at_top_k:
            labels.append("too_scattered_for_focused_query")
        return labels

    if intent in {"comparative", "exploratory"}:
        if unique_notes <= 1 and k >= 3:
            labels.append("single_note_dominance")

        if max_chunks_per_note >= max(3, k - 1):
            labels.append("one_note_fills_topk")

        if coverage < 0.5:
            labels.append("insufficient_expected_coverage")

    return labels


if __name__ == "__main__":
    raise SystemExit(main())
