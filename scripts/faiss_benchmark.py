from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from knowledge_base_agent.config import load_dotenv
from services.rag.embedder import create_embedder
from services.rag.faiss_vector_store import FaissIndexConfig, FaissVectorStore
from services.rag.schema import EmbeddingChunk, embedded_chunk_from_dict


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_INDEX_TYPES = "flat,hnsw,ivf_flat,ivf_pq"


@dataclass(frozen=True)
class QueryBenchmark:
    query: str
    ground_truth: list[str]
    results: list[str]
    latency_ms: float
    recall: dict[str, float]


@dataclass(frozen=True)
class IndexBenchmark:
    index_type: str
    config: dict[str, Any]
    build_time_ms: float
    index_size_bytes: int
    latency: dict[str, float]
    recall: dict[str, float]
    queries: list[QueryBenchmark]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark FAISS dense vector indexes against FlatIP ground truth."
    )
    parser.add_argument("--index", required=True, help="Existing rag-index JSON path")
    parser.add_argument("--eval", default="./eval/rag_eval.json", help="Eval JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model for queries")
    parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="local")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--recall-ks", default="1,5,10,20")
    parser.add_argument("--index-types", default=DEFAULT_INDEX_TYPES)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=80)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--repeat-search", type=int, default=3)
    parser.add_argument("--out", default="./eval-results/faiss-benchmark")
    parser.add_argument("--env-file", default="./.env")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / args.env_file)

    chunks = load_embedding_chunks(Path(args.index))
    queries = load_queries(Path(args.eval))
    recall_ks = parse_int_list(args.recall_ks)
    index_types = parse_index_types(args.index_types)
    validate_recall_ks(recall_ks, args.top_k)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loaded queries: {len(queries)}")
    print(f"Index types: {', '.join(index_types)}")

    embedder = create_embedder(
        provider=args.embedding_provider,
        model_name=args.model,
        batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
    )
    query_embeddings = embedder.embed_texts(queries)

    flat_store, flat_build_time_ms = build_store(
        chunks=chunks,
        index_type="flat",
        args=args,
    )
    ground_truth = search_all(
        store=flat_store,
        queries=queries,
        query_embeddings=query_embeddings,
        top_k=args.top_k,
        repeat_search=args.repeat_search,
    )

    benchmarks: list[IndexBenchmark] = []
    for index_type in index_types:
        if index_type == "flat":
            store = flat_store
            build_time_ms = flat_build_time_ms
        else:
            store, build_time_ms = build_store(
                chunks=chunks,
                index_type=index_type,
                args=args,
            )

        print(f"Benchmarking {index_type} ...")
        query_results = search_all(
            store=store,
            queries=queries,
            query_embeddings=query_embeddings,
            top_k=args.top_k,
            repeat_search=args.repeat_search,
        )
        benchmarks.append(
            build_index_benchmark(
                index_type=index_type,
                store=store,
                config=build_index_config(index_type, args),
                build_time_ms=build_time_ms,
                ground_truth=ground_truth,
                query_results=query_results,
                recall_ks=recall_ks,
            )
        )

    payload = {
        "config": {
            "index": str(Path(args.index)),
            "eval": str(Path(args.eval)),
            "model": args.model,
            "embedding_provider": args.embedding_provider,
            "top_k": args.top_k,
            "recall_ks": recall_ks,
            "repeat_search": args.repeat_search,
            "chunk_count": len(chunks),
            "query_count": len(queries),
        },
        "benchmarks": [benchmark_to_dict(item) for item in benchmarks],
    }

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base if out_base.suffix == ".json" else out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved Markdown: {md_path.resolve()}")
    print_summary(benchmarks, recall_ks)
    return 0


def load_embedding_chunks(path: Path) -> list[EmbeddingChunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise ValueError(f"No chunks found in index: {path}")
    return [embedded_chunk_from_dict(item) for item in chunks]


def load_queries(path: Path) -> list[str]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Eval file must be a JSON list")
    queries = [str(item["query"]) for item in cases if item.get("query")]
    if not queries:
        raise ValueError(f"No queries found in eval file: {path}")
    return queries


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return sorted(set(values))


def parse_index_types(raw: str) -> list[str]:
    allowed = {"flat", "hnsw", "ivf_flat", "ivf_pq"}
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(f"Unsupported index types: {unknown}")
    if "flat" not in values:
        values.insert(0, "flat")
    return values


def validate_recall_ks(recall_ks: list[int], top_k: int) -> None:
    invalid = [value for value in recall_ks if value > top_k]
    if invalid:
        raise ValueError(f"recall-ks must be <= top-k. Invalid: {invalid}")


def build_store(
    *,
    chunks: list[EmbeddingChunk],
    index_type: str,
    args: argparse.Namespace,
) -> tuple[FaissVectorStore, float]:
    config = FaissIndexConfig(
        index_type=index_type,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        nlist=args.nlist,
        nprobe=args.nprobe,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits,
    )
    store = FaissVectorStore(config)
    start = time.perf_counter()
    store.upsert(chunks)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return store, elapsed_ms


def search_all(
    *,
    store: FaissVectorStore,
    queries: list[str],
    query_embeddings: list[list[float]],
    top_k: int,
    repeat_search: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repeats = max(1, repeat_search)
    for query, embedding in zip(queries, query_embeddings):
        latencies: list[float] = []
        last_results = []
        for _ in range(repeats):
            start = time.perf_counter()
            last_results = store.search(embedding, top_k=top_k)
            latencies.append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "query": query,
                "latency_ms": statistics.mean(latencies),
                "results": [result.chunk.chunk_id for result in last_results],
                "scores": [result.score for result in last_results],
            }
        )
    return rows


def build_index_benchmark(
    *,
    index_type: str,
    store: FaissVectorStore,
    config: dict[str, Any],
    build_time_ms: float,
    ground_truth: list[dict[str, Any]],
    query_results: list[dict[str, Any]],
    recall_ks: list[int],
) -> IndexBenchmark:
    query_benchmarks: list[QueryBenchmark] = []
    for truth, result in zip(ground_truth, query_results):
        recall = {
            f"recall@{k}": recall_at_k(
                expected=truth["results"],
                actual=result["results"],
                k=k,
            )
            for k in recall_ks
        }
        query_benchmarks.append(
            QueryBenchmark(
                query=result["query"],
                ground_truth=truth["results"],
                results=result["results"],
                latency_ms=result["latency_ms"],
                recall=recall,
            )
        )

    latencies = [item.latency_ms for item in query_benchmarks]
    recall_summary = {
        f"recall@{k}": statistics.mean(item.recall[f"recall@{k}"] for item in query_benchmarks)
        for k in recall_ks
    }
    return IndexBenchmark(
        index_type=index_type,
        config=config,
        build_time_ms=build_time_ms,
        index_size_bytes=store.serialized_size_bytes(),
        latency=summarize_latency(latencies),
        recall=recall_summary,
        queries=query_benchmarks,
    )


def recall_at_k(*, expected: list[str], actual: list[str], k: int) -> float:
    expected_set = set(expected[:k])
    actual_set = set(actual[:k])
    if not expected_set:
        return 0.0
    return len(expected_set & actual_set) / len(expected_set)


def summarize_latency(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "avg_ms": round(statistics.mean(ordered), 4),
        "p50_ms": round(percentile(ordered, 0.50), 4),
        "p95_ms": round(percentile(ordered, 0.95), 4),
        "max_ms": round(max(ordered), 4),
    }


def percentile(ordered_values: list[float], ratio: float) -> float:
    if not ordered_values:
        return 0.0
    index = min(len(ordered_values) - 1, max(0, int(round((len(ordered_values) - 1) * ratio))))
    return ordered_values[index]


def build_index_config(index_type: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "index_type": index_type,
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_construction": args.hnsw_ef_construction,
        "hnsw_ef_search": args.hnsw_ef_search,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "pq_m": args.pq_m,
        "pq_nbits": args.pq_nbits,
    }


def benchmark_to_dict(item: IndexBenchmark) -> dict[str, Any]:
    payload = asdict(item)
    payload["queries"] = [asdict(query) for query in item.queries]
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    config = payload["config"]
    lines.append("# FAISS Benchmark")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    for key, value in config.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| index_type | build_ms | index_size_mb | avg_ms | p50_ms | p95_ms | recall@1 | recall@5 | recall@10 | recall@20 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for benchmark in payload["benchmarks"]:
        recall = benchmark["recall"]
        latency = benchmark["latency"]
        lines.append(
            "| {index_type} | {build:.2f} | {size:.2f} | {avg:.4f} | {p50:.4f} | {p95:.4f} | {r1:.4f} | {r5:.4f} | {r10:.4f} | {r20:.4f} |".format(
                index_type=benchmark["index_type"],
                build=benchmark["build_time_ms"],
                size=benchmark["index_size_bytes"] / 1024 / 1024,
                avg=latency["avg_ms"],
                p50=latency["p50_ms"],
                p95=latency["p95_ms"],
                r1=recall.get("recall@1", 0.0),
                r5=recall.get("recall@5", 0.0),
                r10=recall.get("recall@10", 0.0),
                r20=recall.get("recall@20", 0.0),
            )
        )
    lines.append("")
    lines.append("## Low Recall Queries")
    lines.append("")
    for benchmark in payload["benchmarks"]:
        if benchmark["index_type"] == "flat":
            continue
        lines.append(f"### {benchmark['index_type']}")
        low_rows = [
            query
            for query in benchmark["queries"]
            if query["recall"].get("recall@20", 1.0) < 1.0
        ]
        if not low_rows:
            lines.append("")
            lines.append("No recall@20 loss against Flat.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| query | recall@1 | recall@5 | recall@10 | recall@20 |")
        lines.append("|---|---:|---:|---:|---:|")
        for query in low_rows:
            recall = query["recall"]
            lines.append(
                f"| {query['query']} | {recall.get('recall@1', 0.0):.4f} | {recall.get('recall@5', 0.0):.4f} | {recall.get('recall@10', 0.0):.4f} | {recall.get('recall@20', 0.0):.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def print_summary(benchmarks: list[IndexBenchmark], recall_ks: list[int]) -> None:
    print()
    print("index_type build_ms size_mb avg_ms p95_ms " + " ".join(f"recall@{k}" for k in recall_ks))
    for item in benchmarks:
        recall_values = " ".join(f"{item.recall[f'recall@{k}']:.4f}" for k in recall_ks)
        print(
            f"{item.index_type} "
            f"{item.build_time_ms:.2f} "
            f"{item.index_size_bytes / 1024 / 1024:.2f} "
            f"{item.latency['avg_ms']:.4f} "
            f"{item.latency['p95_ms']:.4f} "
            f"{recall_values}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
