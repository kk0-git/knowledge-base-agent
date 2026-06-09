from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rag_eval
from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.rag.answerer import RAGAnswerer
from services.rag.context_packer import (
    PackedContext,
    pack_search_results,
    packed_context_to_dict,
    score_type_for_mode,
)
from services.rag.schema import SearchResult
from services.rag.vector_store_loader import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
)


DEFAULT_MODEL = "BAAI/bge-m3"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-shot RAG answer generation debug flow")
    parser.add_argument("--index", default="./rag-index/index.json", help="Vector index JSON path")
    parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="local")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--vector-index", choices=["flat", "hnsw"], default="flat")
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW_M)
    parser.add_argument("--hnsw-ef-construction", type=int, default=DEFAULT_HNSW_EF_CONSTRUCTION)
    parser.add_argument("--hnsw-ef-search", type=int, default=DEFAULT_HNSW_EF_SEARCH)
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid", "hybrid-rerank"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-type", choices=["local", "dashscope"], default="dashscope")
    parser.add_argument("--reranker-model", default="qwen3-rerank")
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1200)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--out", default="./eval-results/rag-answer-debug")
    args = parser.parse_args()

    started_at = time.perf_counter()
    manager = rag_eval.build_manager(
        index_path=Path(args.index),
        bm25_index_path=Path(args.bm25_index) if args.bm25_index else None,
        model_name=args.model,
        mode=args.mode,
        embedding_provider=args.embedding_provider,
        embed_batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
        reranker_type=args.reranker_type,
        reranker_model=args.reranker_model,
        rerank_batch_size=args.rerank_batch_size,
        rerank_max_length=args.rerank_max_length,
        vector_index=args.vector_index,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )
    retrieval_started_at = time.perf_counter()
    results = retrieve(args, manager)
    retrieval_ms = int((time.perf_counter() - retrieval_started_at) * 1000)

    context = pack_search_results(
        results,
        score_type=score_type_for_mode(args.mode),
        max_chunks=args.max_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
        max_context_chars=args.max_context_chars,
    )

    llm_config = load_llm_config(PROJECT_ROOT)
    answerer = RAGAnswerer(
        client=create_llm_client(llm_config),
        model=llm_config.model,
        temperature=llm_config.temperature,
    )
    answer_started_at = time.perf_counter()
    answer = answerer.answer(query=args.query, context=context)
    answer_ms = int((time.perf_counter() - answer_started_at) * 1000)

    payload = build_payload(
        args=args,
        results=results,
        context=context,
        answer=answer,
        retrieval_ms=retrieval_ms,
        answer_ms=answer_ms,
        total_ms=int((time.perf_counter() - started_at) * 1000),
    )
    write_outputs(Path(args.out), payload)
    print(answer.answer)
    print()
    print(f"Saved: {Path(args.out).with_suffix('.json').resolve()}")
    return 0


def retrieve(args: argparse.Namespace, manager) -> list[SearchResult]:
    if args.mode == "dense":
        return manager.dense_search(query=args.query, top_k=args.top_k)
    if args.mode == "bm25":
        return manager.bm25_search(query=args.query, top_k=args.top_k)
    if args.mode == "hybrid":
        return manager.hybrid_search(
            query=args.query,
            top_k=args.top_k,
            dense_top_k=args.dense_top_k,
            bm25_top_k=args.bm25_top_k,
            rrf_k=args.rrf_k,
        )
    if args.mode == "hybrid-rerank":
        return manager.hybrid_rerank_search(
            query=args.query,
            top_k=args.top_k,
            rerank_candidates=args.rerank_candidates,
            dense_top_k=args.dense_top_k,
            bm25_top_k=args.bm25_top_k,
            rrf_k=args.rrf_k,
        )
    raise ValueError(f"Unsupported mode: {args.mode}")


def build_payload(
    *,
    args: argparse.Namespace,
    results: list[SearchResult],
    context: PackedContext,
    answer,
    retrieval_ms: int,
    answer_ms: int,
    total_ms: int,
) -> dict[str, Any]:
    return {
        "config": {
            "index": args.index,
            "bm25_index": args.bm25_index,
            "query": args.query,
            "mode": args.mode,
            "embedding_model": args.model,
            "embedding_provider": args.embedding_provider,
            "vector_index": args.vector_index,
            "top_k": args.top_k,
            "dense_top_k": args.dense_top_k,
            "bm25_top_k": args.bm25_top_k,
            "rrf_k": args.rrf_k,
            "reranker_type": args.reranker_type if args.mode == "hybrid-rerank" else None,
            "reranker_model": args.reranker_model if args.mode == "hybrid-rerank" else None,
            "max_chunks": args.max_chunks,
            "max_chars_per_chunk": args.max_chars_per_chunk,
            "max_context_chars": args.max_context_chars,
        },
        "timing": {
            "retrieval_ms": retrieval_ms,
            "answer_ms": answer_ms,
            "total_ms": total_ms,
        },
        "answer": asdict(answer),
        "packed_context": packed_context_to_dict(context),
        "retrieval_results": [search_result_to_debug_dict(result, rank) for rank, result in enumerate(results, start=1)],
    }


def search_result_to_debug_dict(result: SearchResult, rank: int) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "rank": rank,
        "note_path": chunk.note_path,
        "heading": " > ".join(chunk.heading_path) if chunk.heading_path else "",
        "lines": line_range(chunk.start_line, chunk.end_line),
        "score": result.score,
        "chunk_id": chunk.chunk_id,
        "preview": chunk.text[:300],
    }


def write_outputs(out_base: Path, payload: dict[str, Any]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base if out_base.suffix == ".json" else out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# RAG Answer Debug")
    lines.append("")
    lines.append("## Query")
    lines.append("")
    lines.append(payload["config"]["query"])
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    lines.append(payload["answer"]["answer"])
    lines.append("")
    lines.append("## Context")
    lines.append("")
    for chunk in payload["packed_context"]["chunks"]:
        lines.append(f"### [{chunk['citation_id']}] {chunk['note_path']}:{chunk['lines']}")
        lines.append("")
        lines.append(f"- heading: `{chunk['heading'] or '(no heading)'}`")
        lines.append(f"- score: `{chunk['search_score']}` (`{chunk['score_type']}`)")
        lines.append("")
        lines.append(chunk["text"])
        lines.append("")
    lines.append("## Retrieval Results")
    lines.append("")
    lines.append("| rank | note_path | heading | lines | score |")
    lines.append("|---:|---|---|---|---:|")
    for result in payload["retrieval_results"]:
        lines.append(
            f"| {result['rank']} | {result['note_path']} | {result['heading'] or '(no heading)'} | {result['lines']} | {result['score']} |"
        )
    lines.append("")
    lines.append("## Timing")
    lines.append("")
    for key, value in payload["timing"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def line_range(start_line: int | None, end_line: int | None) -> str:
    if start_line is None or end_line is None:
        return ""
    return f"{start_line}-{end_line}"


if __name__ == "__main__":
    raise SystemExit(main())
