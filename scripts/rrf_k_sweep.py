"""RRF k 值网格搜索。

对不同的 k 值跑 hybrid 检索，对比 MRR 和 Hit@1，输出曲线数据。
用法：uv run python scripts\rrf_k_sweep.py --index ./rag-index/xxx.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_SRC = SCRIPTS_DIR.parent / "src"
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from knowledge_base_agent.config import load_dotenv
from services.rag.bm25 import BM25Index
from services.rag.chunker import HeadingChunker
from services.rag.embedder import create_embedder
from services.rag.manager import RAGManager
from services.rag.memory_vector_store import MemoryVectorStore

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_K_VALUES = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120]


def derive_bm25_index_path(vector_index_path: Path) -> Path:
    return vector_index_path.with_suffix(".bm25.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="RRF k-value grid search")
    parser.add_argument("--index", default="./rag-index/index.json")
    parser.add_argument("--bm25-index", default=None)
    parser.add_argument("--eval", default="./eval/rag_eval.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="local")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--k-values", default="0,10,20,30,40,50,60,80,100,120")
    parser.add_argument("--out", default=None, help="Save sweep result JSON")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    index_path = Path(args.index)
    bm25_index_path = Path(args.bm25_index) if args.bm25_index else derive_bm25_index_path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    if not bm25_index_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_index_path}")

    eval_path = Path(args.eval)
    eval_cases = json.loads(eval_path.read_text(encoding="utf-8"))

    embedder = create_embedder(
        provider=args.embedding_provider,
        model_name=args.model,
        batch_size=args.embed_batch_size,
    )
    vector_store = MemoryVectorStore(persist_path=index_path)
    bm25_index = BM25Index(persist_path=bm25_index_path)

    manager = RAGManager(
        chunker=HeadingChunker(),
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    print(f"Eval cases: {len(eval_cases)}")
    print(f"k values: {k_values}")
    print(f"{'k':>6}  {'Hit@1':>7}  {'Hit@3':>7}  {'Hit@5':>7}  {'MRR':>7}")
    print("-" * 42)

    all_results: list[dict] = []

    for k in k_values:
        hit_counts = {1: 0, 3: 0, 5: 0}
        mrr_total = 0.0

        for case in eval_cases:
            query = case["query"]
            expected = set(case.get("expected_notes", []))
            results = manager.hybrid_search(
                query=query,
                top_k=args.top_k,
                dense_top_k=args.dense_top_k,
                bm25_top_k=args.bm25_top_k,
                rrf_k=k,
            )

            hit_rank: int | None = None
            for rank, result in enumerate(results, start=1):
                if result.chunk.note_path in expected:
                    hit_rank = rank
                    break

            if hit_rank is not None:
                mrr_total += 1.0 / hit_rank
                for ks in hit_counts:
                    if hit_rank <= ks:
                        hit_counts[ks] += 1

        n = len(eval_cases)
        mrr = mrr_total / n if n else 0.0
        hits = {ks: count / n for ks, count in hit_counts.items()}

        print(f"{k:>6}  {hits[1]:>7.4f}  {hits[3]:>7.4f}  {hits[5]:>7.4f}  {mrr:>7.4f}")

        all_results.append({
            "k": k,
            "hit@1": round(hits[1], 6),
            "hit@3": round(hits[3], 6),
            "hit@5": round(hits[5], 6),
            "mrr": round(mrr, 6),
        })

    if args.out:
        payload = {
            "config": {
                "index": str(index_path),
                "eval": str(eval_path),
                "top_k": args.top_k,
                "dense_top_k": args.dense_top_k,
                "bm25_top_k": args.bm25_top_k,
            },
            "results": all_results,
        }
        out_path = Path(args.out)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
