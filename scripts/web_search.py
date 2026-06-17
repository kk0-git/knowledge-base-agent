from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import uvicorn

from services.rag.vector_store_loader import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
)
from web.app import create_app


DEFAULT_MODEL = "BAAI/bge-m3"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Knowledge Agent search web UI")
    parser.add_argument("--index", default="./rag-index/bge-m3-v2.json", help="Vector index JSON path")
    parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    parser.add_argument("--vault", default=None, help="Vault root path for chat rg search")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="local")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-chunk-chars", type=int, default=1500)
    parser.add_argument("--target-chunk-chars", type=int, default=900)
    parser.add_argument("--min-chunk-chars", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--chunk-split-mode", choices=["streaming", "indexed"], default="indexed")
    parser.add_argument("--strip-code-blocks", action="store_true")
    parser.add_argument("--vector-index", choices=["flat", "hnsw"], default="flat")
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW_M)
    parser.add_argument("--hnsw-ef-construction", type=int, default=DEFAULT_HNSW_EF_CONSTRUCTION)
    parser.add_argument("--hnsw-ef-search", type=int, default=DEFAULT_HNSW_EF_SEARCH)
    parser.add_argument("--wiki-state", default=None, help="Wiki state JSON path for the /wiki page")
    parser.add_argument("--wiki-dir", default=None, help="Wiki output directory for the /wiki page")
    parser.add_argument("--wiki-min-notes-per-tag", type=int, default=2)
    parser.add_argument("--wiki-overview-note-threshold", type=int, default=12)
    parser.add_argument(
        "--sync-on-start",
        action="store_true",
        help="Run one background workspace sync after the web server starts",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    index_path = Path(args.index)
    bm25_index_path = Path(args.bm25_index) if args.bm25_index else None

    app = create_app(
        index_path=index_path,
        bm25_index_path=bm25_index_path,
        model_name=args.model,
        project_root=PROJECT_ROOT,
        vault_path=Path(args.vault) if args.vault else None,
        embedding_provider=args.embedding_provider,
        embed_batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
        max_chunk_chars=args.max_chunk_chars,
        target_chunk_chars=args.target_chunk_chars,
        min_chunk_chars=args.min_chunk_chars,
        chunk_overlap=args.chunk_overlap,
        chunk_split_mode=args.chunk_split_mode,
        strip_code_blocks=args.strip_code_blocks,
        vector_index=args.vector_index,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        wiki_state_path=Path(args.wiki_state) if args.wiki_state else None,
        wiki_dir=Path(args.wiki_dir) if args.wiki_dir else None,
        wiki_min_notes_per_tag=args.wiki_min_notes_per_tag,
        wiki_overview_note_threshold=args.wiki_overview_note_threshold,
        sync_on_start=args.sync_on_start,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
