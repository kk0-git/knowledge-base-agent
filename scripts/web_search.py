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

from web.app import create_app


DEFAULT_MODEL = "BAAI/bge-m3"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Knowledge Agent search web UI")
    parser.add_argument("--index", default="./rag-index/bge-m3-v2.json", help="Vector index JSON path")
    parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="local")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
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
        embedding_provider=args.embedding_provider,
        embed_batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
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
