from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_SRC = SCRIPTS_DIR.parent / "src"
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from knowledge_base_agent.config import load_dotenv, load_exclusion_patterns
from knowledge_base_agent.scanner import ExclusionFilter, scan_vault
from services.rag.bm25 import BM25Index
from services.rag.chunker import ChunkerConfig, HeadingChunker
from services.rag.embedder import build_chunk_embedding_text, create_embedder
from services.rag.incremental import (
    build_current_file_states,
    build_index_config,
    index_config_matches,
    metadata_chunk_ids,
    plan_incremental_update,
)
from services.rag.manager import RAGManager
from services.rag.memory_vector_store import MemoryVectorStore
from services.rag.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker, DashScopeReranker
from services.rag.schema import EmbeddingChunk, TextChunk


DEFAULT_MODEL = "BAAI/bge-m3"


def derive_bm25_index_path(vector_index_path: Path) -> Path:
    return vector_index_path.with_suffix(".bm25.json")


def add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "openai_compatible"],
        default="local",
        help="Embedding provider: local SentenceTransformer or OpenAI-compatible /v1/embeddings API",
    )
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)


def create_index_embedder(args: argparse.Namespace):
    load_dotenv(PROJECT_ROOT / ".env")
    return create_embedder(
        provider=args.embedding_provider,
        model_name=args.model,
        batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
    )


def create_reranker(args: argparse.Namespace) -> CrossEncoderReranker | DashScopeReranker:
    if args.reranker_type == "dashscope":
        return DashScopeReranker(model_name=args.reranker_model)
    return CrossEncoderReranker(
        model_name=args.reranker_model,
        batch_size=args.rerank_batch_size,
        max_length=args.rerank_max_length,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporary RAG debug script")
    subcommands = parser.add_subparsers(dest="command", required=True)

    index_parser = subcommands.add_parser("index", help="Build chunk embedding index")
    index_parser.add_argument("--vault", required=True, help="Path to Obsidian vault")
    index_parser.add_argument("--index", default="./rag-index/index.json", help="Index JSON path")
    index_parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    index_parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    add_embedding_args(index_parser)
    index_parser.add_argument("--max-chunk-chars", type=int, default=1500)
    index_parser.add_argument("--target-chunk-chars", type=int, default=900)
    index_parser.add_argument("--min-chunk-chars", type=int, default=200)
    index_parser.add_argument("--chunk-overlap", type=int, default=200)
    index_parser.add_argument(
        "--strip-code-blocks",
        action="store_true",
        help="Compatibility flag; current chunker normally treats code blocks as ordinary text",
    )
    index_parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Delete the existing index file before indexing this vault",
    )
    index_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only embed added or modified markdown files when index metadata matches",
    )

    search_parser = subcommands.add_parser("search", help="Search chunk embedding index")
    search_parser.add_argument("--index", default="./rag-index/index.json", help="Index JSON path")
    search_parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    add_embedding_args(search_parser)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--mode", choices=["dense", "bm25", "hybrid", "hybrid-rerank"], default="dense")
    search_parser.add_argument("--dense-top-k", type=int, default=50)
    search_parser.add_argument("--bm25-top-k", type=int, default=50)
    search_parser.add_argument("--rrf-k", type=int, default=60)
    search_parser.add_argument("--reranker-type", choices=["local", "dashscope"], default="local")
    search_parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    search_parser.add_argument("--rerank-candidates", type=int, default=50)
    search_parser.add_argument("--rerank-batch-size", type=int, default=16)
    search_parser.add_argument("--rerank-max-length", type=int, default=512)

    index_eval_parser = subcommands.add_parser("index-and-eval", help="Index vault then run eval (single model load)")
    index_eval_parser.add_argument("--vault", required=True, help="Path to Obsidian vault")
    index_eval_parser.add_argument("--index", default="./rag-index/index.json", help="Index JSON path")
    index_eval_parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    index_eval_parser.add_argument("--eval", default="./eval/rag_eval.json", help="Eval JSON path")
    index_eval_parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    add_embedding_args(index_eval_parser)
    index_eval_parser.add_argument("--max-chunk-chars", type=int, default=1500)
    index_eval_parser.add_argument("--target-chunk-chars", type=int, default=900)
    index_eval_parser.add_argument("--min-chunk-chars", type=int, default=200)
    index_eval_parser.add_argument("--chunk-overlap", type=int, default=200)
    index_eval_parser.add_argument(
        "--strip-code-blocks",
        action="store_true",
        help="Compatibility flag; current chunker normally treats code blocks as ordinary text",
    )
    index_eval_parser.add_argument("--reset-index", action="store_true")
    index_eval_parser.add_argument("--top-k", type=int, default=5)
    index_eval_parser.add_argument("--hit-ks", default="1,3,5")
    index_eval_parser.add_argument("--mode", choices=["dense", "bm25", "hybrid", "hybrid-rerank"], default="dense")
    index_eval_parser.add_argument("--dense-top-k", type=int, default=50)
    index_eval_parser.add_argument("--bm25-top-k", type=int, default=50)
    index_eval_parser.add_argument("--rrf-k", type=int, default=60)
    index_eval_parser.add_argument("--reranker-type", choices=["local", "dashscope"], default="local")
    index_eval_parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    index_eval_parser.add_argument("--rerank-candidates", type=int, default=50)
    index_eval_parser.add_argument("--rerank-batch-size", type=int, default=16)
    index_eval_parser.add_argument("--rerank-max-length", type=int, default=512)
    index_eval_parser.add_argument("--out", default=None, help="Path to save eval result JSON")

    args = parser.parse_args()

    if args.command == "index":
        return run_index(args)

    if args.command == "search":
        return run_search(args)

    if args.command == "index-and-eval":
        return run_index_and_eval(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def run_index(args: argparse.Namespace) -> int:
    vault_path = Path(args.vault)
    index_path = Path(args.index)
    bm25_index_path = Path(args.bm25_index) if args.bm25_index else derive_bm25_index_path(index_path)

    if args.reset_index and index_path.exists():
        index_path.unlink()
        print(f"Reset index: {index_path.resolve()}")
    if args.reset_index and bm25_index_path.exists():
        bm25_index_path.unlink()
        print(f"Reset BM25 index: {bm25_index_path.resolve()}")

    exclusions = load_exclusion_patterns(vault_path)
    exclusion_filter = ExclusionFilter(exclusions)
    scan_result = scan_vault(vault_path, exclusion_filter)

    markdown_files = [note.path for note in scan_result.notes]

    print(f"Vault: {scan_result.vault_path}")
    print(f"Markdown files: {len(markdown_files)}")
    print(f"Excluded markdown files: {scan_result.excluded_count}")
    print(f"Failed files: {len(scan_result.failed)}")

    chunker_config = ChunkerConfig(
        max_chunk_chars=args.max_chunk_chars,
        target_chunk_chars=args.target_chunk_chars,
        min_chunk_chars=args.min_chunk_chars,
        chunk_overlap=args.chunk_overlap,
        strip_code_blocks=args.strip_code_blocks,
    )
    index_config = build_index_config(
        embedding_model=args.model,
        embedding_provider=args.embedding_provider,
        embedding_batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
        chunker_config=chunker_config,
    )

    vector_store = MemoryVectorStore(persist_path=index_path)

    if args.incremental and index_config_matches(vector_store.get_index_config(), index_config):
        return run_incremental_index(
            args=args,
            vault_path=vault_path,
            markdown_files=markdown_files,
            index_path=index_path,
            bm25_index_path=bm25_index_path,
            chunker_config=chunker_config,
            index_config=index_config,
            vector_store=vector_store,
        )

    if args.incremental:
        print("Index config missing or changed. Falling back to full rebuild.")

    return run_full_index(
        args=args,
        vault_path=vault_path,
        markdown_files=markdown_files,
        index_path=index_path,
        bm25_index_path=bm25_index_path,
        chunker_config=chunker_config,
        index_config=index_config,
        vector_store=vector_store,
    )


def run_full_index(
    args: argparse.Namespace,
    vault_path: Path,
    markdown_files: list[Path],
    index_path: Path,
    bm25_index_path: Path,
    chunker_config: ChunkerConfig,
    index_config: dict,
    vector_store: MemoryVectorStore,
) -> int:
    current_files = build_current_file_states(
        vault_root=vault_path,
        markdown_files=markdown_files,
    )
    chunker = HeadingChunker(chunker_config)
    vector_store.clear()
    vector_store.set_index_config(index_config)
    vector_store.persist()

    embedder = create_index_embedder(args)
    indexed_count = index_files_checkpointed(
        embedder=embedder,
        vector_store=vector_store,
        chunker=chunker,
        vault_path=vault_path,
        current_files=current_files,
        note_paths=sorted(current_files),
        batch_size=args.embed_batch_size,
    )

    vector_store.set_index_config(index_config)
    vector_store.persist()
    rebuild_bm25_index(bm25_index_path, vector_store.get_text_chunks())

    print("Index mode: full")
    print(f"Indexed chunks: {indexed_count}")
    print(f"Total chunks: {vector_store.count()}")
    print(f"Vector index path: {index_path.resolve()}")
    print(f"BM25 index path: {bm25_index_path.resolve()}")
    return 0


def run_incremental_index(
    args: argparse.Namespace,
    vault_path: Path,
    markdown_files: list[Path],
    index_path: Path,
    bm25_index_path: Path,
    chunker_config: ChunkerConfig,
    index_config: dict,
    vector_store: MemoryVectorStore,
) -> int:
    current_files = build_current_file_states(
        vault_root=vault_path,
        markdown_files=markdown_files,
    )
    old_files = vector_store.get_files_metadata()
    plan = plan_incremental_update(
        old_files=old_files,
        current_files=current_files,
    )

    for note_path in sorted(set(plan.deleted) | set(plan.modified)):
        old_metadata = old_files.get(note_path, {})
        chunk_ids = metadata_chunk_ids(old_metadata)
        if chunk_ids:
            vector_store.delete_chunks(chunk_ids)
        else:
            vector_store.delete(note_path)
        vector_store.remove_file_metadata(note_path)

    indexed_count = 0
    if plan.changed:
        chunker = HeadingChunker(chunker_config)
        embedder = create_index_embedder(args)
        indexed_count = index_files_checkpointed(
            embedder=embedder,
            vector_store=vector_store,
            chunker=chunker,
            vault_path=vault_path,
            current_files=current_files,
            note_paths=plan.changed,
            batch_size=args.embed_batch_size,
        )

    vector_store.set_index_config(index_config)
    vector_store.persist()
    rebuild_bm25_index(bm25_index_path, vector_store.get_text_chunks())

    print("Index mode: incremental")
    print(f"Added files: {len(plan.added)}")
    print(f"Modified files: {len(plan.modified)}")
    print(f"Deleted files: {len(plan.deleted)}")
    print(f"Unchanged files: {len(plan.unchanged)}")
    print(f"Embedded chunks: {indexed_count}")
    print(f"Total chunks: {vector_store.count()}")
    print(f"Vector index path: {index_path.resolve()}")
    print(f"BM25 index path: {bm25_index_path.resolve()}")
    return 0


def chunk_selected_files(
    chunker: HeadingChunker,
    vault_path: Path,
    current_files: dict,
    note_paths: list[str],
) -> tuple[list, dict[str, list[str]]]:
    chunks = []
    chunks_by_note: dict[str, list[str]] = {}
    for note_path in note_paths:
        state = current_files[note_path]
        note_chunks = chunker.chunk_file(vault_root=vault_path, file_path=state.file_path)
        chunks.extend(note_chunks)
        chunks_by_note[note_path] = [chunk.chunk_id for chunk in note_chunks]
    return chunks, chunks_by_note


def index_files_checkpointed(
    embedder,
    vector_store: MemoryVectorStore,
    chunker: HeadingChunker,
    vault_path: Path,
    current_files: dict,
    note_paths: list[str],
    batch_size: int,
) -> int:
    indexed_count = 0
    total_files = len(note_paths)
    for file_index, note_path in enumerate(note_paths, start=1):
        state = current_files[note_path]
        note_chunks = chunker.chunk_file(vault_root=vault_path, file_path=state.file_path)
        print(f"Indexing file {file_index}/{total_files}: {note_path} ({len(note_chunks)} chunks)")

        indexed_count += index_chunks_checkpointed(
            embedder=embedder,
            vector_store=vector_store,
            chunks=note_chunks,
            batch_size=batch_size,
        )
        vector_store.set_file_metadata(
            note_path,
            state.to_metadata([chunk.chunk_id for chunk in note_chunks]),
        )
        vector_store.persist()

    return indexed_count


def index_chunks_checkpointed(
    embedder,
    vector_store: MemoryVectorStore,
    chunks: list[TextChunk],
    batch_size: int,
) -> int:
    if not chunks:
        return 0

    indexed_count = 0
    safe_batch_size = max(batch_size, 1)
    for start in range(0, len(chunks), safe_batch_size):
        batch_chunks = chunks[start : start + safe_batch_size]
        texts = [build_chunk_embedding_text(chunk) for chunk in batch_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.upsert(
            [
                EmbeddingChunk(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(batch_chunks, embeddings, strict=False)
            ]
        )
        indexed_count += len(batch_chunks)
        vector_store.persist()
        print(f"  persisted chunks {indexed_count}/{len(chunks)}")

    return indexed_count


def rebuild_bm25_index(bm25_index_path: Path, chunks: list) -> None:
    bm25_index = BM25Index(persist_path=bm25_index_path)
    bm25_index.build(chunks)
    bm25_index.persist()


def run_search(args: argparse.Namespace) -> int:
    index_path = Path(args.index)
    bm25_index_path = Path(args.bm25_index) if args.bm25_index else derive_bm25_index_path(index_path)

    if args.mode in {"dense", "hybrid", "hybrid-rerank"} and not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if args.mode in {"bm25", "hybrid", "hybrid-rerank"} and not bm25_index_path.exists():
        raise FileNotFoundError(f"BM25 index file not found: {bm25_index_path}")

    embedder = create_index_embedder(args) if args.mode in {"dense", "hybrid", "hybrid-rerank"} else None
    vector_store = MemoryVectorStore(persist_path=index_path if index_path.exists() else None)
    bm25_index = BM25Index(persist_path=bm25_index_path) if args.mode in {"bm25", "hybrid", "hybrid-rerank"} else None
    reranker = create_reranker(args) if args.mode == "hybrid-rerank" else None

    manager = RAGManager(
        chunker=HeadingChunker(),
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
    )

    if args.mode == "dense":
        results = manager.dense_search(query=args.query, top_k=args.top_k)
    elif args.mode == "bm25":
        results = manager.bm25_search(query=args.query, top_k=args.top_k)
    elif args.mode == "hybrid":
        results = manager.hybrid_search(
            query=args.query,
            top_k=args.top_k,
            dense_top_k=args.dense_top_k,
            bm25_top_k=args.bm25_top_k,
            rrf_k=args.rrf_k,
        )
    else:
        results = manager.hybrid_rerank_search(
            query=args.query,
            top_k=args.top_k,
            rerank_candidates=args.rerank_candidates,
            dense_top_k=args.dense_top_k,
            bm25_top_k=args.bm25_top_k,
            rrf_k=args.rrf_k,
        )

    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"Results: {len(results)}")
    print("")

    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else "(no heading)"
        preview = chunk.text[:500].replace("\n", " ")

        print(f"{index}. {chunk.note_path}")
        print(f"   heading: {heading}")
        print(f"   score: {result.score}")
        if chunk.start_line is not None and chunk.end_line is not None:
            print(f"   lines: {chunk.start_line}-{chunk.end_line}")
        print(f"   text: {preview}")
        print("")

    return 0


def run_index_and_eval(args: argparse.Namespace) -> int:
    """一次加载模型，先建索引再跑评测。

    省掉 index 和 eval 两个命令分别加载模型的开销，
    模型只从磁盘加载一次到内存。
    """
    import rag_eval as rag_eval_module

    vault_path = Path(args.vault)
    index_path = Path(args.index)
    bm25_index_path = Path(args.bm25_index) if args.bm25_index else derive_bm25_index_path(index_path)

    # ── 索引阶段 ──
    if args.reset_index and index_path.exists():
        index_path.unlink()
        print(f"Reset index: {index_path.resolve()}")
    if args.reset_index and bm25_index_path.exists():
        bm25_index_path.unlink()
        print(f"Reset BM25 index: {bm25_index_path.resolve()}")

    exclusions = load_exclusion_patterns(vault_path)
    exclusion_filter = ExclusionFilter(exclusions)
    scan_result = scan_vault(vault_path, exclusion_filter)

    markdown_files = [note.path for note in scan_result.notes]

    print(f"Vault: {scan_result.vault_path}")
    print(f"Markdown files: {len(markdown_files)}")
    print(f"Excluded: {scan_result.excluded_count}")
    print(f"Failed: {len(scan_result.failed)}")

    chunker_config = ChunkerConfig(
        max_chunk_chars=args.max_chunk_chars,
        target_chunk_chars=args.target_chunk_chars,
        min_chunk_chars=args.min_chunk_chars,
        chunk_overlap=args.chunk_overlap,
        strip_code_blocks=args.strip_code_blocks,
    )
    index_config = build_index_config(
        embedding_model=args.model,
        embedding_provider=args.embedding_provider,
        embedding_batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
        chunker_config=chunker_config,
    )
    chunker = HeadingChunker(chunker_config)
    current_files = build_current_file_states(
        vault_root=vault_path,
        markdown_files=markdown_files,
    )
    chunks, chunks_by_note = chunk_selected_files(
        chunker=chunker,
        vault_path=vault_path,
        current_files=current_files,
        note_paths=sorted(current_files),
    )

    embedder = create_index_embedder(args)
    vector_store = MemoryVectorStore(persist_path=index_path)
    vector_store.clear()
    vector_store.set_index_config(index_config)

    manager = RAGManager(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    indexed_count = manager.index_chunks(chunks)

    bm25_index = BM25Index(persist_path=bm25_index_path)
    bm25_index.build(chunks)
    bm25_index.persist()
    manager.bm25_index = bm25_index
    if args.mode == "hybrid-rerank":
        manager.reranker = create_reranker(args)

    for note_path, state in current_files.items():
        vector_store.set_file_metadata(
            note_path,
            state.to_metadata(chunks_by_note.get(note_path, [])),
        )
    vector_store.set_index_config(index_config)
    vector_store.persist()

    print(f"Indexed chunks: {indexed_count}")
    print(f"Vector index path: {index_path.resolve()}")
    print(f"BM25 index path: {bm25_index_path.resolve()}")

    # ── 评测阶段（复用已加载的 manager） ──
    print("")
    print("=" * 50)
    print("Running eval (reusing loaded model)...")
    print("")

    eval_cases = rag_eval_module.load_eval_cases(Path(args.eval))
    hit_ks = rag_eval_module.parse_hit_ks(args.hit_ks, args.top_k)

    results: list[rag_eval_module.EvalCaseResult] = []
    for case in eval_cases:
        results.append(
            rag_eval_module.evaluate_case(
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

    rag_eval_module.print_summary(results, hit_ks)
    rag_eval_module.print_cases(results, show_success=False)

    if args.out:
        payload = rag_eval_module.build_eval_payload(
            args=args,
            eval_cases=eval_cases,
            results=results,
            hit_ks=hit_ks,
        )
        rag_eval_module.write_eval_payload(Path(args.out), payload)
        print(f"Saved eval result: {Path(args.out).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
