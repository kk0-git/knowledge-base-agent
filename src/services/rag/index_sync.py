from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from knowledge_base_agent.config import load_dotenv, load_exclusion_patterns
from knowledge_base_agent.scanner import ExclusionFilter, scan_vault
from services.rag.bm25 import BM25Index
from services.rag.chunker import ChunkerConfig, HeadingChunker
from services.rag.embedder import build_chunk_embedding_text, create_embedder
from services.rag.incremental import (
    build_current_file_states,
    build_index_config,
    metadata_chunk_ids,
    plan_incremental_update,
)
from services.rag.memory_vector_store import MemoryVectorStore
from services.rag.schema import EmbeddingChunk, TextChunk

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class RAGIndexSyncConfig:
    vault_path: Path
    index_path: Path
    bm25_index_path: Path
    project_root: Path
    model_name: str
    embedding_provider: str = "local"
    embed_batch_size: int = 32
    max_seq_length: int | None = None
    max_chunk_chars: int = 1500
    target_chunk_chars: int = 900
    min_chunk_chars: int = 200
    chunk_overlap: int = 200
    chunk_split_mode: str = "indexed"
    strip_code_blocks: bool = False
    allow_full_rebuild: bool = False
    excluded_roots: tuple[Path, ...] = ()
    changed_note_paths: tuple[str, ...] | None = None
    deleted_note_paths: tuple[str, ...] | None = None


def sync_rag_index(
    config: RAGIndexSyncConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Incrementally sync the vector and BM25 indexes with the vault.

    This is intentionally conservative for web-triggered sync: if the persisted
    index config does not match the current chunk/embed config, it returns a
    skipped result unless allow_full_rebuild is explicitly enabled.
    """
    load_dotenv(config.project_root / ".env")

    markdown_files, excluded_count, failed_count = scan_source_markdown_files(
        vault_path=config.vault_path,
        excluded_roots=list(config.excluded_roots),
    )
    emit_progress(
        progress,
        "rag_scanned",
        {
            "markdown_files": len(markdown_files),
            "excluded_markdown_files": excluded_count,
            "failed_files": failed_count,
        },
    )
    chunker_config = ChunkerConfig(
        max_chunk_chars=config.max_chunk_chars,
        target_chunk_chars=config.target_chunk_chars,
        min_chunk_chars=config.min_chunk_chars,
        chunk_overlap=config.chunk_overlap,
        chunk_split_mode=config.chunk_split_mode,
        strip_code_blocks=config.strip_code_blocks,
    )
    index_config = build_index_config(
        embedding_model=config.model_name,
        embedding_provider=config.embedding_provider,
        embedding_batch_size=config.embed_batch_size,
        max_seq_length=config.max_seq_length,
        chunker_config=chunker_config,
    )

    vector_store = MemoryVectorStore(persist_path=config.index_path)
    config_matches = sync_index_config_matches(vector_store.get_index_config(), index_config)
    if not config_matches and not config.allow_full_rebuild:
        emit_progress(progress, "rag_skipped", {"reason": "index_config_mismatch"})
        return {
            "mode": "skipped",
            "reason": "index_config_mismatch",
            "config_matches": False,
            "markdown_files": len(markdown_files),
            "excluded_markdown_files": excluded_count,
            "failed_files": failed_count,
            "index_path": str(config.index_path),
            "bm25_index_path": str(config.bm25_index_path),
        }

    if not config_matches and config.allow_full_rebuild:
        emit_progress(progress, "rag_full_rebuild", {"reason": "index_config_mismatch"})
        vector_store.clear()

    current_files = build_current_file_states(
        vault_root=config.vault_path,
        markdown_files=markdown_files,
    )
    old_files = vector_store.get_files_metadata()
    plan = plan_incremental_update(old_files=old_files, current_files=current_files)
    use_external_dirty_plan = config_matches and (
        config.changed_note_paths is not None or config.deleted_note_paths is not None
    )
    if use_external_dirty_plan:
        plan = override_incremental_plan(
            old_files=old_files,
            current_files=current_files,
            changed_note_paths=config.changed_note_paths or (),
            deleted_note_paths=config.deleted_note_paths or (),
        )
    emit_progress(
        progress,
        "rag_plan",
        {
            "added_files": len(plan.added),
            "modified_files": len(plan.modified),
            "deleted_files": len(plan.deleted),
            "unchanged_files": len(plan.unchanged),
            "changed_files": len(plan.changed),
        },
    )

    for note_path in sorted(set(plan.deleted) | set(plan.modified)):
        old_metadata = old_files.get(note_path, {})
        chunk_ids = metadata_chunk_ids(old_metadata)
        if chunk_ids:
            vector_store.delete_chunks(chunk_ids)
        else:
            vector_store.delete(note_path)
        vector_store.remove_file_metadata(note_path)
    if plan.deleted or plan.modified:
        emit_progress(
            progress,
            "rag_removed_old_chunks",
            {
                "deleted_files": len(plan.deleted),
                "modified_files": len(plan.modified),
            },
        )

    embedded_chunks = 0
    if plan.changed:
        chunker = HeadingChunker(chunker_config)
        embedder = create_embedder(
            provider=config.embedding_provider,
            model_name=config.model_name,
            batch_size=config.embed_batch_size,
            max_seq_length=config.max_seq_length,
        )
        embedded_chunks = index_changed_files(
            embedder=embedder,
            vector_store=vector_store,
            chunker=chunker,
            vault_path=config.vault_path,
            current_files=current_files,
            note_paths=plan.changed,
            batch_size=config.embed_batch_size,
            progress=progress,
        )

    vector_store.set_index_config(index_config)
    vector_store.persist()
    emit_progress(progress, "rag_rebuild_bm25", {"chunks": vector_store.count()})
    rebuild_bm25_index(config.bm25_index_path, vector_store.get_text_chunks())
    emit_progress(
        progress,
        "rag_done",
        {
            "embedded_chunks": embedded_chunks,
            "total_chunks": vector_store.count(),
        },
    )

    return {
        "mode": "full" if not config_matches else "incremental",
        "reason": None,
        "config_matches": config_matches,
        "markdown_files": len(markdown_files),
        "excluded_markdown_files": excluded_count,
        "failed_files": failed_count,
        "added_files": len(plan.added),
        "modified_files": len(plan.modified),
        "deleted_files": len(plan.deleted),
        "unchanged_files": len(plan.unchanged),
        "embedded_chunks": embedded_chunks,
        "total_chunks": vector_store.count(),
        "index_path": str(config.index_path),
        "bm25_index_path": str(config.bm25_index_path),
        "changed_files": plan.changed,
    }


def scan_source_markdown_files(
    *,
    vault_path: Path,
    excluded_roots: list[Path] | None = None,
) -> tuple[list[Path], int, int]:
    exclusions = load_exclusion_patterns(vault_path)
    exclusion_filter = ExclusionFilter(exclusions)
    scan_result = scan_vault(vault_path, exclusion_filter)
    excluded_resolved = [root.resolve() for root in (excluded_roots or [])]

    markdown_files = []
    for note in scan_result.notes:
        path = note.path
        resolved_path = path.resolve()
        if any(is_relative_to_path(resolved_path, root) for root in excluded_resolved):
            continue
        rel_parts = path.relative_to(vault_path).parts
        if any(part.startswith("wiki_") for part in rel_parts):
            continue
        markdown_files.append(path)

    return markdown_files, scan_result.excluded_count, len(scan_result.failed)


def index_changed_files(
    *,
    embedder,
    vector_store: MemoryVectorStore,
    chunker: HeadingChunker,
    vault_path: Path,
    current_files: dict,
    note_paths: list[str],
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> int:
    embedded_count = 0
    total_files = len(note_paths)
    for file_index, note_path in enumerate(note_paths, start=1):
        state = current_files[note_path]
        chunks = chunker.chunk_file(vault_root=vault_path, file_path=state.file_path)
        emit_progress(
            progress,
            "rag_embedding_file",
            {
                "file_index": file_index,
                "file_count": total_files,
                "chunks": len(chunks),
                "note_path": note_path,
            },
        )
        embedded_count += index_chunks(
            embedder=embedder,
            vector_store=vector_store,
            chunks=chunks,
            batch_size=batch_size,
            progress=progress,
            file_index=file_index,
            file_count=total_files,
        )
        vector_store.set_file_metadata(
            note_path,
            state.to_metadata([chunk.chunk_id for chunk in chunks]),
        )
        vector_store.persist()

    return embedded_count


def index_chunks(
    *,
    embedder,
    vector_store: MemoryVectorStore,
    chunks: list[TextChunk],
    batch_size: int,
    progress: ProgressCallback | None = None,
    file_index: int | None = None,
    file_count: int | None = None,
) -> int:
    if not chunks:
        return 0

    embedded_count = 0
    safe_batch_size = max(batch_size, 1)
    total_batches = (len(chunks) + safe_batch_size - 1) // safe_batch_size
    for batch_index, start in enumerate(range(0, len(chunks), safe_batch_size), start=1):
        batch_chunks = chunks[start : start + safe_batch_size]
        emit_progress(
            progress,
            "rag_embedding_batch",
            {
                "file_index": file_index,
                "file_count": file_count,
                "batch_index": batch_index,
                "batch_count": total_batches,
                "texts": len(batch_chunks),
            },
        )
        texts = [build_chunk_embedding_text(chunk) for chunk in batch_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.upsert(
            [
                EmbeddingChunk(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(batch_chunks, embeddings, strict=False)
            ]
        )
        embedded_count += len(batch_chunks)
        vector_store.persist()

    return embedded_count


def emit_progress(
    progress: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if progress is not None:
        progress(event, payload)


def rebuild_bm25_index(bm25_index_path: Path, chunks: list[TextChunk]) -> None:
    bm25_index = BM25Index(persist_path=bm25_index_path)
    bm25_index.build(chunks)
    bm25_index.persist()


def sync_index_config_matches(old_config: dict[str, Any], current_config: dict[str, Any]) -> bool:
    old_normalized = dict(old_config)
    current_normalized = dict(current_config)

    # Batch size changes request throughput only; it does not change embedding
    # values or chunk identity, so it should not force a rebuild.
    old_normalized.pop("embedding_batch_size", None)
    current_normalized.pop("embedding_batch_size", None)

    # Older indexes were built after indexed splitting became the effective
    # default but before it was persisted in index_config.
    if "chunk_split_mode" not in old_normalized:
        old_normalized["chunk_split_mode"] = "indexed"

    return old_normalized == current_normalized


def override_incremental_plan(
    *,
    old_files: dict[str, dict[str, Any]],
    current_files: dict[str, FileState],
    changed_note_paths: tuple[str, ...],
    deleted_note_paths: tuple[str, ...],
) -> IncrementalPlan:
    current_paths = set(current_files)
    old_paths = set(old_files)
    changed = sorted({path for path in changed_note_paths if path in current_paths})
    deleted = sorted({path for path in deleted_note_paths if path in old_paths})
    added = [path for path in changed if path not in old_paths]
    modified = [path for path in changed if path in old_paths]
    unchanged = sorted(current_paths - set(changed))
    return IncrementalPlan(
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
        current_files=current_files,
    )


def is_relative_to_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
