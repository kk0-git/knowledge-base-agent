from __future__ import annotations

from pathlib import Path
from typing import Any

from services.rag.context_packer import score_type_for_mode
from services.rag.manager import RAGManager
from services.rag.schema import SearchResult
from services.wiki.manager import rebuild_tag_index, scan_markdown_files
from services.wiki.state_store import WikiStateStore
from services.workflows.schema import ScopeResult, ScopeSpec


class ScopeResolver:
    def __init__(
        self,
        *,
        vault_root: Path,
        wiki_state_store: WikiStateStore | None = None,
        wiki_dir: Path | None = None,
        rag_manager: RAGManager | None = None,
        overview_note_threshold: int = 30,
    ) -> None:
        self.vault_root = vault_root
        self.wiki_state_store = wiki_state_store
        self.wiki_dir = wiki_dir
        self.rag_manager = rag_manager
        self.overview_note_threshold = overview_note_threshold

    def resolve(self, scope: ScopeSpec) -> ScopeResult:
        if scope.type == "tag":
            return self.resolve_tag(scope)
        if scope.type == "search":
            return self.resolve_search(scope)
        if scope.type == "selected_notes":
            return self.resolve_selected_notes(scope)
        if scope.type == "folder":
            return self.resolve_folder(scope)
        if scope.type == "all_vault":
            return self.resolve_all_vault(scope)
        if scope.type == "current_context":
            return ScopeResult(scope=scope, metadata={"source": "current_context"})
        raise ValueError(f"Unsupported scope type: {scope.type}")

    def resolve_tag(self, scope: ScopeSpec) -> ScopeResult:
        if self.wiki_state_store is None:
            raise ValueError("wiki_state_store is required for tag scope")
        tag = str(scope.value or "").strip()
        if not tag:
            raise ValueError("tag scope requires value")

        state = rebuild_tag_index(
            self.wiki_state_store.load(),
            overview_note_threshold=self.overview_note_threshold,
        )
        if tag not in state.tags:
            raise ValueError(f"Tag not found in wiki state: {tag}")
        record = state.tags[tag]
        notes = tuple(
            note_item(path=path, reason=f"tag:{tag}")
            for path in record.source_paths
            if (self.vault_root / path).exists()
        )
        return ScopeResult(
            scope=scope,
            notes=notes,
            metadata={
                "tag": tag,
                "note_count": len(notes),
                "wiki_policy": record.wiki_policy,
                "dirty": record.dirty,
            },
        )

    def resolve_search(self, scope: ScopeSpec) -> ScopeResult:
        if self.rag_manager is None:
            raise ValueError("rag_manager is required for search scope")
        query = str(scope.value or "").strip()
        if not query:
            raise ValueError("search scope requires value")

        results = self.rag_manager.hybrid_search(
            query=query,
            top_k=scope.top_k,
            dense_top_k=int(scope.options.get("dense_top_k", 50)),
            bm25_top_k=int(scope.options.get("bm25_top_k", 50)),
            rrf_k=int(scope.options.get("rrf_k", 60)),
        )
        notes_by_path: dict[str, dict[str, Any]] = {}
        chunks: list[dict[str, Any]] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            notes_by_path.setdefault(
                chunk.note_path,
                note_item(path=chunk.note_path, reason=f"search:{query}"),
            )
            chunks.append(search_result_to_chunk_item(index=index, result=result))
        return ScopeResult(
            scope=scope,
            notes=tuple(notes_by_path.values()),
            chunks=tuple(chunks),
            metadata={"query": query, "result_count": len(results)},
        )

    def resolve_selected_notes(self, scope: ScopeSpec) -> ScopeResult:
        paths = tuple(path for path in scope.paths if path and (self.vault_root / path).exists())
        return ScopeResult(
            scope=scope,
            notes=tuple(note_item(path=path, reason="selected") for path in paths),
            metadata={"note_count": len(paths)},
        )

    def resolve_folder(self, scope: ScopeSpec) -> ScopeResult:
        folder = str(scope.value or "").strip()
        if not folder:
            raise ValueError("folder scope requires value")
        root = (self.vault_root / folder).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        notes = [
            note_item(path=path.relative_to(self.vault_root).as_posix(), reason=f"folder:{folder}")
            for path in root.rglob("*.md")
            if path.is_file()
        ]
        return ScopeResult(
            scope=scope,
            notes=tuple(sorted(notes, key=lambda item: item["path"])),
            metadata={"folder": folder, "note_count": len(notes)},
        )

    def resolve_all_vault(self, scope: ScopeSpec) -> ScopeResult:
        notes = [
            note_item(path=path.relative_to(self.vault_root).as_posix(), reason="all_vault")
            for path in scan_markdown_files(self.vault_root, excluded_roots=[self.wiki_dir] if self.wiki_dir else None)
        ]
        return ScopeResult(
            scope=scope,
            notes=tuple(sorted(notes, key=lambda item: item["path"])),
            metadata={"note_count": len(notes)},
        )


def note_item(*, path: str, reason: str) -> dict[str, Any]:
    return {
        "path": path,
        "title": Path(path).stem,
        "reason": reason,
    }


def search_result_to_chunk_item(*, index: int, result: SearchResult) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "citation_id": f"N{index}",
        "source_type": "local_hybrid",
        "path": chunk.note_path,
        "heading": " > ".join(chunk.heading_path) if chunk.heading_path else "",
        "lines": f"{chunk.start_line}-{chunk.end_line}",
        "score": round(float(result.score), 6),
        "score_type": score_type_for_mode("hybrid"),
        "chunk_id": chunk.chunk_id,
        "text": chunk.text.strip(),
    }
