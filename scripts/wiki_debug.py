from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.rag.index_sync import RAGIndexSyncConfig, sync_rag_index
from services.rag.memory_vector_store import MemoryVectorStore
from services.workspace_state import (
    WorkspaceFileRecord,
    WorkspaceState,
    WorkspaceStateStore,
    build_workspace_dirty_plan,
    mark_rag_indexed,
    mark_tags_extracted,
    remove_deleted,
    scan_workspace_files,
)
from services.wiki.manager import (
    WikiManager,
    import_manual_policies,
    rebuild_tag_index,
    scan_markdown_files,
    set_tag_policy,
)
from services.wiki.report_writer import write_obsidian_wiki_report
from services.wiki.state_store import WikiStateStore
from services.wiki.tag_consolidation import (
    LLMTagConsolidator,
    propose_deterministic_cleanup,
    tag_cleanup_proposal_to_dict,
)
from services.wiki.tag_refinement import LLMTagRefiner, tag_refinement_proposal_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug wiki tagging and synthesis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_common_args(subparsers.add_parser("tag", help="Tag changed notes with LLM"))
    add_common_args(subparsers.add_parser("synthesize", help="Generate dirty wiki pages"))
    synthesize_tag_parser = subparsers.add_parser("synthesize-tag", help="Generate one wiki page by tag")
    add_common_args(synthesize_tag_parser)
    synthesize_tag_parser.add_argument("--tag", required=True, help="Tag to synthesize, e.g. java/servlet")
    add_common_args(subparsers.add_parser("rebuild", help="Tag notes and generate wiki pages"))
    add_common_args(subparsers.add_parser("report", help="Print wiki state report"))
    add_common_args(subparsers.add_parser("write-report", help="Write Obsidian wiki status report markdown"))
    add_common_args(subparsers.add_parser("consolidate-tags", help="Propose tag merges"))
    refine_parser = subparsers.add_parser("refine-tag", help="Propose a split/refinement for one broad tag")
    add_common_args(refine_parser)
    refine_parser.add_argument("--tag", required=True, help="Tag to inspect, e.g. web/backend")
    policy_parser = subparsers.add_parser("set-policy", help="Set wiki generation policy for one tag")
    add_common_args(policy_parser)
    policy_parser.add_argument("--tag", required=True, help="Tag to update")
    policy_parser.add_argument("--policy", choices=["generate", "overview", "skip"], required=True)
    import_policy_parser = subparsers.add_parser("import-policies", help="Import manual policies from another wiki state")
    add_common_args(import_policy_parser)
    import_policy_parser.add_argument("--from-state", required=True, help="Source wiki state JSON path")

    args = parser.parse_args()
    manager = build_manager(args)

    if args.command == "tag":
        result = manager.tag_changed_notes(force=args.force, limit=args.limit)
        print_json(result)
        return 0

    if args.command == "synthesize":
        result = manager.synthesize_dirty_wikis(
            force=args.force,
            limit=args.limit,
            policy_filter=args.synthesize_policy,
        )
        print_json(result)
        return 0

    if args.command == "synthesize-tag":
        tag = normalize_cli_tag(args.tag)
        result = manager.synthesize_tag(tag, force=args.force)
        report = manager.report()
        report_path = write_obsidian_wiki_report(
            report=report,
            wiki_dir=manager.wiki_dir,
            report_path=Path(args.report_path) if args.report_path else None,
            sync_result={"wiki": result},
        )
        print_synthesize_success(tag, result, report_path)
        return 0

    if args.command == "rebuild":
        tag_result = manager.tag_changed_notes(force=args.force, limit=args.limit)
        synth_result = manager.synthesize_dirty_wikis(
            force=args.force,
            limit=args.limit,
            policy_filter=args.synthesize_policy,
        )
        print_json({"tag": tag_result, "synthesize": synth_result})
        return 0

    if args.command == "report":
        report = manager.report()
        write_report(args, report)
        print_json(report)
        return 0

    if args.command == "write-report":
        report = manager.report()
        report_path = write_obsidian_wiki_report(
            report=report,
            wiki_dir=manager.wiki_dir,
            report_path=Path(args.report_path) if args.report_path else None,
        )
        print_json({"report_path": str(report_path), "report": report})
        return 0

    if args.command == "consolidate-tags":
        payload = run_tag_consolidation(args, manager)
        print_json(payload)
        return 0

    if args.command == "refine-tag":
        proposal = run_tag_refinement(args)
        payload = {"tag_refinement": tag_refinement_proposal_to_dict(proposal)}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print_json(payload)
        return 0

    if args.command == "set-policy":
        payload = run_set_policy(args)
        print_json(payload)
        return 0

    if args.command == "import-policies":
        payload = run_import_policies(args)
        print_json(payload)
        return 0

    return 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vault", required=True, help="Vault root")
    parser.add_argument("--state", default="./wiki-state/wiki_state.json", help="Wiki state JSON path")
    parser.add_argument("--wiki-dir", default=None, help="Wiki output directory. Default: <vault>/wiki")
    parser.add_argument("--min-notes-per-tag", type=int, default=3)
    parser.add_argument("--min-tag-depth", type=int, default=2)
    parser.add_argument("--overview-note-threshold", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--synthesize-policy", choices=["overview", "generate"], default=None)
    parser.add_argument("--out", default=None, help="Optional report/proposal output JSON path")
    parser.add_argument("--report-path", default=None, help="Optional Obsidian wiki report markdown path")


def add_sync_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-state", default="./wiki-state/workspace_state.json")
    parser.add_argument("--index", default="./rag-index/mixed-siliconflow-bge-m3.json")
    parser.add_argument("--bm25-index", default="./rag-index/mixed-siliconflow-bge-m3.bm25.json")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="openai_compatible")
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-chunk-chars", type=int, default=1500)
    parser.add_argument("--target-chunk-chars", type=int, default=900)
    parser.add_argument("--min-chunk-chars", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--chunk-split-mode", choices=["streaming", "indexed"], default="indexed")
    parser.add_argument("--strip-code-blocks", action="store_true")
    parser.add_argument("--allow-full-rebuild", action="store_true")
    parser.add_argument("--no-rag", action="store_true", help="Only sync wiki tags and report")


def build_manager(args: argparse.Namespace) -> WikiManager:
    llm_config = load_llm_config(PROJECT_ROOT)
    llm_client = create_llm_client(llm_config)
    vault_root = Path(args.vault)
    wiki_dir = Path(args.wiki_dir) if args.wiki_dir else None
    return WikiManager(
        vault_root=vault_root,
        state_store=WikiStateStore(Path(args.state)),
        llm_client=llm_client,
        llm_model=llm_config.model,
        wiki_dir=wiki_dir,
        min_notes_per_tag=args.min_notes_per_tag,
        min_tag_depth=args.min_tag_depth,
        overview_note_threshold=args.overview_note_threshold,
    )


def run_tag_consolidation(args: argparse.Namespace, manager: WikiManager):
    llm_config = load_llm_config(PROJECT_ROOT)
    llm_client = create_llm_client(llm_config)
    state = rebuild_tag_index(
        WikiStateStore(Path(args.state)).load(),
        overview_note_threshold=args.overview_note_threshold,
    )
    deterministic = propose_deterministic_cleanup(state)
    llm_proposals = LLMTagConsolidator(
        client=llm_client,
        model=llm_config.model,
    ).propose_cleanup(
        state=state,
        vault_root=Path(args.vault),
    )
    payload = {
        "deterministic_proposals": [
            tag_cleanup_proposal_to_dict(proposal) for proposal in deterministic
        ],
        "llm_proposals": [
            tag_cleanup_proposal_to_dict(proposal) for proposal in llm_proposals
        ],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_tag_refinement(args: argparse.Namespace):
    llm_config = load_llm_config(PROJECT_ROOT)
    llm_client = create_llm_client(llm_config)
    state = WikiStateStore(Path(args.state)).load()
    return LLMTagRefiner(
        client=llm_client,
        model=llm_config.model,
    ).refine_tag(
        state=state,
        vault_root=Path(args.vault),
        tag=normalize_cli_tag(args.tag),
    )


def run_set_policy(args: argparse.Namespace) -> dict:
    store = WikiStateStore(Path(args.state))
    state = store.load()
    tag = normalize_cli_tag(args.tag)
    updated = set_tag_policy(state, tag=tag, wiki_policy=args.policy)
    store.save(updated)
    payload = {
        "tag": tag,
        "wiki_policy": args.policy,
        "dirty": updated.tags[tag].dirty,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_import_policies(args: argparse.Namespace) -> dict:
    target_store = WikiStateStore(Path(args.state))
    target_state = rebuild_tag_index(
        target_store.load(),
        overview_note_threshold=args.overview_note_threshold,
    )
    source_state = WikiStateStore(Path(args.from_state)).load()
    updated, imported = import_manual_policies(target_state, source_state)
    target_store.save(updated)
    payload = {
        "state": args.state,
        "from_state": args.from_state,
        "imported": imported,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_watch(args: argparse.Namespace, manager: WikiManager) -> dict:
    if args.once:
        return sync_workspace_and_report(args, manager)

    last_snapshot = vault_snapshot(Path(args.vault), manager.wiki_dir)
    pending_since: float | None = time.monotonic() if args.sync_on_start else None
    last_change_at: float | None = time.monotonic() if args.sync_on_start else None
    last_result: dict | None = None
    print(
        json.dumps(
            {
                "watching": args.vault,
                "quiet_seconds": args.quiet_seconds,
                "poll_seconds": args.poll_seconds,
                "sync_on_start": args.sync_on_start,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    while True:
        time.sleep(max(args.poll_seconds, 1.0))
        current_snapshot = vault_snapshot(Path(args.vault), manager.wiki_dir)
        if current_snapshot != last_snapshot:
            last_snapshot = current_snapshot
            pending_since = pending_since or time.monotonic()
            last_change_at = time.monotonic()
            print(json.dumps({"event": "changed", "pending": True}, ensure_ascii=False), flush=True)
            continue

        if pending_since is None or last_change_at is None:
            continue

        quiet_for = time.monotonic() - last_change_at
        if quiet_for < args.quiet_seconds:
            continue

        print(json.dumps({"event": "quiet_period_elapsed"}, ensure_ascii=False), flush=True)
        last_result = sync_workspace_and_report(args, manager)
        print(json.dumps({"event": "synced", "result": last_result}, ensure_ascii=False), flush=True)
        pending_since = None
        last_change_at = None

    return last_result or {}


def run_update_topic(args: argparse.Namespace, manager: WikiManager) -> dict:
    requested_tag = normalize_cli_tag(args.tag)
    sync_result = sync_workspace_and_report(args, manager, force_tags=False)
    state = rebuild_tag_index(
        WikiStateStore(Path(args.state)).load(),
        overview_note_threshold=args.overview_note_threshold,
    )
    if requested_tag not in state.tags:
        report = manager.report()
        report_path = write_obsidian_wiki_report(
            report=report,
            wiki_dir=manager.wiki_dir,
            report_path=Path(args.report_path) if args.report_path else None,
            sync_result=sync_result,
        )
        return {
            "tag": requested_tag,
            "tag_found": False,
            "sync": sync_result,
            "synthesize": None,
            "report_path": str(report_path),
        }

    synth_result = manager.synthesize_tag(requested_tag, force=True)
    report = manager.report()
    report_path = write_obsidian_wiki_report(
        report=report,
        wiki_dir=manager.wiki_dir,
        report_path=Path(args.report_path) if args.report_path else None,
        sync_result={
            "rag": sync_result.get("rag"),
            "wiki": sync_result.get("wiki"),
        },
    )
    return {
        "tag": requested_tag,
        "tag_found": True,
        "sync": sync_result,
        "synthesize": synth_result,
        "report_path": str(report_path),
    }


def sync_workspace_and_report(
    args: argparse.Namespace,
    manager: WikiManager,
    *,
    force_tags: bool | None = None,
) -> dict:
    workspace_store = WorkspaceStateStore(Path(args.workspace_state))
    workspace_state = workspace_store.load()
    current_workspace_files = scan_workspace_files(
        vault_root=Path(args.vault),
        excluded_roots=(manager.wiki_dir,),
    )
    workspace_state = hydrate_workspace_state(
        workspace_state=workspace_state,
        current_files=current_workspace_files,
        args=args,
        manager=manager,
    )
    workspace_plan = build_workspace_dirty_plan(
        previous_state=workspace_state,
        current_files=current_workspace_files,
    )
    workspace_state = WorkspaceState(files=workspace_plan.current_files)
    print_workspace_progress("workspace_plan", workspace_plan.to_summary())

    rag_result = None
    if not args.no_rag:
        print_workspace_progress("workspace_rag_start", {"index": args.index})
        rag_result = sync_rag_index(
            RAGIndexSyncConfig(
                vault_path=Path(args.vault),
                index_path=Path(args.index),
                bm25_index_path=Path(args.bm25_index),
                project_root=PROJECT_ROOT,
                model_name=args.model,
                embedding_provider=args.embedding_provider,
                embed_batch_size=args.embed_batch_size,
                max_seq_length=args.max_seq_length,
                max_chunk_chars=args.max_chunk_chars,
                target_chunk_chars=args.target_chunk_chars,
                min_chunk_chars=args.min_chunk_chars,
                chunk_overlap=args.chunk_overlap,
                chunk_split_mode=args.chunk_split_mode,
                strip_code_blocks=args.strip_code_blocks,
                allow_full_rebuild=args.allow_full_rebuild,
                excluded_roots=(manager.wiki_dir,),
                changed_note_paths=tuple(workspace_plan.embed_dirty_files),
                deleted_note_paths=tuple(workspace_plan.deleted_files),
            ),
            progress=print_workspace_progress,
        )
        if rag_result.get("mode") != "skipped":
            workspace_state = mark_rag_indexed(workspace_state, workspace_plan.embed_dirty_files)
    else:
        print_workspace_progress("workspace_rag_skipped", {"reason": "no_rag"})

    print_workspace_progress("workspace_tags_start", {})
    wiki_result = manager.tag_changed_notes(
        force=args.force if force_tags is None else force_tags,
        limit=args.limit,
        changed_note_paths=workspace_plan.tags_dirty_files,
    )
    if wiki_result.get("failed", 0) == 0:
        workspace_state = mark_tags_extracted(workspace_state, workspace_plan.tags_dirty_files)
    else:
        failed_paths = set(wiki_result.get("failed_paths", []))
        successful_tag_paths = [
            path for path in workspace_plan.tags_dirty_files if path not in failed_paths
        ]
        workspace_state = mark_tags_extracted(workspace_state, successful_tag_paths)
    workspace_state = remove_deleted(workspace_state, workspace_plan.deleted_files)
    workspace_store.save(workspace_state)
    final_workspace_summary = workspace_state_summary(workspace_state)
    print_workspace_progress(
        "workspace_tags_done",
        {
            "tagged": wiki_result.get("tagged", 0),
            "skipped": wiki_result.get("skipped", 0),
            "failed": wiki_result.get("failed", 0),
        },
    )
    report = manager.report()
    print_workspace_progress("workspace_report_start", {})
    report_path = write_obsidian_wiki_report(
        report=report,
        wiki_dir=manager.wiki_dir,
        report_path=Path(args.report_path) if args.report_path else None,
        sync_result={"workspace": final_workspace_summary, "rag": rag_result, "wiki": wiki_result},
    )
    print_workspace_progress("workspace_report_done", {"report_path": str(report_path)})
    return {
        "workspace": final_workspace_summary,
        "rag": rag_result,
        "wiki": wiki_result,
        "report_path": str(report_path),
    }


def hydrate_workspace_state(
    *,
    workspace_state: WorkspaceState,
    current_files: dict[str, WorkspaceFileRecord],
    args: argparse.Namespace,
    manager: WikiManager,
) -> WorkspaceState:
    existing = workspace_state.file_map()
    rag_metadata = MemoryVectorStore(persist_path=Path(args.index)).get_files_metadata()
    wiki_files = manager.state_store.load().files
    hydrated: dict[str, WorkspaceFileRecord] = {}

    for note_path, current in current_files.items():
        previous = existing.get(note_path)
        if previous and previous.content_hash == current.content_hash:
            record = replace(
                current,
                rag_indexed_hash=previous.rag_indexed_hash,
                tag_extracted_hash=previous.tag_extracted_hash,
            )
        else:
            record = current

        if record.rag_indexed_hash is None:
            rag_hash = str(rag_metadata.get(note_path, {}).get("content_hash", ""))
            if rag_hash == current.content_hash:
                record = replace(record, rag_indexed_hash=current.content_hash)
        if record.tag_extracted_hash is None:
            wiki_record = wiki_files.get(note_path)
            if wiki_record and wiki_record.content_hash == current.content_hash:
                record = replace(record, tag_extracted_hash=current.content_hash)
        hydrated[note_path] = record

    return WorkspaceState(files=hydrated)


def workspace_state_summary(state: WorkspaceState) -> dict:
    files = state.file_map()
    return {
        "files": len(files),
        "deleted_files": 0,
        "embed_dirty_files": sum(1 for record in files.values() if record.embed_dirty),
        "tags_dirty_files": sum(1 for record in files.values() if record.tags_dirty),
    }


def print_workspace_progress(event: str, payload: dict) -> None:
    message = format_workspace_progress(event, payload)
    if message:
        print(message, flush=True)


def format_workspace_progress(event: str, payload: dict) -> str:
    if event == "workspace_plan":
        return (
            "Workspace: "
            f"files={payload.get('files', 0)}, "
            f"embed_dirty={payload.get('embed_dirty_files', 0)}, "
            f"tags_dirty={payload.get('tags_dirty_files', 0)}, "
            f"deleted={payload.get('deleted_files', 0)}."
        )
    if event == "workspace_rag_start":
        return "RAG: sync started."
    if event == "workspace_rag_skipped":
        return "RAG: skipped."
    if event == "rag_scanned":
        return (
            "RAG: scanned "
            f"{payload.get('markdown_files', 0)} markdown files "
            f"({payload.get('excluded_markdown_files', 0)} excluded, "
            f"{payload.get('failed_files', 0)} failed)."
        )
    if event == "rag_skipped":
        return f"RAG: skipped ({payload.get('reason', 'unknown')})."
    if event == "rag_full_rebuild":
        return "RAG: full rebuild allowed."
    if event == "rag_plan":
        return (
            "RAG: plan "
            f"added={payload.get('added_files', 0)}, "
            f"modified={payload.get('modified_files', 0)}, "
            f"deleted={payload.get('deleted_files', 0)}, "
            f"changed={payload.get('changed_files', 0)}."
        )
    if event == "rag_removed_old_chunks":
        return "RAG: removed stale chunks."
    if event == "rag_embedding_file":
        return (
            "RAG: embedding file "
            f"{payload.get('file_index', 0)}/{payload.get('file_count', 0)} "
            f"chunks={payload.get('chunks', 0)}."
        )
    if event == "rag_embedding_batch":
        return (
            "RAG: embedding batch "
            f"{payload.get('batch_index', 0)}/{payload.get('batch_count', 0)} "
            f"texts={payload.get('texts', 0)}."
        )
    if event == "rag_rebuild_bm25":
        return f"RAG: rebuilding BM25 for {payload.get('chunks', 0)} chunks."
    if event == "rag_done":
        return (
            "RAG: done "
            f"embedded_chunks={payload.get('embedded_chunks', 0)}, "
            f"total_chunks={payload.get('total_chunks', 0)}."
        )
    if event == "workspace_tags_start":
        return "Wiki tags: sync started."
    if event == "workspace_tags_done":
        return (
            "Wiki tags: done "
            f"tagged={payload.get('tagged', 0)}, "
            f"skipped={payload.get('skipped', 0)}, "
            f"failed={payload.get('failed', 0)}."
        )
    if event == "workspace_report_start":
        return "Report: writing _Wiki Report.md."
    if event == "workspace_report_done":
        return f"Report: updated {payload.get('report_path', '')}."
    return ""


def vault_snapshot(vault_root: Path, wiki_dir: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in scan_markdown_files(vault_root, excluded_roots=[wiki_dir]):
        stat = path.stat()
        snapshot[path.relative_to(vault_root).as_posix()] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def write_report(args: argparse.Namespace, report: dict) -> None:
    if not args.out:
        return
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_report_markdown(report), encoding="utf-8")


def render_report_markdown(report: dict) -> str:
    lines = [
        "# Wiki State Report",
        "",
        f"- files: `{report['files']}`",
        f"- tags: `{report['tags']}`",
        f"- eligible_tags: `{report['eligible_tags']}`",
        f"- dirty_tags: `{report['dirty_tags']}`",
        "",
        "| tag | notes | evidence | policy | source | hints | dirty | eligible | wiki |",
        "|---|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in report["tag_rows"][:200]:
        evidence = ", ".join(
            f"{source}:{count}"
            for source, count in sorted(row.get("evidence_counts", {}).items())
        )
        hints = ", ".join(row.get("review_hints", []))
        lines.append(
            f"| {row['tag']} | {row['note_count']} | {evidence} | {row.get('wiki_policy', 'generate')} | {row.get('wiki_policy_source', 'auto')} | {hints} | {row['dirty']} | {row['eligible']} | {row.get('wiki_path') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def print_synthesize_success(tag: str, result: dict, report_path: Path) -> None:
    generated = int(result.get("generated", 0))
    failed = int(result.get("failed", 0))
    if failed:
        print(f"FAILED: wiki synthesis failed. failed={failed}")
        print_failed_wiki_summary(result)
        return
    print_retried_wiki_summary(result)
    if generated:
        print("OK: wiki refreshed and _Wiki Report.md updated.")
    else:
        print("OK: no wiki page needed refresh; _Wiki Report.md updated.")
    print(f"Report: {report_path}")


def print_update_topic_success(payload: dict) -> None:
    tag = payload.get("tag", "")
    sync = payload.get("sync") or {}
    rag = sync.get("rag") or {}
    wiki_sync = sync.get("wiki") or {}
    synth = payload.get("synthesize") or {}
    changed_notes = int(wiki_sync.get("tagged", 0))
    embedded_chunks = rag.get("embedded_chunks")
    if not payload.get("tag_found"):
        print("TAG NOT FOUND AFTER SYNC.")
        print("Open _Wiki Report.md and choose a current topic.")
        print(f"Report: {payload.get('report_path')}")
        return

    generated = int(synth.get("generated", 0))
    failed = int(synth.get("failed", 0))
    if failed:
        print(f"FAILED: topic update failed. failed={failed}")
        print_failed_wiki_summary(synth)
        print(f"Report: {payload.get('report_path')}")
        return

    print_retried_wiki_summary(synth)
    if embedded_chunks is None:
        print(f"OK: synced {changed_notes} changed notes, refreshed topic wiki.")
    else:
        print(
            "OK: "
            f"synced {changed_notes} changed notes, "
            f"embedded {embedded_chunks} chunks, "
            f"refreshed {generated} wiki."
        )
    print("Topic wiki updated.")
    print(f"Report: {payload.get('report_path')}")


def print_failed_wiki_summary(result: dict) -> None:
    failed_tags = result.get("failed_tags") or []
    if not failed_tags:
        return
    print("Failed topics:")
    for item in failed_tags[:10]:
        print(
            "- "
            f"{item.get('tag', '-')}: "
            f"{item.get('error_type', 'Error')}: {item.get('message', '')} "
            f"(attempts={item.get('attempts', '-')}, "
            f"retryable={item.get('retryable', False)})"
        )


def print_retried_wiki_summary(result: dict) -> None:
    retried_tags = result.get("retried_tags") or []
    if not retried_tags:
        return
    print("Retried topics:")
    for item in retried_tags[:10]:
        print(
            "- "
            f"{item.get('tag', '-')}: "
            f"succeeded after {item.get('attempts', '-')} attempts"
        )


def normalize_cli_tag(raw_tag: str) -> str:
    tag = str(raw_tag or "").strip()
    if "{{" in tag or "}}" in tag:
        raise ValueError(
            f"Tag variable was not resolved by Obsidian Shell Commands: {tag}. "
            "Use {{selection}} instead of {{selected_text}}."
        )
    if "`" in tag:
        start = tag.find("`")
        end = tag.find("`", start + 1)
        if start >= 0 and end > start:
            tag = tag[start + 1 : end].strip()
    tag = tag.strip("`").strip()
    if "·" in tag:
        tag = tag.split("·", 1)[0].strip()
    if tag.startswith("- [ ]"):
        tag = tag[len("- [ ]") :].strip()
    if tag.startswith("- [x]"):
        tag = tag[len("- [x]") :].strip()
    if tag.startswith("-"):
        tag = tag[1:].strip()
    return tag


if __name__ == "__main__":
    raise SystemExit(main())
