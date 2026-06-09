from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.wiki.manager import WikiManager, import_manual_policies, rebuild_tag_index, set_tag_policy
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
    add_common_args(subparsers.add_parser("rebuild", help="Tag notes and generate wiki pages"))
    add_common_args(subparsers.add_parser("report", help="Print wiki state report"))
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
        tag=args.tag,
    )


def run_set_policy(args: argparse.Namespace) -> dict:
    store = WikiStateStore(Path(args.state))
    state = store.load()
    updated = set_tag_policy(state, tag=args.tag, wiki_policy=args.policy)
    store.save(updated)
    payload = {
        "tag": args.tag,
        "wiki_policy": args.policy,
        "dirty": updated.tags[args.tag].dirty,
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


if __name__ == "__main__":
    raise SystemExit(main())
