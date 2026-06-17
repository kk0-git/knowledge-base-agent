from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.wiki.manager import WikiManager
from services.wiki.state_store import WikiStateStore
from services.workflows.context_builder import ContextBuilder
from services.workflows.runner import WorkflowRunner, task_result_to_dict
from services.workflows.schema import ScopeSpec, WorkflowSpec, WritebackSpec
from services.workflows.scope_resolver import ScopeResolver


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug unified knowledge workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-scope", help="Resolve scope and build context")
    add_common_args(inspect_parser)
    inspect_parser.add_argument("--scope-type", required=True, choices=["tag", "folder", "all_vault", "selected_notes"])
    inspect_parser.add_argument("--value", default="")
    inspect_parser.add_argument("--paths", nargs="*", default=[])
    inspect_parser.add_argument("--context-mode", default="wiki_context")

    wiki_parser = subparsers.add_parser("synthesize-wiki", help="Run synthesize_wiki workflow for one tag")
    add_common_args(wiki_parser)
    wiki_parser.add_argument("--tag", required=True)
    wiki_parser.add_argument("--force", action="store_true")

    audit_parser = subparsers.add_parser("audit", help="Run deterministic audit workflow")
    add_common_args(audit_parser)
    audit_parser.add_argument("--scope-type", required=True, choices=["tag", "folder", "all_vault", "selected_notes"])
    audit_parser.add_argument("--value", default="")
    audit_parser.add_argument("--paths", nargs="*", default=[])
    audit_parser.add_argument("--max-issues", type=int, default=50)

    review_parser = subparsers.add_parser("review-notes", help="Run LLM organize_suggestions workflow")
    add_common_args(review_parser)
    review_parser.add_argument("--scope-type", required=True, choices=["tag", "folder", "all_vault", "selected_notes"])
    review_parser.add_argument("--value", default="")
    review_parser.add_argument("--paths", nargs="*", default=[])
    review_parser.add_argument("--max-notes", type=int, default=8)
    review_parser.add_argument("--max-chars-per-note", type=int, default=1800)
    review_parser.add_argument("--review-mode", choices=["auto", "topic", "notes"], default="auto")

    organize_parser = subparsers.add_parser("organize", help="Run unified organize workflow: audit + LLM review")
    add_common_args(organize_parser)
    organize_parser.add_argument("--scope-type", required=True, choices=["tag", "folder", "all_vault", "selected_notes"])
    organize_parser.add_argument("--value", default="")
    organize_parser.add_argument("--paths", nargs="*", default=[])
    organize_parser.add_argument("--max-issues", type=int, default=50)
    organize_parser.add_argument("--max-notes", type=int, default=8)
    organize_parser.add_argument("--max-chars-per-note", type=int, default=1800)
    organize_parser.add_argument("--review-mode", choices=["auto", "topic", "notes"], default="auto")

    args = parser.parse_args()

    if args.command == "inspect-scope":
        resolver = build_scope_resolver(args)
        context_builder = ContextBuilder(vault_root=Path(args.vault))
        scope = ScopeSpec(
            type=args.scope_type,
            value=args.value or None,
            paths=tuple(args.paths or ()),
        )
        scope_result = resolver.resolve(scope)
        context = context_builder.build(scope_result, mode=args.context_mode)
        print_json(
            {
                "scope": {
                    "notes": len(scope_result.notes),
                    "chunks": len(scope_result.chunks),
                    "metadata": scope_result.metadata,
                },
                "context": {
                    "mode": context.mode,
                    "items": len(context.items),
                    "chars": len(context.context_text),
                    "stats": context.stats,
                },
                "sample_items": list(context.items[:3]),
            }
        )
        return 0

    if args.command == "synthesize-wiki":
        runner = build_runner(args)
        result = runner.run(
            WorkflowSpec(
                task_type="synthesize_wiki",
                scope=ScopeSpec(type="tag", value=args.tag),
                user_request=f"Generate wiki for {args.tag}",
                writeback=WritebackSpec(type="wiki_file"),
                options={"force": args.force},
            )
        )
        print_json(task_result_to_dict(result))
        return 0

    if args.command == "audit":
        runner = build_runner(args, task_type="audit")
        result = runner.run(
            WorkflowSpec(
                task_type="audit",
                scope=ScopeSpec(
                    type=args.scope_type,
                    value=args.value or None,
                    paths=tuple(args.paths or ()),
                ),
                user_request="Audit knowledge scope",
                writeback=WritebackSpec(type="none"),
                options={"max_issues": args.max_issues},
            )
        )
        print_json(task_result_to_dict(result))
        return 0

    if args.command == "review-notes":
        runner = build_runner(args, task_type="organize_suggestions")
        result = runner.run(
            WorkflowSpec(
                task_type="organize_suggestions",
                scope=ScopeSpec(
                    type=args.scope_type,
                    value=args.value or None,
                    paths=tuple(args.paths or ()),
                ),
                user_request="Review notes and suggest organization improvements",
                context_mode="suggestion_context",
                writeback=WritebackSpec(type="none"),
                options={
                    "max_notes": args.max_notes,
                    "max_chars_per_note": args.max_chars_per_note,
                    "review_mode": args.review_mode,
                },
            )
        )
        payload = task_result_to_dict(result)
        json_path, md_path = write_review_outputs(payload, args)
        suggestions = payload.get("output", {}).get("organize_suggestions", {}).get("suggestions", {})
        print(
            "\n".join(
                [
                    "OK: review suggestions generated.",
                    f"Summary: {suggestions.get('summary', '')}",
                    f"JSON: {json_path}",
                    f"Markdown: {md_path}",
                ]
            )
        )
        return 0

    if args.command == "organize":
        runner = build_runner(args, task_type="organize")
        result = runner.run(
            WorkflowSpec(
                task_type="organize",
                scope=ScopeSpec(
                    type=args.scope_type,
                    value=args.value or None,
                    paths=tuple(args.paths or ()),
                ),
                user_request="Organize knowledge scope",
                context_mode="organize_context",
                writeback=WritebackSpec(type="none"),
                options={
                    "max_issues": args.max_issues,
                    "max_notes": args.max_notes,
                    "max_chars_per_note": args.max_chars_per_note,
                    "review_mode": args.review_mode,
                },
            )
        )
        payload = task_result_to_dict(result)
        json_path, md_path = write_organize_outputs(payload, args)
        organize = payload.get("output", {}).get("organize", {})
        summary = organize.get("summary", {})
        review = organize.get("review", {})
        suggestions = review.get("suggestions", {})
        print(
            "\n".join(
                [
                    "OK: organize workflow completed.",
                    f"Issues: {summary.get('issues', 0)}",
                    f"Review: {suggestions.get('summary', '')}",
                    f"JSON: {json_path}",
                    f"Markdown: {md_path}",
                ]
            )
        )
        return 0

    return 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vault", required=True)
    parser.add_argument("--state", default="./wiki-state/wiki_state.json")
    parser.add_argument("--wiki-dir", default=None)
    parser.add_argument("--min-notes-per-tag", type=int, default=3)
    parser.add_argument("--overview-note-threshold", type=int, default=30)


def build_scope_resolver(args: argparse.Namespace) -> ScopeResolver:
    return ScopeResolver(
        vault_root=Path(args.vault),
        wiki_state_store=WikiStateStore(Path(args.state)),
        wiki_dir=Path(args.wiki_dir) if args.wiki_dir else None,
        overview_note_threshold=args.overview_note_threshold,
    )


def build_runner(args: argparse.Namespace, *, task_type: str = "synthesize_wiki") -> WorkflowRunner:
    vault_root = Path(args.vault)
    state_store = WikiStateStore(Path(args.state))
    wiki_manager = None
    llm_client = None
    llm_model = None
    llm_temperature = 0.2
    if task_type in {"synthesize_wiki", "organize_suggestions", "generate_review", "organize"}:
        llm_config = load_llm_config(PROJECT_ROOT)
        llm_client = create_llm_client(llm_config)
        llm_model = llm_config.model
        llm_temperature = llm_config.temperature
    if task_type == "synthesize_wiki":
        wiki_manager = WikiManager(
            vault_root=vault_root,
            state_store=state_store,
            llm_client=llm_client,
            llm_model=llm_model,
            wiki_dir=Path(args.wiki_dir) if args.wiki_dir else None,
            min_notes_per_tag=args.min_notes_per_tag,
            overview_note_threshold=args.overview_note_threshold,
        )
    return WorkflowRunner(
        scope_resolver=ScopeResolver(
            vault_root=vault_root,
            wiki_state_store=state_store,
            wiki_dir=Path(args.wiki_dir) if args.wiki_dir else None,
            overview_note_threshold=args.overview_note_threshold,
        ),
        context_builder=ContextBuilder(vault_root=vault_root),
        wiki_manager=wiki_manager,
        vault_root=vault_root,
        wiki_state_store=state_store,
        wiki_dir=Path(args.wiki_dir) if args.wiki_dir else None,
        overview_note_threshold=args.overview_note_threshold,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
    )


def print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def write_review_outputs(payload: dict, args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = PROJECT_ROOT / "eval-results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = default_review_output_stem(args)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_review_markdown(payload), encoding="utf-8")
    return json_path, md_path


def write_organize_outputs(payload: dict, args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = PROJECT_ROOT / "eval-results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = default_organize_output_stem(args)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_organize_markdown(payload), encoding="utf-8")
    return json_path, md_path


def default_organize_output_stem(args: argparse.Namespace) -> str:
    if args.scope_type == "selected_notes":
        raw = "-".join(Path(path).stem for path in args.paths[:3]) or "selected-notes"
    elif args.scope_type == "all_vault":
        raw = "all-vault"
    else:
        raw = args.value or args.scope_type
    return "organize-" + sanitize_filename(f"{args.scope_type}-{raw}")


def default_review_output_stem(args: argparse.Namespace) -> str:
    mode = args.review_mode
    if mode == "auto":
        mode = "topic" if args.scope_type == "tag" else "notes"
    if args.scope_type == "selected_notes":
        raw = "-".join(Path(path).stem for path in args.paths[:3]) or "selected-notes"
    elif args.scope_type == "all_vault":
        raw = "all-vault"
    else:
        raw = args.value or args.scope_type
    return f"review-{mode}-" + sanitize_filename(f"{args.scope_type}-{raw}")


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "untitled"


def render_review_markdown(payload: dict) -> str:
    workflow = payload.get("output", {}).get("organize_suggestions", {})
    packet = workflow.get("packet", {})
    suggestions = workflow.get("suggestions", {})
    validation = workflow.get("validation") or suggestions.get("_validation") or {}
    scope = packet.get("scope", {})
    notes = packet.get("notes", [])
    review_mode = workflow.get("review_mode") or packet.get("review_mode") or "notes"
    title = "Topic Review" if review_mode == "topic" else "Notes Review"

    lines: list[str] = [
        f"# {title}",
        "",
        f"- Review mode: `{review_mode}`",
        f"- Scope: `{scope.get('type', '')}` `{scope.get('value') or ''}`",
        f"- Notes reviewed: `{len(notes)}`",
        f"- Workflow ms: `{payload.get('timing', {}).get('workflow_ms', '')}`",
        "",
    ]

    summary = suggestions.get("summary")
    if summary:
        lines.extend(["## Summary", "", str(summary), ""])

    if validation:
        corrections = validation.get("corrections") or []
        warnings = validation.get("warnings") or []
        if corrections or warnings:
            lines.extend(["## Validation", ""])
            if corrections:
                lines.append("### Auto-corrections")
                lines.append("")
                for item in corrections:
                    lines.append(f"- `{item.get('field', '')}`: `{item.get('from', '')}` -> `{item.get('to', '')}`")
                lines.append("")
            if warnings:
                lines.append("### Needs Manual Check")
                lines.append("")
                for item in warnings:
                    if item.get("path"):
                        lines.append(f"- `{item.get('type', '')}` at `{item.get('field', '')}`: `{item.get('path', '')}`")
                    else:
                        lines.append(f"- `{item.get('type', '')}` at `{item.get('field', '')}`")
                lines.append("")

    topic_structure = suggestions.get("topic_structure") or {}
    if topic_structure:
        lines.extend(["## Topic Structure", ""])
        if topic_structure.get("coverage"):
            lines.append(f"- Coverage: {topic_structure.get('coverage')}")
        if topic_structure.get("core_notes"):
            lines.append("- Core notes: " + ", ".join(f"`{path}`" for path in topic_structure.get("core_notes", [])))
        if topic_structure.get("edge_notes"):
            lines.append("- Edge notes: " + ", ".join(f"`{path}`" for path in topic_structure.get("edge_notes", [])))
        if topic_structure.get("missing_parts"):
            lines.append("- Missing parts: " + ", ".join(str(item) for item in topic_structure.get("missing_parts", [])))
        lines.append("")

    next_actions = suggestions.get("next_actions") or []
    if next_actions and review_mode != "topic":
        lines.extend(["## Next Actions", ""])
        for action in next_actions:
            lines.append(f"- {action}")
        lines.append("")

    note_reviews = suggestions.get("note_reviews") or []
    if note_reviews:
        lines.extend(["## Note Reviews", ""])
        for item in note_reviews:
            lines.extend(
                [
                    f"### {item.get('path', '(unknown)')}",
                    "",
                    f"- Role: `{item.get('role', '')}`",
                    f"- Action: `{item.get('recommended_action', '')}`",
                    f"- Risk: `{item.get('risk', '')}`",
                    f"- Reason: {item.get('reason', '')}",
                ]
            )
            tags = item.get("suggested_tags") or []
            links = item.get("suggested_links") or []
            if tags:
                lines.append("- Suggested tags: " + ", ".join(f"`{tag}`" for tag in tags))
            if links:
                lines.append("- Suggested links: " + ", ".join(f"`{link}`" for link in links))
            lines.append("")

    relationships = suggestions.get("relationship_suggestions") or []
    if relationships:
        lines.extend(["## Relationship Suggestions", ""])
        for item in relationships:
            lines.extend(
                [
                    f"- `{item.get('source', '')}` -> `{item.get('target', '')}`",
                    f"  - Relationship: `{item.get('relationship', '')}`",
                    f"  - Action: `{item.get('recommended_action', '')}`",
                    f"  - Reason: {item.get('reason', '')}",
                ]
            )
        lines.append("")

    topics = suggestions.get("topic_suggestions") or []
    if topics:
        heading = "Wiki Suggestions" if review_mode == "topic" else "Topic Suggestions"
        lines.extend([f"## {heading}", ""])
        for item in topics:
            lines.extend(
                [
                    f"### {item.get('topic', '(untitled)')}",
                    "",
                    f"- Suggested output: `{item.get('suggested_output', '')}`",
                    f"- Reason: {item.get('reason', '')}",
                    "- Notes:",
                ]
            )
            for note in item.get("notes") or []:
                lines.append(f"  - `{note}`")
            lines.append("")

    questions = suggestions.get("review_questions") or []
    if questions:
        lines.extend(["## Review Questions", ""])
        for item in questions:
            lines.extend(
                [
                    f"- {item.get('question', '')}",
                    f"  - Source notes: {', '.join(f'`{note}`' for note in (item.get('source_notes') or []))}",
                    f"  - Reason: {item.get('reason', '')}",
                ]
            )
        lines.append("")

    if next_actions and review_mode == "topic":
        lines.extend(["## Next Actions", ""])
        for action in next_actions:
            lines.append(f"- {action}")
        lines.append("")

    if notes:
        lines.extend(["## Reviewed Notes", ""])
        for note in notes:
            signals = note.get("signals") or []
            lines.append(f"- `{note.get('path', '')}` ({note.get('chars', 0)} chars)")
            if signals:
                lines.append("  - Signals: " + ", ".join(f"`{signal}`" for signal in signals))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_organize_markdown(payload: dict) -> str:
    workflow = payload.get("output", {}).get("organize", {})
    summary = workflow.get("summary", {})
    audit = workflow.get("audit", {})
    review = workflow.get("review", {})
    review_payload = {
        **payload,
        "output": {"organize_suggestions": review},
    }
    audit_summary = audit.get("summary", {})
    suggestions = review.get("suggestions", {})
    validation = review.get("validation") or suggestions.get("_validation") or {}

    lines: list[str] = [
        "# Knowledge Organization Report",
        "",
        "- Workflow: `organize`",
        f"- Notes: `{summary.get('notes', 0)}`",
        f"- Issues: `{summary.get('issues', 0)}`",
        f"- Review mode: `{summary.get('review_mode', '')}`",
        f"- Workflow ms: `{payload.get('timing', {}).get('workflow_ms', '')}`",
        "",
    ]

    if suggestions.get("summary"):
        lines.extend(["## 整理结论", "", str(suggestions.get("summary")), ""])

    lines.extend(
        [
            "## 结构检查",
            "",
            f"- Notes checked: `{audit_summary.get('notes_checked', 0)}`",
            f"- Issues: `{audit_summary.get('issues', 0)}`",
            f"- Errors: `{audit_summary.get('errors', 0)}`",
            f"- Warnings: `{audit_summary.get('warnings', 0)}`",
            f"- Info: `{audit_summary.get('info', 0)}`",
            "",
        ]
    )

    issues = audit.get("issues") or []
    if issues:
        for issue in issues[:30]:
            line = f":{issue.get('line')}" if issue.get("line") else ""
            lines.append(
                f"- `{issue.get('severity', '')}` `{issue.get('code', '')}` "
                f"{issue.get('path', '')}{line} - {issue.get('message', '')}"
            )
        if len(issues) > 30:
            lines.append(f"- ... {len(issues) - 30} more issues omitted")
        lines.append("")
    else:
        lines.extend(["没有发现确定性结构问题。", ""])

    if validation:
        corrections = validation.get("corrections") or []
        warnings = validation.get("warnings") or []
        if corrections or warnings:
            lines.extend(["## 建议校验", ""])
            for item in corrections:
                lines.append(f"- Auto-correct `{item.get('field', '')}`: `{item.get('from', '')}` -> `{item.get('to', '')}`")
            for item in warnings:
                if item.get("path"):
                    lines.append(f"- Check `{item.get('type', '')}` at `{item.get('field', '')}`: `{item.get('path', '')}`")
                else:
                    lines.append(f"- Check `{item.get('type', '')}` at `{item.get('field', '')}`")
            lines.append("")

    lines.extend(["## 整理建议", ""])
    rendered_review = render_review_markdown(review_payload)
    review_lines = rendered_review.splitlines()
    if review_lines and review_lines[0].startswith("# "):
        review_lines = review_lines[1:]
    lines.extend(review_lines)
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
