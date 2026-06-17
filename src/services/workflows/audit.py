from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.wiki.manager import rebuild_tag_index
from services.wiki.state_store import WikiStateStore
from services.workflows.schema import ScopeResult


HEADING_RE = re.compile(r"^(#{1,6})\s*(.*?)\s*$")
ABSOLUTE_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([A-Za-z]:\\|/Users/|/home/)[^)]+\)")
WIKILINK_RE = re.compile(r"!?\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    line: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


def run_deterministic_audit(
    *,
    vault_root: Path,
    scope_result: ScopeResult,
    wiki_state_store: WikiStateStore | None = None,
    wiki_dir: Path | None = None,
    overview_note_threshold: int = 30,
    min_note_chars: int = 80,
    max_issues: int = 200,
) -> dict[str, Any]:
    issues: list[AuditIssue] = []
    note_paths = [str(note.get("path", "")) for note in scope_result.notes if note.get("path")]
    existing_links = build_existing_link_index(vault_root)

    for note_path in note_paths:
        if len(issues) >= max_issues:
            break
        full_path = vault_root / note_path
        if not full_path.exists():
            issues.append(
                AuditIssue(
                    severity="error",
                    code="missing_note",
                    message="Scope references a note that no longer exists.",
                    path=note_path,
                )
            )
            continue
        text = full_path.read_text(encoding="utf-8", errors="replace")
        issues.extend(audit_note_text(note_path=note_path, text=text, min_note_chars=min_note_chars))
        issues.extend(
            audit_note_links(
                vault_root=vault_root,
                note_path=note_path,
                text=text,
                existing_links=existing_links,
            )
        )

    if wiki_state_store is not None:
        issues.extend(
            audit_wiki_state(
                wiki_state_store=wiki_state_store,
                wiki_dir=wiki_dir,
                vault_root=vault_root,
                scope_note_paths=set(note_paths),
                overview_note_threshold=overview_note_threshold,
            )
        )

    issues = issues[:max_issues]
    counts = Counter(issue.severity for issue in issues)
    by_code = Counter(issue.code for issue in issues)
    return {
        "summary": {
            "notes_checked": len(note_paths),
            "issues": len(issues),
            "errors": counts.get("error", 0),
            "warnings": counts.get("warning", 0),
            "info": counts.get("info", 0),
            "by_code": dict(sorted(by_code.items())),
        },
        "issues": [audit_issue_to_dict(issue) for issue in issues],
        "markdown": render_audit_markdown(issues, note_count=len(note_paths)),
    }


def audit_note_text(*, note_path: str, text: str, min_note_chars: int) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    stripped = text.strip()
    if len(stripped) < min_note_chars:
        issues.append(
            AuditIssue(
                severity="warning",
                code="short_note",
                message=f"Note is very short ({len(stripped)} chars).",
                path=note_path,
                evidence={"chars": len(stripped)},
            )
        )

    lines = text.splitlines()
    heading_lines: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            heading_lines.append((index, len(match.group(1)), match.group(2).strip()))
        if ABSOLUTE_IMAGE_RE.search(line):
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="absolute_image_path",
                    message="Image link uses an absolute local path and may not work across machines.",
                    path=note_path,
                    line=index,
                )
            )

    heading_counts = Counter(heading for _, _, heading in heading_lines if heading)
    for heading, count in heading_counts.items():
        if count > 1:
            first_line = next(line_no for line_no, _, value in heading_lines if value == heading)
            issues.append(
                AuditIssue(
                    severity="info",
                    code="duplicate_heading",
                    message=f"Heading appears {count} times in the same note.",
                    path=note_path,
                    line=first_line,
                    evidence={"heading": heading, "count": count},
                )
            )

    for idx, (line_no, level, heading) in enumerate(heading_lines):
        next_heading = heading_lines[idx + 1] if idx + 1 < len(heading_lines) else None
        next_line = next_heading[0] if next_heading else len(lines) + 1
        section_lines = lines[line_no: next_line - 1]
        non_empty = [line.strip() for line in section_lines if line.strip()]
        if not heading:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="empty_heading_text",
                    message="Heading marker has no title text.",
                    path=note_path,
                    line=line_no,
                    evidence={"level": level},
                )
            )
        elif not non_empty and not heading_has_child(next_heading=next_heading, level=level):
            issues.append(
                AuditIssue(
                    severity="info",
                    code="heading_without_body",
                    message="Leaf heading has no body before the next sibling or parent heading.",
                    path=note_path,
                    line=line_no,
                    evidence={"heading": heading},
                )
            )

    return issues


def heading_has_child(*, next_heading: tuple[int, int, str] | None, level: int) -> bool:
    if next_heading is None:
        return False
    return next_heading[1] > level


def build_existing_link_index(vault_root: Path) -> set[str]:
    existing: set[str] = set()
    for path in vault_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(vault_root).as_posix()
        except ValueError:
            continue
        existing.add(rel)
        existing.add(rel.removesuffix(".md"))
        existing.add(path.name)
        existing.add(path.stem)
    return existing


def audit_note_links(
    *,
    vault_root: Path,
    note_path: str,
    text: str,
    existing_links: set[str],
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    lines = text.splitlines()
    note_dir = Path(note_path).parent

    for index, line in enumerate(lines, start=1):
        for match in WIKILINK_RE.finditer(line):
            target = normalize_wikilink_target(match.group(1))
            if not target or target.startswith("#"):
                continue
            if not link_target_exists(target, existing_links):
                issues.append(
                    AuditIssue(
                        severity="warning",
                        code="broken_wikilink",
                        message="Obsidian wikilink target does not resolve to a scanned note or attachment.",
                        path=note_path,
                        line=index,
                        evidence={"target": target},
                    )
                )

        for match in MARKDOWN_LINK_RE.finditer(line):
            target = normalize_markdown_link_target(match.group(1))
            if not target or should_skip_markdown_link(target):
                continue
            if re.match(r"^[A-Za-z]:\\|^/Users/|^/home/", target):
                continue
            candidate = (note_dir / target).as_posix()
            if not link_target_exists(candidate, existing_links) and not link_target_exists(target, existing_links):
                issues.append(
                    AuditIssue(
                        severity="warning",
                        code="broken_markdown_link",
                        message="Relative Markdown link target does not resolve to a scanned note or attachment.",
                        path=note_path,
                        line=index,
                        evidence={"target": target},
                    )
                )

    return issues


def normalize_wikilink_target(raw_target: str) -> str:
    target = raw_target.strip().replace("\\", "/")
    target = target.split("#", 1)[0].split("^", 1)[0]
    return target.strip()


def normalize_markdown_link_target(raw_target: str) -> str:
    target = raw_target.strip().strip("<>").replace("\\", "/")
    target = target.split("#", 1)[0]
    return target.strip()


def should_skip_markdown_link(target: str) -> bool:
    lowered = target.lower()
    return (
        not target
        or target.startswith("#")
        or lowered.startswith(("http://", "https://", "mailto:", "obsidian://", "data:"))
    )


def link_target_exists(target: str, existing_links: set[str]) -> bool:
    cleaned = target.strip().replace("\\", "/").strip("/")
    if not cleaned:
        return True
    candidates = {
        cleaned,
        cleaned.removesuffix(".md"),
        f"{cleaned}.md",
        Path(cleaned).name,
        Path(cleaned).stem,
    }
    return bool(candidates & existing_links)


def audit_wiki_state(
    *,
    wiki_state_store: WikiStateStore,
    wiki_dir: Path | None,
    vault_root: Path,
    scope_note_paths: set[str],
    overview_note_threshold: int,
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    state = rebuild_tag_index(wiki_state_store.load(), overview_note_threshold=overview_note_threshold)
    for tag, record in sorted(state.tags.items()):
        if scope_note_paths and not (set(record.source_paths) & scope_note_paths):
            continue
        if record.last_error:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="wiki_synthesis_failed",
                    message=f"Wiki synthesis failed: {record.last_error_type}: {record.last_error}",
                    path=record.wiki_path,
                    evidence={"tag": tag, "retry_count": record.retry_count},
                )
            )
        if record.wiki_policy != "skip" and not record.wiki_path:
            issues.append(
                AuditIssue(
                    severity="info",
                    code="wiki_not_generated",
                    message="Eligible tag has no generated wiki path recorded.",
                    evidence={"tag": tag, "notes": len(record.source_paths)},
                )
            )
        if record.wiki_policy == "generate" and len(record.source_paths) >= overview_note_threshold:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="large_generate_topic",
                    message="Large topic is still using generate policy; overview may be more appropriate.",
                    path=record.wiki_path,
                    evidence={"tag": tag, "notes": len(record.source_paths)},
                )
            )
        if wiki_dir and record.wiki_path:
            wiki_path = (vault_root / record.wiki_path) if not Path(record.wiki_path).is_absolute() else Path(record.wiki_path)
            if record.wiki_policy != "skip" and not wiki_path.exists():
                issues.append(
                    AuditIssue(
                        severity="warning",
                        code="wiki_file_missing",
                        message="Wiki path is recorded but the file does not exist.",
                        path=record.wiki_path,
                        evidence={"tag": tag},
                    )
                )
    return issues


def audit_issue_to_dict(issue: AuditIssue) -> dict[str, Any]:
    return {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
        "path": issue.path,
        "line": issue.line,
        "evidence": issue.evidence,
    }


def render_audit_markdown(issues: list[AuditIssue], *, note_count: int) -> str:
    counts = Counter(issue.severity for issue in issues)
    lines = [
        "# Knowledge Audit Report",
        "",
        f"- Notes checked: `{note_count}`",
        f"- Issues: `{len(issues)}`",
        f"- Errors: `{counts.get('error', 0)}`",
        f"- Warnings: `{counts.get('warning', 0)}`",
        f"- Info: `{counts.get('info', 0)}`",
        "",
    ]
    if not issues:
        lines.extend(["No issues found.", ""])
        return "\n".join(lines)

    for severity in ("error", "warning", "info"):
        group = [issue for issue in issues if issue.severity == severity]
        if not group:
            continue
        lines.extend([f"## {severity.title()}", ""])
        for issue in group:
            location = issue.path or "(state)"
            if issue.line is not None:
                location += f":{issue.line}"
            lines.append(f"- `{issue.code}` {location} - {issue.message}")
        lines.append("")
    return "\n".join(lines)
