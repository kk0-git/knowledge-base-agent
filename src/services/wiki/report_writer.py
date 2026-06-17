from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPORT_FILENAME = "_Wiki Report.md"


def write_obsidian_wiki_report(
    *,
    report: dict[str, Any],
    wiki_dir: Path,
    report_path: Path | None = None,
    sync_result: dict[str, Any] | None = None,
) -> Path:
    output_path = report_path or (wiki_dir / DEFAULT_REPORT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_obsidian_wiki_report(report=report, sync_result=sync_result),
        encoding="utf-8",
    )
    return output_path


def render_obsidian_wiki_report(
    *,
    report: dict[str, Any],
    sync_result: dict[str, Any] | None = None,
) -> str:
    rows = [
        row
        for row in report.get("tag_rows", [])
        if row.get("eligible") or row.get("wiki_exists")
    ]
    dirty_rows = [
        row for row in rows
        if row.get("dirty") and row.get("wiki_exists")
    ]
    missing_rows = [
        row for row in rows
        if row.get("eligible") and not row.get("wiki_exists")
    ]
    generated_rows = [
        row for row in rows
        if row.get("wiki_exists") and not row.get("dirty")
    ]
    failed_rows = [
        row for row in rows
        if row.get("last_error")
    ]

    lines = [
        "# Wiki 状态",
        "",
        f"更新时间：{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 需要更新：`{len(dirty_rows)}`",
        f"- 合成失败：`{len(failed_rows)}`",
        f"- 尚未合成：`{len(missing_rows)}`",
        f"- 已生成：`{len(generated_rows)}`",
        "",
    ]

    if failed_rows:
        lines.extend(render_failed_section("合成失败", failed_rows))
    if dirty_rows:
        lines.extend(render_section("需要更新", dirty_rows, status="源笔记或主题归属已变化，建议重新合成。"))
    if missing_rows:
        lines.extend(render_section("尚未合成", missing_rows, status="尚未生成 wiki。"))
    if generated_rows:
        lines.extend(render_section("已生成", generated_rows, status=None))

    if sync_result:
        lines.extend(render_sync_result(sync_result))

    lines.extend(
        [
            "## 使用方式",
            "",
            "在上方复制 tag，使用 Obsidian Shell Commands 触发 `workspace_sync.py update-topic` 刷新对应 wiki。",
            "后台同步只更新 tag、RAG index 和本报告，不会自动改写 wiki 正文。",
            "",
        ]
    )
    return "\n".join(lines)


def render_failed_section(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    for row in sorted(rows, key=lambda item: (-int(item.get("retry_count", 0)), str(item.get("tag", "")))):
        tag = str(row.get("tag", ""))
        error_type = str(row.get("last_error_type") or "Error")
        message = str(row.get("last_error") or "")
        retryable = "是" if row.get("retryable") else "否"
        error_at = str(row.get("last_error_at") or "")
        retry_count = int(row.get("retry_count", 0))
        lines.append(f"- [ ] `{tag}`")
        lines.append(f"  - {error_type}: {message}")
        lines.append(f"  - 可重试：{retryable}；失败次数：`{retry_count}`")
        if error_at:
            lines.append(f"  - 时间：`{error_at}`")
        lines.append("")
    return lines


def render_section(title: str, rows: list[dict[str, Any]], *, status: str | None) -> list[str]:
    lines = [f"## {title}", ""]
    for row in sorted(rows, key=lambda item: (-int(item.get("note_count", 0)), str(item.get("tag", "")))):
        tag = str(row.get("tag", ""))
        note_count = int(row.get("note_count", 0))
        wiki_path = str(row.get("wiki_path") or "")
        lines.append(f"- [ ] `{tag}` · {note_count} 篇笔记")
        if wiki_path:
            lines.append(f"  - [[{wiki_path}]]")
        if status:
            lines.append(f"  - {status}")
        lines.append("")
    return lines


def render_sync_result(sync_result: dict[str, Any]) -> list[str]:
    lines = ["## 最近同步", ""]
    workspace = sync_result.get("workspace", {})
    if workspace:
        lines.append(
            "- Workspace: "
            f"files=`{workspace.get('files', '-')}`, "
            f"embed_dirty=`{workspace.get('embed_dirty_files', '-')}`, "
            f"tags_dirty=`{workspace.get('tags_dirty_files', '-')}`, "
            f"deleted=`{workspace.get('deleted_files', '-')}`"
        )
    rag = sync_result.get("rag")
    wiki = sync_result.get("wiki", {})
    if rag:
        lines.append(
            "- RAG: "
            f"mode=`{rag.get('mode', '-')}`, "
            f"added=`{rag.get('added_files', '-')}`, "
            f"modified=`{rag.get('modified_files', '-')}`, "
            f"deleted=`{rag.get('deleted_files', '-')}`, "
            f"embedded_chunks=`{rag.get('embedded_chunks', '-')}`"
        )
    else:
        lines.append("- RAG: skipped")
    lines.append(
        "- Wiki tags: "
        f"tagged=`{wiki.get('tagged', '-')}`, "
        f"skipped=`{wiki.get('skipped', '-')}`, "
        f"failed=`{wiki.get('failed', '-')}`, "
        f"deleted=`{wiki.get('deleted', '-')}`"
    )
    failed_tags = wiki.get("failed_tags") or []
    retried_tags = wiki.get("retried_tags") or []
    if retried_tags:
        lines.append("- Wiki synthesis retried and succeeded:")
        for item in retried_tags[:10]:
            lines.append(
                f"  - `{item.get('tag', '-')}` "
                f"attempts=`{item.get('attempts', '-')}`"
            )
    if failed_tags:
        lines.append("- Failed wiki synthesis:")
        for item in failed_tags[:10]:
            lines.append(
                f"  - `{item.get('tag', '-')}` "
                f"{item.get('error_type', 'Error')}: {item.get('message', '')} "
                f"(attempts=`{item.get('attempts', '-')}`, "
                f"retryable=`{item.get('retryable', False)}`)"
            )
    lines.append("")
    return lines
