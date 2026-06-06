from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from knowledge_base_agent.graph import KnowledgeGraph
from knowledge_base_agent.profile import NoteProfile


@dataclass(frozen=True)
class Candidate:
    kind: str
    path: str | None
    target: str | None
    priority: str
    reason: str

@dataclass(frozen=True)
class CandidateReport:
    orphan_notes: list[Candidate] = field(default_factory=list)
    weakly_linked_notes: list[Candidate] = field(default_factory=list)
    hub_candidates: list[Candidate] = field(default_factory=list)
    unresolved_links: list[Candidate] = field(default_factory=list)
    missing_tag_notes: list[Candidate] = field(default_factory=list)
    temporary_notes: list[Candidate] = field(default_factory=list)
    troubleshooting_notes: list[Candidate] = field(default_factory=list)

def build_candidate_report(
    profiles: list[NoteProfile],
    graph: KnowledgeGraph,
    hub_threshold: int = 5,
) -> CandidateReport:
    """基于笔记档案和知识图谱构建优化候选项报告。"""
    profile_by_path = {profile.path: profile for profile in profiles}
    degree_by_path = {
        node.path: node.in_degree + node.out_degree
        for node in graph.nodes
    }

    orphan_notes = [
        Candidate(
            kind="orphan_note",
            path=path,
            target=None,
            priority="medium",
            reason="No incoming or outgoing wikilinks.",
        )
        for path in graph.orphan_notes
    ]

    weakly_linked_notes = [
        Candidate(
            kind="weakly_linked_note",
            path=path,
            target=None,
            priority="low",
            reason="Only one or fewer total wikilink connections.",
        )
        for path, degree in sorted(degree_by_path.items(), key=lambda item: item[1])
        if degree <= 1 and path not in graph.orphan_notes
    ]

    hub_candidates = [
        Candidate(
            kind="hub_candidate",
            path=node.path,
            target=None,
            priority="low",
            reason=(
                f"High connection count: "
                f"{node.in_degree} incoming, {node.out_degree} outgoing."
            ),
        )
        for node in sorted(
            graph.nodes,
            key=lambda item: item.in_degree + item.out_degree,
            reverse=True,
        )
        if node.in_degree + node.out_degree >= hub_threshold
    ]

    unresolved_links = [
        Candidate(
            kind="unresolved_link",
            path=None,
            target=target,
            priority="high",
            reason="Wikilink target does not resolve to a scanned note.",
        )
        for target in graph.unresolved_links
    ]

    missing_tag_notes = [
        Candidate(
            kind="missing_tag",
            path=profile.path,
            target=None,
            priority="low",
            reason="Note has no tags.",
        )
        for profile in profiles
        if not profile.tags
    ]

    temporary_notes = [
        Candidate(
            kind="temporary_note",
            path=profile.path,
            target=None,
            priority="medium",
            reason="Note looks like inbox, temporary, draft, or scratch content.",
        )
        for profile in profiles
        if is_temporary_note(profile)
    ]

    troubleshooting_notes = [
        Candidate(
            kind="troubleshooting_note",
            path=profile.path,
            target=None,
            priority="medium",
            reason="Note looks like a problem solving or troubleshooting record.",
        )
        for profile in profiles
        if is_troubleshooting_note(profile)
    ]

    return CandidateReport(
        orphan_notes=orphan_notes,
        weakly_linked_notes=weakly_linked_notes,
        hub_candidates=hub_candidates,
        unresolved_links=unresolved_links,
        missing_tag_notes=missing_tag_notes,
        temporary_notes=temporary_notes,
        troubleshooting_notes=troubleshooting_notes,
    )


def is_temporary_note(profile: NoteProfile) -> bool:
    path = profile.path.lower()
    title = profile.title.lower()
    tags = {tag.lower() for tag in profile.tags}

    path_markers = [
        "inbox/",
        "temp/",
        "temporary/",
        "draft/",
        "草稿/",
        "临时/",
        "收件箱/",
    ]

    title_markers = [
        "todo",
        "draft",
        "temp",
        "临时",
        "草稿",
        "待整理",
    ]

    tag_markers = {
        "todo",
        "draft",
        "temp",
        "temporary",
        "inbox",
        "待整理",
        "临时",
    }

    return (
        any(marker in path for marker in path_markers)
        or any(marker in title for marker in title_markers)
        or bool(tags & tag_markers)
        or profile.note_type == "temporary"
    )


def is_troubleshooting_note(profile: NoteProfile) -> bool:
    path = profile.path.lower()
    title = profile.title.lower()
    tags = {tag.lower() for tag in profile.tags}
    excerpt = profile.excerpt.lower()

    path_markers = [
        "troubleshooting/",
        "debug/",
        "bug/",
        "error/",
        "问题/",
        "报错/",
        "故障/",
    ]

    tag_markers = {
        "troubleshooting",
        "debug",
        "bug",
        "error",
        "fix",
        "问题",
        "报错",
    }

    text_markers = [
        "error",
        "exception",
        "traceback",
        "failed",
        "cannot",
        "not found",
        "timeout",
        "permission denied",
        "报错",
        "错误",
        "失败",
        "异常",
        "无法",
        "超时",
        "解决",
        "原因",
        "现象",
        "排查",
        "问题",
    ]

    return (
        any(marker in path for marker in path_markers)
        or any(marker in title for marker in text_markers)
        or bool(tags & tag_markers)
        or any(marker in excerpt for marker in text_markers)
        or profile.note_type == "troubleshooting"
    )


def candidate_report_to_dict(report: CandidateReport) -> dict[str, Any]:
    return asdict(report)
