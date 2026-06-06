from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

from knowledge_base_agent.candidates import CandidateReport
from knowledge_base_agent.graph import GraphEdge, KnowledgeGraph
from knowledge_base_agent.profile import NoteProfile

# 把 candidate 按 path 聚合成 note signals
# 按目录分组
# 为每篇笔记附上 profile、signals、degree
# 为 packet 附上 internal/external wikilink edges

@dataclass(frozen=True)
class EdgeInfo:
    source: str
    target: str
    raw_target: str
    resolved: bool
    kind: str = "wikilink"


@dataclass(frozen=True)
class TagRootGroup:
    kind: str
    tag_root: str
    notes: list[str]

@dataclass(frozen=True)
class PacketNote:
    path: str
    title: str
    note_type: str
    tags: list[str]
    headings: list[str]
    links: list[dict[str, str]]  # [{"target": "...", "dir": "out|in|mutual"}, ...]
    excerpt: str
    signals: list[str]
    degree_info: str

@dataclass(frozen=True)
class ReviewPacket:
    packet_id: str
    scope_type: str
    scope_id: str
    notes: list[PacketNote]
    internal_edges: list[EdgeInfo] = field(default_factory=list)
    external_edges: list[EdgeInfo] = field(default_factory=list)
    tag_root_groups: list[TagRootGroup] = field(default_factory=list)
    strategic_context: dict[str, Any] = field(default_factory=dict)


def build_review_packets(
        profiles: list[NoteProfile],
        graph: KnowledgeGraph,
        candidate_report: CandidateReport,
        max_notes_per_packet: int = 12,
        max_packets: int = 5,
) -> list[ReviewPacket]:
    profile_by_path = {profile.path: profile for profile in profiles}
    signals_by_path = collect_signals_by_path(candidate_report)

    directory_paths = group_candidate_paths_by_directory(signals_by_path)
    degree_by_path = {
        node.path: f"in:{node.in_degree} out:{node.out_degree}"
        for node in graph.nodes
    }
    # 从 graph edges 构建出入链索引（使用解析后的路径）
    incoming_by_path: dict[str, list[str]] = {}
    outgoing_by_path: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.resolved:
            incoming_by_path.setdefault(edge.target, []).append(edge.source)
            outgoing_by_path.setdefault(edge.source, set()).add(edge.target)
        else:
            # 未解析的链接也记录，但 target 保持原始名
            outgoing_by_path.setdefault(edge.source, set()).add(edge.target)

    packets: list[ReviewPacket] = []
    for directory, candidate_paths in sorted(
        directory_paths.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        note_paths = expand_directory_context(
            directory=directory,
            candidate_paths=candidate_paths,
            profiles=profiles,
            graph=graph,
            max_notes=max_notes_per_packet,
        )

        packet_notes: list[PacketNote] = []

        for path in note_paths:
            profile = profile_by_path.get(path)
            if profile is None:
                continue

            # 构建统一 links：从 graph edges 取解析后的出入链
            out_set = outgoing_by_path.get(profile.path, set())
            in_set = set(incoming_by_path.get(profile.path, []))
            links: list[dict[str, str]] = []
            for t in sorted(out_set | in_set):
                if t in out_set and t in in_set:
                    links.append({"target": t, "dir": "mutual"})
                elif t in out_set:
                    links.append({"target": t, "dir": "out"})
                else:
                    links.append({"target": t, "dir": "in"})

            packet_notes.append(
                PacketNote(
                    path=profile.path,
                    title=profile.title,
                    note_type=profile.note_type,
                    tags=profile.tags,
                    headings=profile.headings,
                    links=links,
                    excerpt=(profile.semantic_excerpt or profile.excerpt)[:1500],
                    signals=sorted(signals_by_path.get(profile.path, [])),
                    degree_info=degree_by_path.get(profile.path, "in:0 out:0"),
                )
            )
        if not packet_notes:
            continue

        packet_path_set = {note.path for note in packet_notes}
        internal_edges, external_edges = collect_packet_edges(graph.edges, packet_path_set)
        tag_root_groups = build_tag_root_groups(packet_notes)

        packet_id = build_packet_id(directory)

        packets.append(
            ReviewPacket(
                packet_id=packet_id,
                scope_type="directory",
                scope_id=directory,
                notes=packet_notes,
                internal_edges=internal_edges,
                external_edges=external_edges,
                tag_root_groups=tag_root_groups,
                strategic_context={
                    "goal": "audit_directory_context",
                    "hint": "这组笔记位于同一目录或与该目录候选笔记直接相关。请结合候选信号、笔记内容和 wikilink 关系给出整理建议。",
                    "focus": [
                        "semantic_type_correction",
                        "missing_tags",
                        "cross_links",
                        "troubleshooting_structure",
                        "knowledge_blocks",
                        "duplicate_or_complementary_notes",
                    ],
                },
            )
        )

        if len(packets) >= max_packets:
            break

    return packets

def expand_directory_context(
        directory: str,
        candidate_paths: list[str],
        profiles: list[NoteProfile],
        graph: KnowledgeGraph,
        max_notes: int,
) -> list[str]:
    selected: list[str] = []
    selected_set: set[str] = set()
    def add(path: str) -> None:
        if path not in selected_set and len(selected) < max_notes:
            selected.append(path)
            selected_set.add(path)

    # 1. 先放有 candidate signal 的笔记。
    for path in candidate_paths:
        add(path)

    # 2. 再补同目录下的其他笔记，给 LLM 邻居上下文。
    for profile in sorted(profiles, key=lambda item: item.path.lower()):
        if get_directory(profile.path) == directory:
            add(profile.path)

    # 3. 再补直接链接或反链邻居，优先保留跨目录线索。
    for edge in graph.edges:
        if edge.source in selected_set and edge.resolved:
            add(edge.target)
        elif edge.target in selected_set and edge.resolved:
            add(edge.source)

    return selected

def collect_signals_by_path(candidate_report: CandidateReport) -> dict[str, set[str]]:
    signals: dict[str, set[str]] = {}
    """把 CandidateReport 中的候选项按路径聚合成 signals，形成每个路径对应的信号集合。"""
    # 构建 CandidateReport 中的每个候选项列表bucket
    buckets = [
        candidate_report.orphan_notes,
        candidate_report.weakly_linked_notes,
        candidate_report.hub_candidates,
        candidate_report.missing_tag_notes,
        candidate_report.temporary_notes,
        candidate_report.troubleshooting_notes,
    ]

    for bucket in buckets:
        for candidate in bucket:
            if candidate is None or candidate.path is None:
                continue
            # 同一路径的候选项可能有不同的类型，收集所有类型作为该路径的signals。
            signals.setdefault(candidate.path, set()).add(candidate.kind)
    return signals

def group_candidate_paths_by_directory(
        siginals_by_path: dict[str, set[str]]
) -> dict[str, list[str]]:
    """把路径按目录分组，形成 packet 的 scope。比如 "folder1/subfolder/note.md" 会被分到 "folder1/subfolder" 这个目录下。"""
    grouped: dict[str, list[str]] = {}

    for path in siginals_by_path:
        directory = get_directory(path)
        # 同一目录下的路径聚合到一起，形成一个 packet 的 scope。
        grouped.setdefault(directory, []).append(path)

    for directory in grouped:
        grouped[directory].sort(key=str.lower)  # 同一目录下的路径按字母序排序，保持一致性。
    return grouped

def collect_packet_edges(
        edges: list[GraphEdge],
        packet_paths: set[str],
) -> tuple[list[EdgeInfo], list[EdgeInfo]]:
    internal_edges: list[EdgeInfo] = []
    external_edges: list[EdgeInfo] = []
    for edge in edges:
        source_inside = edge.source in packet_paths
        target_inside = edge.target in packet_paths

        if not source_inside and not target_inside:
            continue

        edge_info = EdgeInfo(
            source=edge.source,
            target=edge.target,
            raw_target=edge.raw_target,
            resolved=edge.resolved,
        )

        if source_inside and target_inside:
            internal_edges.append(edge_info)
        else:
            external_edges.append(edge_info)
    return internal_edges, external_edges

def build_tag_root_groups(packet_notes: list[PacketNote]) -> list[TagRootGroup]:
    notes_by_root: dict[str, set[str]] = {}

    for note in packet_notes:
        for tag_root in collect_tag_roots(note.tags):
            notes_by_root.setdefault(tag_root, set()).add(note.path)

    groups: list[TagRootGroup] = []
    for tag_root, note_paths in sorted(notes_by_root.items(), key=lambda item: item[0]):
        if len(note_paths) < 2:
            continue

        groups.append(
            TagRootGroup(
                kind="shared_tag_root",
                tag_root=tag_root,
                notes=sorted(note_paths, key=str.lower),
            )
        )

    return groups

def collect_tag_roots(tags: list[str]) -> set[str]:
    roots: set[str] = set()
    for tag in tags:
        normalized = tag.strip().strip("#").replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]

        # 只处理深度 >= 3 的标签：
        # tags/web框架/fastapi -> tags/web框架
        # tags/python -> 忽略
        # python -> 忽略
        if len(parts) < 3:
            continue

        roots.add("/".join(parts[:2]))

    return roots

    
def get_directory(path: str) -> str:
    """从路径中提取目录部分，作为 packet 的 scope_id。比如 "folder1/subfolder/note.md" 的目录是 "folder1/subfolder"。"""
    parent = PurePosixPath(path).parent.as_posix()

    if parent == ".":
        return "root"
    return parent

def build_packet_id(directory: str) -> str:
    """把目录名转换成适合用作 packet_id 的字符串，替换掉斜杠和空格等特殊字符。"""
    safe = directory.replace("/", "_").replace(" ", "_")
    return f"dir_{safe}"

def review_packet_to_dict(packet: ReviewPacket) -> dict:
    return asdict(packet)
