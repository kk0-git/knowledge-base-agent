from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from knowledge_base_agent.profile import NoteProfile

@dataclass(frozen=True)
class GraphNode:
    id: str
    path: str
    title: str
    note_type: str
    tags: list[str] = field(default_factory=list)
    in_degree: int = 0
    out_degree: int = 0

@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    raw_target: str
    resolved: bool


@dataclass(frozen=True)
class KnowledgeGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    orphan_notes: list[str]
    hub_notes: list[str]
    unresolved_links: list[str]
    backlinks: dict[str, list[str]]

def build_knowledge_graph(profiles: list[NoteProfile], hub_threshold: int = 5) -> KnowledgeGraph:
    path_by_stem = build_path_lookup(profiles)
    paths = {profile.path for profile in profiles}

    edges: list[GraphEdge] = []
    backlinks: dict[str, set[str]] = {profile.path: set() for profile in profiles}
    out_degree: dict[str, int] = {profile.path: 0 for profile in profiles}
    unresolved_links: set[str] = set()

    for profile in profiles:
        # 解析链接，构建边和反向链接
        for raw_target in profile.links_out:
            resolved_target = resolve_wikilink(raw_target, paths, path_by_stem)
            resolved = resolved_target is not None

            if resolved_target is None:
                target = raw_target
                unresolved_links.add(raw_target)
            else:
                target = resolved_target
                backlinks[target].add(profile.path)

            edges.append(
                GraphEdge(
                    source=profile.path,
                    target=target,
                    raw_target=raw_target,
                    resolved=resolved,
                )
            )

            out_degree[profile.path] += 1

    nodes: list[GraphNode] = []
    for profile in profiles:
        in_degree = len(backlinks[profile.path])
        node_out_degree = out_degree[profile.path]

        nodes.append(
            GraphNode(
                id=profile.path,
                path=profile.path,
                title=profile.title,
                note_type=profile.note_type,
                tags=profile.tags,
                in_degree=in_degree,
                out_degree=node_out_degree,
            )
        )

    orphan_notes = sorted(
        node.path
        for node in nodes
        if node.in_degree == 0 and node.out_degree == 0
    )

    hub_notes = sorted(
        (node.path for node in nodes if node.in_degree + node.out_degree >= hub_threshold),
        key=lambda path: get_total_degree(path, nodes),
        reverse=True,
    )

    return KnowledgeGraph(
        nodes=sorted(nodes, key=lambda node: node.path.lower()),
        edges=sorted(edges, key=lambda edge: (edge.source.lower(), edge.target.lower())),
        orphan_notes=orphan_notes,
        hub_notes=hub_notes,
        unresolved_links=sorted(unresolved_links),
        backlinks={
            path: sorted(sources)
            for path, sources in sorted(backlinks.items(), key=lambda item: item[0].lower())
        },
    )


def build_path_lookup(profiles: list[NoteProfile]) -> dict[str, str]:
    """构建一个路径查找表，支持通过不同形式的路径（完整路径、去掉 .md 后缀、仅文件名）来查找笔记。"""
    lookup = dict[str, str]()
    for profile in profiles:
        path = profile.path
        stem = Path(path).stem

        lookup.setdefault(stem, path) # 支持通过文件名（不带路径）来查找
        lookup.setdefault(path, path) # 支持通过完整路径来查找
        lookup.setdefault(path.removesuffix(".md"), path) # 支持通过去掉 .md 后缀的路径来查找

    return lookup

def resolve_wikilink(
    raw_target: str,
    paths: set[str],
    path_by_stem: dict[str, str],
) -> str | None:
    target = normalize_wikilink_target(raw_target)

    if target in paths:
        return target

    if f"{target}.md" in paths:
        return f"{target}.md"

    if target in path_by_stem:
        return path_by_stem[target]

    return None


def normalize_wikilink_target(raw_target: str) -> str:
    target = raw_target.strip().replace("\\", "/")

    if "#" in target:
        target = target.split("#", 1)[0]

    if "^" in target:
        target = target.split("^", 1)[0]

    return target.strip()


def get_total_degree(path: str, nodes: list[GraphNode]) -> int:
    for node in nodes:
        if node.path == path:
            return node.in_degree + node.out_degree
    return 0


def graph_to_dict(graph: KnowledgeGraph) -> dict[str, Any]:
    return asdict(graph)