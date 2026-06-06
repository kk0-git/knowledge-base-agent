from __future__ import annotations

from collections import Counter
from typing import Any

from knowledge_base_agent.candidates import CandidateReport
from knowledge_base_agent.graph import KnowledgeGraph
from knowledge_base_agent.packet import ReviewPacket
from knowledge_base_agent.profile import NoteProfile

def build_vault_audit_report(
        profiles: list[NoteProfile],
        graph: KnowledgeGraph,
        candidate_report: CandidateReport,
        review_packets: list[ReviewPacket],
        llm_review: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []

    lines.append("# Vault Audit")
    lines.append("")
    lines.append("## 1. 总览")
    lines.append("")
    lines.extend(build_overview_lines(profiles, graph, candidate_report, review_packets, llm_review))
    lines.append("")

    lines.append("## 2. 笔记类型分布")
    lines.append("")
    lines.extend(build_note_type_distribution_lines(profiles))
    lines.append("")

    lines.append("## 3. 候选信号概览")
    lines.append("")
    lines.extend(build_candidate_summary_lines(candidate_report))
    lines.append("")

    lines.append("## 4. 图谱结构")
    lines.append("")
    lines.extend(build_graph_summary_lines(graph))
    lines.append("")

    if llm_review and llm_review.get("packet_reviews"):
        lines.append("## 5. LLM Packet 评审结果")
        lines.append("")
        for packet_review in llm_review.get("packet_reviews", []):
            lines.extend(build_packet_review_lines(packet_review))
            lines.append("")
    else:
        lines.append("## 5. LLM Packet 评审结果")
        lines.append("")
        lines.append("未启用 LLM 评审，或没有生成 packet review。")
        lines.append("")

    lines.append("## 6. Review Packets")
    lines.append("")
    lines.extend(build_review_packet_summary_lines(review_packets))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_overview_lines(
    profiles: list[NoteProfile],
    graph: KnowledgeGraph,
    candidate_report: CandidateReport,
    review_packets: list[ReviewPacket],
    llm_review: dict[str, Any] | None,
) -> list[str]:
    llm_review_count = 0
    if llm_review:
        llm_review_count = len(llm_review.get("packet_reviews", []))

    return [
        f"- 笔记数量：{len(profiles)}",
        f"- 图节点数：{len(graph.nodes)}",
        f"- 图边数：{len(graph.edges)}",
        f"- 孤立笔记数：{len(graph.orphan_notes)}",
        f"- Hub 笔记数：{len(graph.hub_notes)}",
        f"- 未解析链接数：{len(graph.unresolved_links)}",
        f"- Review Packet 数：{len(review_packets)}",
        f"- LLM Packet Review 数：{llm_review_count}",
        f"- 候选信号总数：{count_candidates(candidate_report)}",
    ]

def build_note_type_distribution_lines(profiles: list[NoteProfile]) -> list[str]:
    counter = Counter(profile.note_type for profile in profiles)

    if not counter:
        return ["无笔记类型数据。"]

    lines = ["| 类型 | 数量 |", "|---|---:|"]

    for note_type, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{escape_table(note_type)}` | {count} |")

    return lines

def build_candidate_summary_lines(candidate_report: CandidateReport) -> list[str]:
    rows = [
        ("孤立笔记", len(candidate_report.orphan_notes)),
        ("弱连接笔记", len(candidate_report.weakly_linked_notes)),
        ("Hub 候选", len(candidate_report.hub_candidates)),
        ("未解析链接", len(candidate_report.unresolved_links)),
        ("缺少标签", len(candidate_report.missing_tag_notes)),
        ("临时笔记", len(candidate_report.temporary_notes)),
        ("疑似排错笔记", len(candidate_report.troubleshooting_notes)),
    ]

    lines = ["| 候选类型 | 数量 |", "|---|---:|"]

    for name, count in rows:
        lines.append(f"| {name} | {count} |")

    return lines


def build_graph_summary_lines(graph: KnowledgeGraph) -> list[str]:
    lines: list[str] = []

    lines.append("### Hub 笔记")
    lines.append("")
    if graph.hub_notes:
        for path in graph.hub_notes[:20]:
            lines.append(f"- {obsidian_link(path)}")
    else:
        lines.append("暂无 Hub 笔记。")

    lines.append("")
    lines.append("### 孤立笔记")
    lines.append("")
    if graph.orphan_notes:
        for path in graph.orphan_notes[:30]:
            lines.append(f"- {obsidian_link(path)}")
    else:
        lines.append("暂无孤立笔记。")

    lines.append("")
    lines.append("### 未解析链接")
    lines.append("")
    if graph.unresolved_links:
        for target in graph.unresolved_links[:30]:
            lines.append(f"- `{target}`")
    else:
        lines.append("暂无未解析链接。")

    return lines

def build_packet_review_lines(packet_review: dict[str, Any]) -> list[str]:
    packet_id = str(packet_review.get("packet_id", "unknown"))
    scope_type = str(packet_review.get("scope_type", "unknown"))
    scope_id = str(packet_review.get("scope_id", "unknown"))
    summary = str(packet_review.get("summary", ""))

    lines: list[str] = []

    lines.append(f"### Packet: `{packet_id}`")
    lines.append("")
    lines.append(f"- Scope：`{scope_type}` / `{scope_id}`")
    if summary:
        lines.append(f"- 摘要：{summary}")
    lines.append("")

    note_reviews = packet_review.get("note_reviews", [])
    lines.append("#### 单篇笔记建议")
    lines.append("")
    if note_reviews:
        lines.append("| 笔记 | 类型 | 动作 | 风险 | 理由 |")
        lines.append("|---|---|---|---|---|")
        for item in note_reviews:
            path = str(item.get("path", ""))
            semantic_type = str(item.get("semantic_type", "unknown"))
            action = str(item.get("recommended_action", "keep"))
            risk = str(item.get("risk", "low"))
            reason = str(item.get("reason", ""))
            lines.append(
                "| "
                f"{obsidian_link(path)} | "
                f"`{escape_table(semantic_type)}` | "
                f"`{escape_table(action)}` | "
                f"`{escape_table(risk)}` | "
                f"{escape_table(reason)} |"
            )

        lines.append("")
        lines.append("#### 推荐标签与双链")
        lines.append("")
        for item in note_reviews:
            path = str(item.get("path", ""))
            tags = item.get("suggested_tags", []) or []
            links = item.get("suggested_links", []) or []
            if not tags and not links:
                continue

            lines.append(f"- {obsidian_link(path)}")
            if tags:
                lines.append(f"  - 标签：{', '.join(format_tag(tag) for tag in tags)}")
            if links:
                lines.append(f"  - 双链：{', '.join(obsidian_link(link) for link in links)}")
    else:
        lines.append("无单篇笔记建议。")

    lines.append("")
    lines.append("#### 关系建议")
    lines.append("")
    relationship_suggestions = packet_review.get("relationship_suggestions", [])
    if relationship_suggestions:
        lines.append("| Source | Target | 关系 | 动作 | 理由 |")
        lines.append("|---|---|---|---|---|")
        for item in relationship_suggestions:
            source = str(item.get("source", ""))
            target = str(item.get("target", ""))
            relationship = str(item.get("relationship", "unknown"))
            action = str(item.get("recommended_action", "review_needed"))
            reason = str(item.get("reason", ""))
            lines.append(
                "| "
                f"{obsidian_link(source)} | "
                f"{obsidian_link(target)} | "
                f"`{escape_table(relationship)}` | "
                f"`{escape_table(action)}` | "
                f"{escape_table(reason)} |"
            )
    else:
        lines.append("无关系建议。")

    lines.append("")
    lines.append("#### 知识块建议")
    lines.append("")
    knowledge_blocks = packet_review.get("knowledge_blocks", [])
    if knowledge_blocks:
        for block in knowledge_blocks:
            topic = str(block.get("topic", "未命名主题"))
            notes = block.get("notes", []) or []
            suggested_note = block.get("suggested_moc_or_topic_note")
            reason = str(block.get("reason", ""))

            lines.append(f"- **{topic}**")
            if notes:
                lines.append(f"  - 涉及笔记：{', '.join(obsidian_link(str(note)) for note in notes)}")
            if suggested_note:
                lines.append(f"  - 建议主题页：{obsidian_link(str(suggested_note))}")
            if reason:
                lines.append(f"  - 理由：{reason}")
    else:
        lines.append("无知识块建议。")

    return lines


def build_review_packet_summary_lines(review_packets: list[ReviewPacket]) -> list[str]:
    if not review_packets:
        return ["暂无 Review Packet。"]

    lines = [
        "| Packet | Scope | 笔记数 | 包内边 | 包外边 | 共享标签组 |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for packet in review_packets:
        lines.append(
            "| "
            f"`{escape_table(packet.packet_id)}` | "
            f"`{escape_table(packet.scope_type)}` / `{escape_table(packet.scope_id)}` | "
            f"{len(packet.notes)} | "
            f"{len(packet.internal_edges)} | "
            f"{len(packet.external_edges)} | "
            f"{len(packet.tag_root_groups)} |"
        )

    return lines


def count_candidates(candidate_report: CandidateReport) -> int:
    return sum(
        len(bucket)
        for bucket in [
            candidate_report.orphan_notes,
            candidate_report.weakly_linked_notes,
            candidate_report.hub_candidates,
            candidate_report.unresolved_links,
            candidate_report.missing_tag_notes,
            candidate_report.temporary_notes,
            candidate_report.troubleshooting_notes,
        ]
    )


def obsidian_link(path: str) -> str:
    if not path:
        return ""

    normalized = path.replace("\\", "/")
    target = normalized.removesuffix(".md")
    alias = target.split("/")[-1]

    if "/" in target:
        return f"[[{target}|{alias}]]"

    return f"[[{target}]]"


def format_tag(tag: str) -> str:
    cleaned = str(tag).strip().removeprefix("#")
    if not cleaned:
        return ""
    return f"`#{cleaned}`"


def escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
