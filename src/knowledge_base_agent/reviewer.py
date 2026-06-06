from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from knowledge_base_agent.config import LLMConfig
from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from knowledge_base_agent.packet import ReviewPacket


PACKET_SYSTEM_PROMPT = """你是一个 Obsidian vault 审计助手。

你的任务是评审一组相关笔记，而不是孤立评审单个信号。

你会收到 ReviewPacket，里面包含：
- notes：同一目录或同一上下文中的笔记
- 每篇笔记的 links（统一链接，含方向 out/in/mutual）和 packet_peers（与包内其他笔记的链接状态 mutual/out_only/in_only/none）
- 每篇笔记的 tags 为当前标签（体系不统一，仅供参考），你可以基于笔记内容重新设计标签，而非沿用现有标签
- 每篇笔记的 candidate signals
- internal_edges：包内笔记之间的显式 wikilink
- external_edges：包内笔记指向外部的显式 wikilink
- tag_root_groups：隐式主题分组信号，表示多篇笔记共享层级标签前缀。不是显式 wikilink，也不是 source-target 边——适合推荐互链和主题页，不要仅凭它建议合并
- strategic_context：这次评审的目标

规则：
- 你的输出是审计建议而非直接操作。整理知识库的方式是加标签、补链接、建索引、发现主题簇——而不是删除或改写正文。
- 当信息不足以做出明确判断时，标记为需要人工复核而非强行给结论。
- note_type 为机器初步分类（可能不准确），你可以基于实际内容纠正
- 允许纠正机器分类，例如 troubleshooting 信号可能实际是 concept。
- 重点判断笔记之间的关系：重复、互补、历史记录、应该互链、应该形成主题页。
- packet_peers 已标注每对笔记的链接方向状态，你的 suggested_links 只建议还需要补充的方向
- 输出必须是合法 JSON，不要输出 Markdown。
"""


@dataclass(frozen=True)
class PacketReviewReport:
    packet_reviews: list[dict[str, Any]] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)

def review_packets_with_llm(
    packets: list[ReviewPacket],
    client: LLMClient,
    config: LLMConfig,
    limit: int = 3,
) -> PacketReviewReport:
    packet_reviews: list[dict[str, Any]] = []
    raw_responses: list[str] = []

    for packet in packets[:limit]:
        prompt = build_packet_review_prompt(packet)

        response = client.complete(
            LLMRequest(
                model=config.model,
                messages=[
                    LLMMessage(role="system", content=PACKET_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=config.temperature,
                response_format={"type": "json_object"},
            )
        )

        raw_responses.append(response.content)

        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError:
            payload = {
                "packet_id": packet.packet_id,
                "parse_error": True,
                "raw_response": response.content,
            }

        packet_reviews.append(payload)

    return PacketReviewReport(
        packet_reviews=packet_reviews,
        raw_responses=raw_responses,
    )


def build_packet_review_prompt(packet: ReviewPacket) -> str:
    packet_json = json.dumps(
        compact_packet(packet),
        ensure_ascii=False,
        indent=2,
    )

    return f"""请评审以下 ReviewPacket。

ReviewPacket:
{packet_json}

请输出 JSON，格式必须为：
{{
  "packet_id": "string",
  "scope_type": "directory|graph_neighborhood|similarity_cluster",
  "scope_id": "string",
  "summary": "string",
  "note_reviews": [
    {{
      "path": "string",
      "semantic_type": "concept|summary|troubleshooting|project|reference|daily|temporary|index|code|unknown",
      "confidence": "low|medium|high",
      "recommended_action": "keep|add_links|add_tags|move_or_archive|merge_review_needed|split_review_needed|convert_to_troubleshooting_template|review_needed",
      "reason": "string",
      "suggested_tags": ["string"],
      "suggested_links": ["string"],
      "risk": "low|medium|high"
    }}
  ],
  "relationship_suggestions": [
    {{
      "source": "string",
      "target": "string",
      "relationship": "duplicate|overlap|complementary|history|related|prerequisite|unknown",
      "recommended_action": "add_link|merge_review_needed|keep_separate|create_topic_note|review_needed",
      "reason": "string"
    }}
  ],
  "knowledge_blocks": [
    {{
      "topic": "string",
      "notes": ["string"],
      "suggested_moc_or_topic_note": "string|null",
      "reason": "string"
    }}
  ]
}}

要求：
- reason 使用中文，简洁说明依据。
- suggested_links 不要加 [[ ]]。
- 判断笔记间的重叠程度来决定操作：完全重复 → 建议合并审查；局部重叠 → 补充互链；互补 → 互链或共属同一主题页；历史记录 → 归档
- 机器信号的 semantic_type 仅供参考，你应基于正文内容独立判断每篇笔记的类型，在 note_reviews 中给出你的结论。
- 不要输出 Markdown。
- suggested_tags 基于笔记内容和包内关系设计 2-5 个标签，追求：
    ① 跨笔记可复用，同一概念使用同一标签名 
    ② 粒度适中，不过宽（如"技术"）也不过细（如单篇笔记的标题词） 
    ③ 便于后续构建标签索引和横向关联
    不要加 #。
"""


def compact_packet(packet: ReviewPacket) -> dict[str, Any]:
    packet_paths = {note.path for note in packet.notes}

    return {
        "packet_id": packet.packet_id,
        "scope_type": packet.scope_type,
        "scope_id": packet.scope_id,
        "strategic_context": packet.strategic_context,
        "notes": [
            {
                "path": note.path,
                "title": note.title,
                "note_type": note.note_type,
                "tags": note.tags,
                "headings": note.headings[:20],
                "links": note.links[:60],
                "packet_peers": _build_packet_peers(note.path, note.links, packet_paths),
                "signals": note.signals,
                "degree_info": note.degree_info,
                "excerpt": note.excerpt[:1500],
            }
            for note in packet.notes
        ],
        "internal_edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "raw_target": edge.raw_target,
                "resolved": edge.resolved,
                "kind": edge.kind,
            }
            for edge in packet.internal_edges[:50]
        ],
        "external_edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "raw_target": edge.raw_target,
                "resolved": edge.resolved,
                "kind": edge.kind,
            }
            for edge in packet.external_edges[:50]
        ],
        "tag_root_groups": [
            {
                "kind": group.kind,
                "tag_root": group.tag_root,
                "notes": group.notes,
            }
            for group in packet.tag_root_groups[:50]
        ],
    }

def _build_packet_peers(
    note_path: str,
    links: list[dict[str, str]],
    packet_paths: set[str],
) -> list[dict[str, str]]:
    """计算当前笔记与 packet 内其他笔记的链接状态。

    返回 [{ "peer": "path", "link": "mutual|out_only|in_only|none" }, ...]
    LLM 可直接据此判断需补充哪个方向的链接，无需自行交叉对比 links_out/incoming_links。
    """
    out_targets: set[str] = {l["target"] for l in links if l["dir"] in ("out", "mutual")}
    in_targets: set[str] = {l["target"] for l in links if l["dir"] in ("in", "mutual")}

    peers: list[dict[str, str]] = []
    for peer in sorted(packet_paths - {note_path}):
        out = peer in out_targets
        inc = peer in in_targets
        if out and inc:
            status = "mutual"
        elif out:
            status = "out_only"
        elif inc:
            status = "in_only"
        else:
            status = "none"
        peers.append({"peer": peer, "link": status})
    return peers


def packet_review_report_to_dict(report: PacketReviewReport) -> dict[str, Any]:
    return asdict(report)
