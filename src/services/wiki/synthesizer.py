from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.wiki.schema import NoteTagRecord, WikiTagRecord


RELATED_WIKI_HEADING = "\u76f8\u5173 wiki"
SOURCES_HEADING = "\u6765\u6e90"


WIKI_SYSTEM_PROMPT = """You are a personal knowledge-base wiki synthesis assistant.

Task: synthesize multiple source notes under the same wiki candidate into one readable and auditable wiki page.

Rules:
- Write in Chinese.
- Start directly with the article body. Do not write chatty prefaces such as "OK", "below is", or "based on the provided materials".
- A wiki page is a synthesis layer, not a copy of source notes.
- Use only the provided source notes. Do not add facts that are absent from the sources.
- Preserve differences, limitations, and context across sources.
- Output concise Markdown body only. Do not output YAML frontmatter.
- Cite important claims with the source's relative vault path, for example [[courses/js/JavaWeb/Servlet.md]].
- Do not invent virtual citations such as [[S1]] or [[S2]].
- If the sources are insufficient, say the source evidence is insufficient instead of fabricating.
- Related wiki pages may be provided as navigation context.
- Do not write a related wiki section yourself. The system will render it after your article body.
"""


WIKI_OVERVIEW_SYSTEM_PROMPT = """You are a personal knowledge-base wiki overview assistant.

The current tag is a broad topic. Generate a navigation-oriented overview, not a detailed tutorial.

Rules:
- Write in Chinese.
- Start directly with the article body. Do not write chatty prefaces such as "OK", "below is", or "based on the provided materials".
- Explain what this topic covers in 2-4 paragraphs.
- List the main subtopics or knowledge blocks and point to representative source notes.
- Do not expand into a full tutorial; detailed content belongs to more focused child wiki pages.
- Do not rewrite every source note as a directory dump.
- If there are many source notes, summarize the pattern and subtopics. Representative citations are enough.
- Cite with the source's relative vault path, for example [[courses/js/JavaWeb/Servlet.md]].
- Do not invent virtual citations such as [[S1]] or [[S2]].
- Related wiki pages may be provided as navigation context.
- Do not write a related wiki section yourself. The system will render it after your article body.
"""


@dataclass(frozen=True)
class RelatedWikiPage:
    tag: str
    wiki_path: str
    relation: str


@dataclass(frozen=True)
class WikiSynthesisInput:
    tag: str
    source_records: list[NoteTagRecord]
    source_texts: dict[str, str]
    related_wiki_pages: list[RelatedWikiPage] | None = None
    max_source_chars_per_note: int = 2500
    source_limit_notice: str | None = None


class WikiSynthesizer:
    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def synthesize(self, synthesis_input: WikiSynthesisInput) -> str:
        response = self.client.complete(
            LLMRequest(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    LLMMessage(role="system", content=WIKI_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=build_wiki_user_prompt(synthesis_input)),
                ],
            )
        )
        return cleanup_markdown_response(response.content)

    def synthesize_overview(self, synthesis_input: WikiSynthesisInput) -> str:
        response = self.client.complete(
            LLMRequest(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    LLMMessage(role="system", content=WIKI_OVERVIEW_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=build_wiki_user_prompt(synthesis_input)),
                ],
            )
        )
        return cleanup_markdown_response(response.content)


def build_wiki_user_prompt(synthesis_input: WikiSynthesisInput) -> str:
    parts: list[str] = []
    parts.append(f"Tag: {synthesis_input.tag}")
    parts.append("")

    if synthesis_input.source_limit_notice:
        parts.append(synthesis_input.source_limit_notice)
        parts.append("")

    if synthesis_input.related_wiki_pages:
        parts.append(
            "Existing related wiki pages for navigation context. "
            "Do not write a related wiki section yourself; the system will render it:"
        )
        for page in synthesis_input.related_wiki_pages:
            parts.append(f"- [[{page.wiki_path}]] tag={page.tag} relation={page.relation}")
        parts.append("")

    parts.append("Source notes:")
    for index, record in enumerate(synthesis_input.source_records, start=1):
        text = synthesis_input.source_texts.get(record.note_path, "")
        excerpt = text.strip()[: synthesis_input.max_source_chars_per_note]
        tags_text = ", ".join(record.llm_tags or record.user_tags)
        parts.append("")
        parts.append(f"[S{index}] {record.note_path}")
        parts.append(f"Citation to use: [[{record.note_path}]]")
        parts.append(f"Title: {record.title or Path(record.note_path).stem}")
        parts.append(f"Tags: {tags_text}")
        parts.append("---")
        parts.append(excerpt)
        parts.append("---")

    parts.append("")
    parts.append("Write the wiki article body now. Start directly with the content; do not include conversational prefaces.")
    return "\n".join(parts)


def render_wiki_markdown(
    *,
    tag_record: WikiTagRecord,
    title: str,
    body: str,
    source_hashes: dict[str, str],
    visible_source_paths: list[str] | None = None,
    related_wiki_pages: list[RelatedWikiPage] | None = None,
) -> str:
    source_paths = sorted(tag_record.source_paths)
    visible_paths = visible_source_paths or source_paths
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("---")
    lines.append("type: generated_wiki")
    lines.append(f"tag: {yaml_scalar(tag_record.tag)}")
    lines.append(f"title: {yaml_scalar(title)}")
    lines.append("source_paths:")
    for path in source_paths:
        lines.append(f"  - {yaml_scalar(path)}")
    lines.append("source_hashes:")
    for path in source_paths:
        lines.append(f"  {yaml_scalar(path)}: {yaml_scalar(source_hashes.get(path, ''))}")
    lines.append(f"generated_at: {yaml_scalar(tag_record.generated_at or now)}")
    lines.append(f"updated_at: {yaml_scalar(now)}")
    lines.append("status: clean")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(strip_related_wiki_sections(body).strip())
    lines.append("")
    if related_wiki_pages:
        lines.append(f"## {RELATED_WIKI_HEADING}")
        lines.append("")
        for page in related_wiki_pages:
            lines.append(f"- [[{page.wiki_path}]]")
        lines.append("")
    lines.append(f"## {SOURCES_HEADING}")
    lines.append("")
    for path in visible_paths:
        lines.append(f"- [[{path}]]")
    omitted = len(source_paths) - len(visible_paths)
    if omitted > 0:
        lines.append(f"- \u5176\u4f59 {omitted} \u6761\u6765\u6e90\u89c1 frontmatter `source_paths`\u3002")
    lines.append("")
    return "\n".join(lines)


def cleanup_markdown_response(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:markdown|md)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    chatty_prefixes = (
        "\u597d\u7684",
        "\u4e0b\u9762\u662f",
        "\u4ee5\u4e0b\u662f",
        "\u8fd9\u662f",
        "\u6839\u636e\u4f60\u63d0\u4f9b\u7684",
    )
    for prefix in chatty_prefixes:
        if stripped.startswith(prefix):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            break

    return strip_related_wiki_sections(stripped).strip()


def strip_related_wiki_sections(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    skipping = False

    for line in lines:
        if line.strip() == f"## {RELATED_WIKI_HEADING}":
            skipping = True
            continue

        if skipping:
            if line.startswith("## "):
                skipping = False
                output.append(line)
            continue

        output.append(line)

    return "\n".join(output)


def wiki_title_from_tag(tag: str) -> str:
    return tag.split("/")[-1].strip() or tag


def slug_from_tag(tag: str) -> str:
    slug = tag.strip().strip("/").replace("\\", "/")
    slug = re.sub(r"[<>:\"|?*]", "-", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
