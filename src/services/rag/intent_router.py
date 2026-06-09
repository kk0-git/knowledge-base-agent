from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest


class ConversationCommand(str, Enum):
    NOTES = "Notes"
    REGEX_SEARCH_FILES = "RegexSearchFiles"
    NOTES_ONLINE = "Notes+Online"


@dataclass(frozen=True)
class RouterDecision:
    command: ConversationCommand
    reason: str
    confidence: float
    raw_response: str
    tool_args: dict[str, Any]
    fallback_used: bool = False


SYSTEM_PROMPT = """You are a tool router for a personal knowledge base assistant.

Choose exactly one command and extract the arguments for that command. Do not answer the user's question.

Available commands:

Notes:
Search the user's personal knowledge base with semantic retrieval and keyword retrieval.
Helpful for conceptual, explanatory, how-to, comparison, troubleshooting, broad or cross-note questions.
This search finds a relevant subset of notes, not every exact occurrence.
Pass a natural language query in tool_args.q.

RegexSearchFiles:
Search the user's knowledge base with grep/regex-like exact matching, plus keyword search.
Helpful when the user asks for exact mentions of commands, error messages, paths, config keys,
function names, API names, file names or concrete terms.
You need to know all correct keywords or regex patterns for this tool to be useful.
Do not pass the whole user instruction. Extract the concrete exact pattern into tool_args.regex_pattern.
The regex pattern only matches single lines.
If you cannot extract a concrete pattern, choose Notes.

Notes+Online:
Use both personal knowledge base search and online search. Helpful when local notes and external
context should both be considered.
Pass a natural language query in tool_args.q.

If uncertain, choose Notes.

Return a JSON object only. Do not return Markdown."""


USER_PROMPT_TEMPLATE = """Select the best command and tool arguments for the user question.

{forced_command_instruction}

User question:
{query}

Return JSON in this shape:
{{
  "command": "Notes",
  "reason": "A short explanation of why this command fits.",
  "confidence": 0.82,
  "tool_args": {{
    "q": "Natural language query for semantic search"
  }}
}}

Examples:
- User: "Redis Stream 怎么消费事件"
  Output: {{"command": "Notes", "reason": "Conceptual how-to question needs semantic retrieval.", "confidence": 0.86, "tool_args": {{"q": "Redis Stream 怎么消费事件"}}}}
- User: "XREADGROUP 这个命令在哪些文件里出现过"
  Output: {{"command": "RegexSearchFiles", "reason": "The user asks for exact mentions of a command.", "confidence": 0.95, "tool_args": {{"regex_pattern": "XREADGROUP", "path_prefix": null, "lines_before": 1, "lines_after": 2}}}}
- User: "帮我找 rag 的所有笔记"
  Output: {{"command": "RegexSearchFiles", "reason": "The user asks for notes that directly mention a concrete term.", "confidence": 0.9, "tool_args": {{"regex_pattern": "rag", "path_prefix": null, "lines_before": 1, "lines_after": 2}}}}
- User: "WinError 10060 在哪些笔记里提到过"
  Output: {{"command": "RegexSearchFiles", "reason": "The user asks for exact mentions of an error code.", "confidence": 0.95, "tool_args": {{"regex_pattern": "WinError 10060", "path_prefix": null, "lines_before": 1, "lines_after": 2}}}}"""


class LLMIntentRouter:
    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.0,
        forced_command: ConversationCommand | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.forced_command = forced_command

    def route(self, query: str) -> RouterDecision:
        if not query.strip():
            return fallback_decision("empty query")

        response = self.client.complete(
            LLMRequest(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    LLMMessage(role="system", content=SYSTEM_PROMPT),
                    LLMMessage(
                        role="user",
                        content=USER_PROMPT_TEMPLATE.format(
                            query=query,
                            forced_command_instruction=forced_command_instruction(self.forced_command),
                        ),
                    ),
                ],
            )
        )

        try:
            payload = parse_json_object(response.content)
            command = self.forced_command or parse_command(str(payload.get("command", "")))
            confidence = parse_confidence(payload.get("confidence", 0.0))
            reason = str(payload.get("reason", "")).strip()
            tool_args = parse_tool_args(payload.get("tool_args", {}))
            return RouterDecision(
                command=command,
                reason=reason or "router returned no reason",
                confidence=confidence,
                raw_response=response.content,
                tool_args=tool_args,
                fallback_used=False,
            )
        except Exception as exc:
            decision = fallback_decision(f"router parse failed: {exc}")
            return RouterDecision(
                command=decision.command,
                reason=decision.reason,
                confidence=decision.confidence,
                raw_response=response.content,
                tool_args=decision.tool_args,
                fallback_used=True,
            )


def fallback_decision(reason: str) -> RouterDecision:
    return RouterDecision(
        command=ConversationCommand.NOTES,
        reason=reason,
        confidence=0.0,
        raw_response="",
        tool_args={},
        fallback_used=True,
    )


def parse_command(raw: str) -> ConversationCommand:
    normalized = raw.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if normalized in {"notes"}:
        return ConversationCommand.NOTES
    if normalized in {"regexsearchfiles", "regex", "grep", "rg"}:
        return ConversationCommand.REGEX_SEARCH_FILES
    if normalized in {"notes+online", "notesonline", "online", "notesandonline"}:
        return ConversationCommand.NOTES_ONLINE
    return ConversationCommand.NOTES


def parse_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def parse_tool_args(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def forced_command_instruction(command: ConversationCommand | None) -> str:
    if command is None:
        return ""
    return (
        f"The command is fixed by the user interface: {command.value}. "
        "Do not choose a different command. Only extract the best tool_args for this command."
    )


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def router_decision_to_dict(decision: RouterDecision) -> dict[str, Any]:
    payload = asdict(decision)
    payload["command"] = decision.command.value
    return payload
