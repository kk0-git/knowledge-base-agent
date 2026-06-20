from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent.errors import LLMToolCallError
from agent.schema import AgentMessage, ToolCall
from agent.serialization import json_dumps
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest


ToolMode = Literal["native", "json", "auto"]


@dataclass(frozen=True)
class LLMToolRequest:
    model: str
    messages: list[AgentMessage]
    tools: list[dict[str, Any]]
    temperature: float = 0.2
    tool_choice: str | dict[str, Any] = "auto"
    tool_mode: ToolMode = "auto"
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMToolResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    used_mode: str = ""


class ToolCallingLLMClient(Protocol):
    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        ...


class OpenAICompatibleToolCallingClient:
    def __init__(self, base_client: Any):
        self.base_client = base_client

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        mode = request.tool_mode
        if mode == "json":
            return self._complete_json_fallback(request)
        if mode == "native":
            return self._complete_native(request)
        try:
            return self._complete_native(request)
        except Exception as exc:
            if looks_like_tool_support_error(exc):
                return self._complete_json_fallback(request, fallback_error=str(exc))
            raise

    def _complete_native(self, request: LLMToolRequest) -> LLMToolResponse:
        if not hasattr(self.base_client, "base_url") or not hasattr(self.base_client, "config"):
            raise LLMToolCallError("base client does not expose OpenAI-compatible connection details")

        url = f"{str(self.base_client.base_url).rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [agent_message_to_openai(message) for message in request.messages],
            "temperature": request.temperature,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
        }
        if request.response_format is not None:
            payload["response_format"] = request.response_format

        headers = {"Content-Type": "application/json"}
        api_key = getattr(self.base_client.config, "api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        http_request = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                http_request,
                timeout=getattr(self.base_client.config, "timeout_seconds", 60),
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMToolCallError(f"LLM tools HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMToolCallError(f"LLM tools request failed: {exc}") from exc

        choice = raw.get("choices", [{}])[0]
        message = choice.get("message") or {}
        return LLMToolResponse(
            content=str(message.get("content") or ""),
            tool_calls=parse_openai_tool_calls(message.get("tool_calls") or []),
            finish_reason=str(choice.get("finish_reason") or ""),
            raw=raw,
            used_mode="native",
        )

    def _complete_json_fallback(
        self,
        request: LLMToolRequest,
        *,
        fallback_error: str = "",
    ) -> LLMToolResponse:
        prompt = build_json_fallback_prompt(request, fallback_error=fallback_error)
        response = self.base_client.complete(
            LLMRequest(
                model=request.model,
                messages=[
                    LLMMessage(role="system", content=JSON_FALLBACK_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=request.temperature,
                response_format={"type": "json_object"},
            )
        )
        payload = parse_json_object(response.content)
        tool_calls = parse_fallback_tool_calls(payload.get("tool_calls") or [])
        content = str(payload.get("final") or payload.get("content") or payload.get("answer") or "")
        return LLMToolResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            raw={"fallback_payload": payload, "llm_raw": response.raw},
            used_mode="json",
        )


JSON_FALLBACK_SYSTEM_PROMPT = """You are a JSON tool-calling adapter.

Return exactly one JSON object and no Markdown.
If you need tools, return:
{"tool_calls":[{"id":"call_1","name":"tool_name","arguments":{...}}]}

If you are ready to answer, return:
{"final":"final answer text"}
"""


def build_json_fallback_prompt(request: LLMToolRequest, *, fallback_error: str = "") -> str:
    return "\n\n".join(
        [
            "# Original Messages",
            json_dumps([agent_message_to_openai(message) for message in request.messages], indent=2),
            "",
            "# Available Tools",
            json_dumps(request.tools, indent=2),
            "",
            "# Tool Choice",
            json_dumps(request.tool_choice),
            "",
            "# Native Tool Calling Error",
            fallback_error or "(not applicable)",
        ]
    )


def agent_message_to_openai(message: AgentMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "tool":
        payload["tool_call_id"] = message.tool_call_id or ""
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json_dumps(call.arguments),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def parse_openai_tool_calls(raw_calls: list[Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls, start=1):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") or {}
        name = str(function.get("name") or raw.get("name") or "").strip()
        if not name:
            continue
        arguments = parse_arguments(function.get("arguments") or raw.get("arguments") or {})
        calls.append(ToolCall(id=str(raw.get("id") or f"call_{index}"), name=name, arguments=arguments))
    return calls


def parse_fallback_tool_calls(raw_calls: list[Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls, start=1):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = str(raw.get("name") or function.get("name") or "").strip()
        if not name:
            continue
        arguments = parse_arguments(raw.get("arguments") or function.get("arguments") or {})
        calls.append(ToolCall(id=str(raw.get("id") or f"call_{index}"), name=name, arguments=arguments))
    return calls


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise LLMToolCallError("no JSON object found in fallback response")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise LLMToolCallError("fallback response JSON must be an object")
    return payload


def looks_like_tool_support_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "does not expose openai-compatible" in text:
        return True
    markers = ["tool", "tools", "tool_choice", "function", "unsupported", "unknown parameter", "extra inputs"]
    return any(marker in text for marker in markers)
