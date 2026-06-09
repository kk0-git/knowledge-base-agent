from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict

from knowledge_base_agent.config import LLMConfig
from knowledge_base_agent.llm.schema import LLMRequest, LLMResponse


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        if not config.base_url:
            raise ValueError("LLM_BASE_URL is required.")

        if not config.model:
            raise ValueError("LLM_MODEL is required.")

        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def complete(self, request: LLMRequest) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model,
            "messages": [asdict(message) for message in request.messages],
            "temperature": request.temperature,
        }

        if request.response_format is not None:
            payload["response_format"] = request.response_format

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        http_request = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.config.timeout_seconds,
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {body}") from exc

        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "LLM request failed. Check LLM_BASE_URL, network access, and proxy settings "
                f"(HTTP_PROXY/HTTPS_PROXY). Original error: {exc}"
            ) from exc

        content = raw["choices"][0]["message"]["content"]
        return LLMResponse(content=content, raw=raw)

    def stream_complete(self, request: LLMRequest):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model,
            "messages": [asdict(message) for message in request.messages],
            "temperature": request.temperature,
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        http_request = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.config.timeout_seconds,
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break

                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = payload.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM stream HTTP error {exc.code}: {body}") from exc

        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM stream request failed: {exc}") from exc
