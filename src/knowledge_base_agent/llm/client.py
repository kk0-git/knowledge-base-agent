from __future__ import annotations

from typing import Protocol

from knowledge_base_agent.llm.schema import LLMRequest, LLMResponse

class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...