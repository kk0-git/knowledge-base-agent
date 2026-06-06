from __future__ import annotations

from knowledge_base_agent.config import LLMConfig
from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.openai_compatible import OpenAICompatibleClient


def create_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == "openai_compatible":
        return OpenAICompatibleClient(config)

    raise ValueError(f"Unsupported LLM provider: {config.provider}")
