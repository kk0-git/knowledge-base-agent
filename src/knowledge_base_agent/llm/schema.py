from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str

@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: list[LLMMessage]
    temperature: float = 0.2
    response_format: dict[str, Any] | None = None

@dataclass(frozen=True)
class LLMResponse:
    content: str
    raw: dict[str, Any] = field(default_factory=dict)