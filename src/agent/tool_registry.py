from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent.errors import ToolNotFoundError, ToolRegistryError
from agent.schema import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, *, replace: bool = False) -> None:
        validate_tool_spec(spec)
        if spec.name in self._tools and not replace:
            raise ToolRegistryError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"tool not found: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def subset(self, allowed: Iterable[str]) -> "ToolRegistry":
        registry = ToolRegistry()
        for name in allowed:
            registry.register(self.get(str(name)), replace=True)
        return registry

    def schemas_for(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        selected = list(names) if names is not None else self.names()
        schemas: list[dict[str, Any]] = []
        for name in selected:
            spec = self.get(str(name))
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
            )
        return schemas


def validate_tool_spec(spec: ToolSpec) -> None:
    if not spec.name or not spec.name.replace("_", "").replace("-", "").isalnum():
        raise ToolRegistryError(f"invalid tool name: {spec.name!r}")
    if not spec.description.strip():
        raise ToolRegistryError(f"tool description is required: {spec.name}")
    if not isinstance(spec.parameters, dict):
        raise ToolRegistryError(f"tool parameters must be a JSON schema object: {spec.name}")
    if spec.parameters.get("type") != "object":
        raise ToolRegistryError(f"tool parameters.type must be object: {spec.name}")
    if not callable(spec.handler):
        raise ToolRegistryError(f"tool handler must be callable: {spec.name}")
    if spec.timeout_s <= 0:
        raise ToolRegistryError(f"tool timeout_s must be positive: {spec.name}")
