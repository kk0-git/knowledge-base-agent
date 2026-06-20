from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.errors import SkillLoadError
from agent.tool_registry import ToolRegistry


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    version: int
    description: str
    system_prompt: str
    allowed_tools: frozenset[str]
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    max_steps: int = 8
    temperature: float = 0.2
    output_contract: dict[str, Any] = field(default_factory=dict)
    trace_policy: dict[str, Any] = field(default_factory=dict)
    manifest_path: str = ""
    prompt_path: str = ""


class SkillLoader:
    def __init__(self, skills_root: Path | str, registry: ToolRegistry | None = None):
        self.skills_root = Path(skills_root)
        self.registry = registry

    def list_skills(self) -> list[str]:
        if not self.skills_root.exists():
            return []
        return sorted(path.name for path in self.skills_root.iterdir() if (path / "manifest.json").exists())

    def load(self, name: str) -> LoadedSkill:
        skill_name = str(name).strip()
        if not skill_name:
            raise SkillLoadError("skill name is required")
        skill_dir = self.skills_root / skill_name
        manifest_path = skill_dir / "manifest.json"
        prompt_path = skill_dir / "SKILL.md"
        if not manifest_path.exists():
            raise SkillLoadError(f"skill manifest not found: {manifest_path}")
        if not prompt_path.exists():
            raise SkillLoadError(f"skill prompt not found: {prompt_path}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SkillLoadError(f"invalid skill manifest JSON: {manifest_path}: {exc}") from exc

        loaded = self._from_manifest(
            manifest=manifest,
            system_prompt=prompt_path.read_text(encoding="utf-8").strip(),
            manifest_path=manifest_path,
            prompt_path=prompt_path,
        )
        self._validate_tools(loaded)
        return loaded

    def _from_manifest(
        self,
        *,
        manifest: dict[str, Any],
        system_prompt: str,
        manifest_path: Path,
        prompt_path: Path,
    ) -> LoadedSkill:
        name = str(manifest.get("name") or "").strip()
        if not name:
            raise SkillLoadError(f"manifest missing name: {manifest_path}")
        version = parse_positive_int(manifest.get("version", 1), "version", manifest_path)
        max_steps = parse_positive_int(manifest.get("max_steps", 8), "max_steps", manifest_path)
        temperature = parse_float(manifest.get("temperature", 0.2), "temperature", manifest_path)
        allowed_tools = frozenset(parse_string_list(manifest.get("allowed_tools", []), "allowed_tools", manifest_path))
        denied_tools = frozenset(parse_string_list(manifest.get("denied_tools", []), "denied_tools", manifest_path))
        if allowed_tools & denied_tools:
            raise SkillLoadError(f"allowed_tools and denied_tools overlap in {manifest_path}: {sorted(allowed_tools & denied_tools)}")
        if not system_prompt:
            raise SkillLoadError(f"skill prompt is empty: {prompt_path}")
        return LoadedSkill(
            name=name,
            version=version,
            description=str(manifest.get("description") or "").strip(),
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            max_steps=max_steps,
            temperature=temperature,
            output_contract=dict(manifest.get("output_contract") or {}),
            trace_policy=dict(manifest.get("trace_policy") or {}),
            manifest_path=str(manifest_path),
            prompt_path=str(prompt_path),
        )

    def _validate_tools(self, skill: LoadedSkill) -> None:
        if self.registry is None:
            return
        missing = [name for name in skill.allowed_tools if not self.registry.has(name)]
        if missing:
            raise SkillLoadError(f"skill {skill.name} references unknown tools: {', '.join(sorted(missing))}")


def parse_string_list(value: Any, field_name: str, path: Path) -> list[str]:
    if not isinstance(value, list):
        raise SkillLoadError(f"{field_name} must be a list in {path}")
    return [str(item).strip() for item in value if str(item).strip()]


def parse_positive_int(value: Any, field_name: str, path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SkillLoadError(f"{field_name} must be an integer in {path}") from exc
    if parsed <= 0:
        raise SkillLoadError(f"{field_name} must be positive in {path}")
    return parsed


def parse_float(value: Any, field_name: str, path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SkillLoadError(f"{field_name} must be a number in {path}") from exc
