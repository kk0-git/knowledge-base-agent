from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_EXCLUSIONS = [
    ".obsidian/",
    ".git/",
    ".github/",
    ".trash/",
    ".Trash/",
    ".vscode/",
    ".cursor/",
    "reference/",
    "node_modules/",
    "附件/",
]

CONFIG_FILENAME = ".knowledge-agent.conf"

@dataclass(frozen=True) # fronzen表示该类不可变
class AppConfig:
    vault_path: Path
    output_path: Path
    exclusions: list[str]

def load_exclusion_patterns(vault_path: Path) -> list[str]:
    """导入排除模式列表，包含默认和用户定义的模式"""
    config_path = vault_path / CONFIG_FILENAME
    patterns = list(DEFAULT_EXCLUSIONS)

    if not config_path.exists():
        return patterns

    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)

    return patterns

@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    timeout_seconds: int = 120

def load_dotenv(env_path: Path) ->None:
    if not env_path.exists():
        return
    
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)

def load_llm_config(project_root: Path | None = None) -> LLMConfig:
    if project_root is None:
        project_root = Path.cwd()

    load_dotenv(project_root / ".env")

    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai_compatible"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", ""),
        model=os.getenv("LLM_MODEL", ""),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
    )
    