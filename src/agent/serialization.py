from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if callable(value):
        return repr(value)
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


def json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=indent)


def truncate_text(text: str, max_chars: int = 2000) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"
