from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal


class StepKind(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    FINAL = "final"
    ERROR = "error"


class ToolExecutionStatus(str, Enum):
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    ERROR = "error"


StopReason = Literal["final", "max_steps", "tool_timeout", "tool_error", "llm_error", "cancelled"]
AgentRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    timeout_s: float = 30.0
    permission_level: str = "read"
    side_effect: str = "none"
    requires_confirmation: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: Any = None
    status: str = ToolExecutionStatus.SUCCESS.value
    error: str = ""
    error_type: str = ""
    latency_ms: int = 0
    result_size: int = 0
    summary: str = ""


@dataclass
class AgentMessage:
    role: AgentRole
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class WorkingMemory:
    session_id: str | None = None
    current_topic: str | None = None
    current_layer_index: int = 0
    follow_up_count: int = 0
    plan_topic_names: list[str] = field(default_factory=list)
    notes_read_this_turn: list[str] = field(default_factory=list)
    signals_this_turn: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    messages: list[AgentMessage]
    working: WorkingMemory
    skill_name: str
    step_index: int = 0
    finished: bool = False
    final_answer: str = ""


@dataclass
class AgentStep:
    index: int
    kind: StepKind
    llm_input_chars: int = 0
    llm_output_chars: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    assistant_text: str = ""
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    working_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunConfig:
    skill_name: str
    max_steps: int = 8
    max_tool_calls_per_step: int = 4
    temperature: float = 0.2
    model: str = ""
    stream_final: bool = True
    save_trace: bool = True
    trace_path: str | None = None
    tool_mode: Literal["native", "json", "auto"] = "auto"
    allowed_tools: list[str] | None = None
    reserve_final_step: bool = True


@dataclass
class AgentResult:
    state: AgentState
    steps: list[AgentStep]
    final_answer: str
    total_ms: int
    stopped_reason: StopReason
    trace_id: str = ""
    trace_path: str = ""
    error: str = ""
    error_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStreamEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
