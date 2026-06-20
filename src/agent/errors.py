from __future__ import annotations


class AgentRuntimeError(Exception):
    """Base class for agent runtime failures."""


class SkillLoadError(AgentRuntimeError):
    """Raised when a skill manifest or prompt cannot be loaded."""


class ToolRegistryError(AgentRuntimeError):
    """Raised when tool registration is invalid."""


class ToolNotFoundError(AgentRuntimeError):
    """Raised when a requested tool is not registered."""


class ToolValidationError(AgentRuntimeError):
    """Raised when tool arguments do not match the tool schema."""


class ToolExecutionError(AgentRuntimeError):
    """Raised when a tool handler fails unexpectedly."""


class ToolTimeoutError(AgentRuntimeError):
    """Raised when a tool exceeds its timeout."""


class LLMToolCallError(AgentRuntimeError):
    """Raised when an LLM tool-calling response cannot be produced."""


class MaxStepsExceeded(AgentRuntimeError):
    """Raised when the runtime reaches its configured step limit."""
