from agent.tool_registry import ToolRegistry
from agent.tools.profile.list_profile_signals import list_profile_signals_spec
from agent.tools.profile.recall_profile import recall_profile_spec
from agent.tools.profile.record_signal import record_signal_spec
from agent.tools.profile.write_observation_draft import write_observation_draft_spec


def register_profile_tools(registry: ToolRegistry) -> None:
    registry.register(recall_profile_spec())
    registry.register(record_signal_spec())
    registry.register(list_profile_signals_spec())
    registry.register(write_observation_draft_spec())


__all__ = ["register_profile_tools"]
