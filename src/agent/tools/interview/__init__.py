from agent.tool_registry import ToolRegistry
from agent.tools.interview.advance_layer import advance_layer_spec
from agent.tools.interview.get_interview_state import get_interview_state_spec
from agent.tools.interview.list_plan_topics import list_plan_topics_spec
from agent.tools.interview.select_topic import select_topic_spec


def register_interview_tools(registry: ToolRegistry) -> None:
    registry.register(get_interview_state_spec())
    registry.register(list_plan_topics_spec())
    registry.register(advance_layer_spec())
    registry.register(select_topic_spec())


__all__ = ["register_interview_tools"]
