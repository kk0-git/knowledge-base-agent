from agent.tools.vault.grep_vault import grep_vault_spec
from agent.tools.vault.inspect_note import inspect_note_spec
from agent.tools.vault.list_notes import list_notes_spec
from agent.tools.vault.online_search import online_search_spec
from agent.tools.vault.read_note import read_note_spec
from agent.tools.vault.search_notes import search_notes_spec
from agent.tool_registry import ToolRegistry


def register_vault_tools(registry: ToolRegistry, *, include_online: bool = True) -> None:
    registry.register(search_notes_spec())
    registry.register(inspect_note_spec())
    registry.register(read_note_spec())
    registry.register(grep_vault_spec())
    registry.register(list_notes_spec())
    if include_online:
        registry.register(online_search_spec())


__all__ = ["register_vault_tools"]
