from agent.tools.vault.grep_vault import grep_vault_spec
from agent.tools.vault.read_note import read_note_spec
from agent.tools.vault.search_notes import search_notes_spec
from agent.tool_registry import ToolRegistry


def register_vault_tools(registry: ToolRegistry) -> None:
    registry.register(search_notes_spec())
    registry.register(read_note_spec())
    registry.register(grep_vault_spec())


__all__ = ["register_vault_tools"]
