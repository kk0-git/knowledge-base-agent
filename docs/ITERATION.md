# Iteration Log

## Current Iteration

### 2026-06-22 Librarian SKILL Thin Prompt Rewrite

Problem: layered checklist and audit-framed Role made answers procedural and encyclopedic despite tool-chain fixes. Role contradicted Output; escalate/stop rules competed.

Decision: revert to PHASE5 thin behavior layer — helpful Q&A assistant by default, strict mode only when flagged; single stop-or-read-more policy; positive output shape (lead with answer, no default tables, no retrieval narration).

Implemented:

- Rewrote `skills/librarian/SKILL.md` (removed Retrieval Checklist, merged evidence into one section).
- Softened `# Task` line in `build_librarian_input`.

Verification: re-ask boundary questions; compare trace `verification_search_count` and answer opening line.

### 2026-06-22 Librarian Conditional Retrieval Checklist

Problem: same SKILL produced both good and encyclopedic answers; trace diff showed planning variance (verification search vs grep stop), not missing read_note content.

Decision: encode verified behaviors as conditional checklist branches in `skills/librarian/SKILL.md`, not a fixed tool script.

Implemented:

- Added `# Retrieval Checklist` with L1/L2/L3 branches and strict-evidence exceptions.
- Strengthened stop-synthesis and anti-narration rules in Evidence Policy / Output.
- Extended `derive_librarian_metrics` with `grep_count` and `verification_search_count`.
- Added 3 librarian golden eval cases (L2 selected-only, L1 boundary+grep, strict missing name).

Verification:

- `python scripts/agent_eval.py --cases eval/agent_eval/librarian_golden.json --llm-mode fake`
- `python -m unittest tests.test_librarian_agent`

### 2026-06-22 read_note Contract Simplification

Problem: structured read refactor introduced outline-only responses for long notes, doubling step cost and causing outline retry loops (e.g. Multi-agent.md with no headings).

Decision: single `read_note` always returns content; remove `inspect_note` and mode switching.

Implemented:

- Unified `build_read_note_output` with char budget (default 4000), `section_id` navigation, `offset` pagination.
- `sections` map attached only when `truncated=true`.
- Deleted `inspect_note`; updated librarian manifest, SKILL, frontend process UI, tests.

Verification:

- `python -m unittest tests.test_agent_vault_tools tests.test_librarian_agent tests.test_markdown_sections`

### 2026-06-22 Interviewer Tool Chips: State Machine + Permanent Footprint

Goal: make interviewer-agent tool activity visible without exposing backend internals or model reasoning. The user should see that the agent is preparing a follow-up, checking notes, recalling profile memory, or advancing interview state.

Decision: use a Comet-style state machine and permanent action chips, but map backend tools into user-facing actions.

#### User-Facing Mapping

| Backend tool/event | Visible chip |
|---|---|
| `search_notes` | 查找相关笔记 |
| `read_note` | 阅读参考笔记 |
| `grep_vault` | 核对关键词 |
| `recall_profile` | 回顾你的薄弱点 |
| `advance_layer` | 推进追问层次 |
| `select_topic` | 切换面试主题 |
| `state_updated` | 更新面试进度 |
| `agent_stopped` | 已停止继续查阅 |

Hidden by default:

- `get_interview_state`
- `list_plan_topics`
- `inspect_state`
- other debug/precondition tools

#### Event Contract

Prefer extending existing events with a user-facing `ui_action` summary instead of adding a separate protocol first.

Example:

```json
{
  "type": "tool_result",
  "payload": {
    "name": "read_note",
    "status": "success",
    "latency_ms": 42,
    "ui_action": {
      "id": "call_xxx",
      "kind": "note_read",
      "label": "阅读参考笔记",
      "detail": "MCP.md · Host/Client/Server",
      "status": "success",
      "source_paths": ["个人/面试/agent面试/MCP.md"],
      "stats": {
        "note_count": 1
      }
    }
  }
}
```

Rules:

- `tool_started` emits a `running` action.
- `tool_result` emits `success` or `error`.
- `state_updated` emits only visible transitions, not every state snapshot.
- `agent_stopped` marks active actions as stopped.
- Full raw tool output stays in trace; `ui_action` is user-safe and truncated.

#### Frontend State Machine

State is derived from real events:

```text
preparing
  -> no content yet, no visible tool action

tool_running
  -> at least one visible action is running

composing
  -> visible actions finished, answer not yet streaming

generating
  -> first answer delta arrived

done
  -> done event

stopped
  -> agent_stopped or recoverable non-final stop
```

Display:

- Render above the assistant answer stream.
- Running chip uses a pulse dot.
- Completed chip uses a check mark.
- Error/stopped chip uses a warning state.
- After completion, chips stay collapsed as a permanent footprint.
- Expanding a chip shows only safe details: note path, heading, hit count, topic/layer transition, profile count, latency.

#### Persistence

Persist a lightweight action summary on the assistant turn/session, not raw tool outputs:

```json
{
  "agent_actions": [
    {
      "id": "call_read_1",
      "kind": "note_read",
      "tool": "read_note",
      "label": "阅读参考笔记",
      "status": "success",
      "detail": "MCP.md · Host/Client/Server",
      "source_paths": ["个人/面试/agent面试/MCP.md"],
      "latency_ms": 42
    }
  ]
}
```

Suggested write targets:

- Assistant message metadata for historical rendering.
- Session trace event for audit.
- Full agent trace remains the developer/debug source of truth.

#### Implementation Steps

1. Backend action summarizer
   - Convert `ToolCall`, `ToolResult`, `state_updated`, and `agent_stopped` into `ui_action`.
   - Hide debug/precondition tools.
   - Truncate details and source paths.
   - Produce stats such as hit count, note count, source paths, topic, layer.

2. Streaming integration
   - Attach `ui_action` to `tool_started` and `tool_result`.
   - Attach visible transition action to `state_updated` only when topic/layer changes.
   - Mark active actions stopped on `agent_stopped`.

3. Permanent footprint
   - Collect visible actions during `InterviewInterviewerApp.run_turn_stream`.
   - Save `agent_actions` with the assistant turn after final commit.
   - Keep full trace path separately.

4. Frontend rendering
   - Add `agentActionRuns[]` to current assistant message state.
   - Use the state machine above to render process text and chips.
   - Reuse the same component for live and historical messages.

5. Tests
   - Summarizer hides debug tools.
   - Visible tools map to correct labels.
   - `advance_layer` and `select_topic` produce state-action chips.
   - Session turn persistence includes `agent_actions`.
   - Old sessions without `agent_actions` render normally.

#### Acceptance Criteria

- Interviewer tool activity is visible during generation.
- Chips remain visible after completion and after session reload.
- The UI does not expose raw tool args, prompt text, model thought, or full tool output.
- Debug/precondition tools stay hidden.
- Full trace remains available for developer inspection.

## Historical Iterations

### 2026-06-22 DSML Tool-Call Leakage Recovery

Problem: DeepSeek sometimes emitted Anthropic-style DSML pseudo tool calls inside `content` during the reserved final step. Runtime treated the non-empty content as `final_answer`, so the UI displayed raw tool-call markup as if the answer had completed.

Decision: handle this as a provider protocol adaptation issue, not a semantic answer-quality issue.

Implemented:

- Parse DSML pseudo tool calls in the OpenAI-compatible adapter only when native `tool_calls` is empty.
- Enforce current allowed-tools whitelist and duplicate-call dedupe before execution.
- Execute parsed DSML only in normal tool steps.
- In reserved final step, never execute residual tool intent; stop as recoverable `max_steps` with `DirtyFinalToolIntent`.
- Let Librarian fallback generate a partial answer from completed observations, or a clear retry/narrow-scope message when no observations exist.
- Record observability fields such as `dsml_parsed`, `final_had_residual_tool_intent`, and `fallback_reason`.

Verification:

- `python -m py_compile ...`
- `python -m unittest knowledge_agent.tests.test_agent_runtime knowledge_agent.tests.test_librarian_agent`
- `python -m unittest discover -s knowledge_agent/tests`
- Frontend inline JS `node --check`

Detailed rationale is recorded in `DECISIONS.md` under `2026-06-22 Agent runtime DSML recovery decision`.
