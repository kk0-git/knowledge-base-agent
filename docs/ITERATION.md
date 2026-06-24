# Iteration Log

## Current Iteration

### 2026-06-24 Phase A: Agent 生成与连接解耦

Implemented decoupled agent runs for interview v2 and librarian v2:

- **Ports**: `SessionRepository`, `AgentRunRepository` with `FileSessionRepository` + `InMemoryAgentRunRepository` adapters.
- **Services**: `AgentTurnRunner` + `AgentTurnService` orchestrate pending turn → background task → auto complete/fail (interview).
- **API**: `POST /api/agent/runs`, `GET /api/tasks/{task_id}/stream` (replay + live SSE).
- **Frontend**: chat interview/answer use runs + task stream; answer mode stores `active_answer_task_id` in `sessionStorage` for refresh recovery.
- **Compat**: `POST /api/agent/stream` retained for non-v2 paths.

Validation:

- `python -m pytest tests/test_session_repository.py tests/test_agent_turn_service.py tests/test_agent_run_stream.py -q`

### 2026-06-24 Review page three-state rebuild implemented

Implemented `/review` as a three-state container:

- `selecting`: default entry calls only `GET /api/review/due`, shows due topic overview, defaults all due topics selected, supports multi-select, select all, and clear.
- `card_review`: starts only after user selection, creates a new review run with `POST /api/review/plan` and fixed `topics`, then reuses grouped card polling, verify, and per-weak-point commit.
- `dialogue_review`: starts only after user selection, auto-kicks off `POST /api/review/dialogue` with fixed `topics` and no topic input box.

Backend compatibility:

- `ReviewDialogueRequest` now accepts `topics: string[]` while preserving legacy `topic`.
- `/api/review/dialogue` writes selected topics into working memory and tool context.
- `get_due_reviews` accepts `topics[]`, keeps legacy `topic`, and falls back to context `review_topics`.

Validation:

- `python -m py_compile src/web/app.py src/agent/tools/review.py`
- `python -m pytest tests/test_review_practice.py -q --tb=short`

Follow-up fix: dialogue review assistant bubbles now render lightweight Markdown (bold, emphasis, lists, horizontal rules, code, citations) while user bubbles remain escaped plain text.

### 2026-06-24 Review Rebuild Design: grouped cards, agent dialogue, incremental generation

Problem: the current `/review` implementation still behaves too much like a weak-point defect list. Even after pre-generating prompts, the page waits for all prompts before showing the first card, and single weak-point cards do not fit several review categories:

- `knowledge_gap` can be turned into a direct open-ended question.
- `answer_structure` often needs the previous interview answer/transcript and asks the user to restructure it.
- `thinking_pattern` should test the same thinking weakness in a new scenario.
- `communication` is better practiced by answering the same or similar question again.

Decision: split review into two product modes that share the same due/verify/commit foundation but do not share the same flow controller.

#### Mode B: card review

Card review should become a fixed review flow based on grouped weak points.

- User opens `/review`, sees due weak points grouped by topic, selects a topic, then starts review.
- Backend loads due weak points for that topic and groups them by `topic + planned_layer + category`.
- A card may contain multiple weak points when they belong to the same group.
- `knowledge_gap` produces an open-ended explanation question.
- `answer_structure` uses the previous interview answer or transcript context and asks the user to rewrite/restructure.
- `thinking_pattern` generates a new scenario testing the same reasoning pattern.
- `communication` asks the user to answer the same or similar question again.
- Cards may contain multiple question blocks when a group mixes related review needs.
- The card UI should stay simple: weak summary, generated question, answer box, submit, compact feedback, `confirm improvement` / `retry`.
- SM-2 updates remain per weak point. Even if the user answers one grouped card, each underlying weak point can be committed independently.

Card result shape should support per-weak-point outcomes:

```json
{
  "card_id": "card-1",
  "topic": "MCP 协议",
  "planned_layer": "调用链路与通信机制",
  "weak_point_ids": ["w1", "w2"],
  "question_blocks": [
    {"type": "knowledge_gap", "prompt": "..."},
    {"type": "answer_structure", "prompt": "..."}
  ],
  "reference_answer": "...",
  "strategy_tips": ["展开推理过程"]
}
```

Verification should return a card-level summary plus per-weak-point suggested actions:

```json
{
  "overall": "方向对了，但调用链路的响应格式仍缺失。",
  "weak_results": [
    {"weak_point_id": "w1", "suggested_action": "improve"},
    {"weak_point_id": "w2", "suggested_action": "retry"}
  ]
}
```

#### Mode A: dialogue review

Dialogue review should reuse Chat UI + Runtime, but not reuse card plans.

- User selects a topic and starts dialogue review.
- Runtime loads a reviewer skill, for example `skills/reviewer/SKILL.md`.
- Agent calls review tools such as `get_due_reviews`, `verify_weak_point`, and `commit_review_action`.
- Agent chooses weak point order, follow-up depth, and whether to keep drilling one weak point or move on.
- If the user answers well, the agent may commit improvement and skip similar weak points.
- If the answer is vague, the agent asks a more targeted follow-up.
- If the user cannot answer, the agent explains and marks the weak point for retry/partial progress according to the supported review policy.
- The dialogue ends with a summary of improved weak points and remaining practice targets.

#### Shared backend foundation

Both modes share data access, verification, and SM-2 writeback:

- `GET /api/review/due`: returns due weak points grouped by topic/category/layer.
- `POST /api/review/verify`: checks user answer against one or more weak points and returns structured feedback.
- `POST /api/review/commit`: updates one weak point at a time with `improve` or `retry`.
- Shared helpers group weak points and normalize review categories.

Card mode owns fixed plan generation:

- `POST /api/review/plan`: creates a card review run for a topic/range.
- `GET /api/review/plan/{run_id}`: returns the current run state.

Dialogue mode owns runtime control:

- It should not require `/api/review/plan`.
- It should use review tools and agent state to decide the next prompt dynamically.

#### Incremental generation capability

Decision: all generation-heavy review flows should use `shell first, ready incrementally`.

The current synchronous prepare behavior is not the target design. Waiting for all prompts before showing the first card creates avoidable latency and will get worse once cards contain grouped weak points, multiple question blocks, and reference answers.

Target flow:

```text
User selects topic
  -> POST /api/review/plan
  -> backend returns run shell immediately: card ids, weak summaries, status=pending
  -> backend generates cards in background
  -> frontend polls GET /api/review/plan/{run_id}
  -> first ready card is shown immediately
  -> remaining cards continue becoming ready while the user answers
```

Card status:

```json
{
  "card_id": "card-1",
  "status": "pending | ready | failed",
  "weak_point_ids": ["w1", "w2"],
  "question_blocks": [],
  "reference_answer": "",
  "error": ""
}
```

Implementation order:

1. Replace synchronous `/api/review/prepare` behavior with asynchronous review run creation.
2. Return a plan shell immediately.
3. Generate card prompts/reference answers in a background worker.
4. Let the frontend poll `GET /api/review/plan/{run_id}` and show the first `ready` card without waiting for the full run.
5. Add grouped weak-point cards and per-weak-point verify/commit.
6. Add dialogue review tools and reviewer skill.
7. Consider SSE only after polling behavior is stable; first version can use polling.

Acceptance criteria:

- The first ready card is usable before all cards finish generating.
- A failed card does not block the rest of the review run.
- Card review and dialogue review share due/verify/commit semantics but keep separate flow control.
- SM-2 remains per weak point, not per card.
- Generation and verification outputs are cached only for the current review run unless a later persistence design is explicitly added.

### 2026-06-23 Frontend IA Shell status element fix

Problem: after the shared shell wrapper stripped legacy page headers, Chat and Search lost their only `#status` nodes. Sending a message called `setBusy()`, which threw on null `#status` and aborted the SSE flow before any answer or process chips rendered.

Decision: keep page status elements outside stripped navigation regions and add null-safe guards in `setBusy()`.

Implemented:

- Moved Chat and Search `#status` elements into shell-safe locations inside `<main>`.
- Added null checks in Chat and Search `setBusy()` helpers.

Verification:

- `python tmp_verify_status.py` (rendered pages contain exactly one `#status` each)
- `python -m py_compile src/web/app.py`

### 2026-06-23 Frontend IA Topics/Wiki/Chat Cleanup Phases 3-6

Problem: Topics still exposed wiki maintenance actions, `/wiki` and `/admin/wiki` were not clearly separated as reading vs maintenance surfaces, Chat still exposed the legacy `study` mode, and `/audit` duplicated `/organize`.

Decision: finish the IA split. User-facing pages should read and navigate; maintenance actions stay under admin/maintenance routes. Review is an independent `/review` flow, not a Chat mode.

Implemented:

- Changed Topics into a browsing page: overview stats, read links to `/wiki?tag=...`, Obsidian open links, and no synthesize/re-synthesize POST actions.
- Added a read-only Wiki reader at `/wiki`; kept `/admin/wiki` as the maintenance surface.
- Removed the `study` option from the Chat mode selector while preserving legacy `mode=study` redirects to `/review`.
- Redirected `/audit` to `/organize` and kept organize as the canonical note-organization route.
- Updated command docs to point users to `/organize`, `/review`, `/wiki`, and `/admin/wiki`.

Verification:

- `python -m py_compile src/web/app.py`
- Inline JS syntax checks for Topics, Wiki reader, and Chat pages.

### 2026-06-23 Frontend IA Search Phase 2

Problem: `/search` was still presented as a retrieval-debug page, exposing dense/BM25/RRF parameters before the user-facing search results.

Decision: keep one route with two entry modes. `/search` and `/search?q=...` are user-facing search; `/search?debug=true` preserves the existing debug surface and opens advanced settings.

Implemented:

- Added `debug=true` mode detection and auto-opened advanced search settings only in debug mode.
- Changed default search rendering to three user-facing columns: note snippets, generated Wiki pages, and tag matches.
- Added `?q=` prefill and auto-search support for global search.
- Kept debug stage tabs, rewrite metadata, and full result cards available only in debug mode.
- Used `/api/wiki/report` client-side filtering for Wiki and tag result columns without adding a new API.

Verification:

- `python -m py_compile src/web/app.py`

### 2026-06-23 Frontend IA Chat Routing Phase 1

Problem: Chat was still internally defaulting to interview mode, `/chat` remained a user-facing duplicate route, and sidebar mode entries could not reliably initialize or reflect Chat mode.

Decision: make `/` the canonical Chat route. Keep `/chat` as a compatibility redirect, and treat `mode` as a URL-level preset for the shared Chat UI.

Implemented:

- Changed Chat default mode to QA (`answer`) and updated the initial assistant message.
- Added URL mode initialization for `/?mode=answer` and `/?mode=interview`.
- Redirected `/chat` to `/?mode=answer`, `/chat?mode=interview` to `/?mode=interview`, and study-mode legacy entries to `/review`.
- Synced Chat mode changes back into the URL and sidebar active state.
- Synced restored active interview sessions to `/?mode=interview`.

Verification:

- `python -m py_compile src/web/app.py`

### 2026-06-23 Frontend IA Shell Phase 0

Problem: web pages were separate tool surfaces with duplicated top navigation, no grouped entry point, no global search, and no shared current-state highlighting.

Decision: start with a shared shell wrapper instead of rewriting each inline HTML page. Existing page bodies remain intact while routes render through a Comet-style sidebar, sticky header, and global search.

Implemented:

- Added shared `SHELL_CSS`, `SIDEBAR_HTML`, shell script, and `render_web_page()` in `src/web/app.py`.
- Routed existing pages through the shared shell (`/`, `/chat`, `/search`, `/topics`, `/wiki`, `/admin/wiki`, `/organize`, `/audit`, `/settings`).
- Added `/review` placeholder page so the Phase 0 sidebar entry does not 404 before Phase 1.5.
- Added sidebar active-state logic for grouped navigation and `debug=true` search mode.

Verification:

- `python -m py_compile src/web/app.py`

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
