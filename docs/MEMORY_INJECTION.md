# Learner Memory Injection

Prompt injection rules for skills that read canonical learner memory. Normative budgets and reader matrix live in [`MEMORY_CHARTER.md`](MEMORY_CHARTER.md) §8; this document specifies **what each skill receives** and **how it is rendered**.

## Global rules

| Rule | Behavior |
|------|----------|
| **candidate** | Never injected |
| **archived** | Never injected |
| **improved** active beliefs | Excluded from active-belief selection (`is_injectable_weak_point`) |
| **Commitment** | Injected when present (Interviewer / Reviewer); highest selection priority |
| **Derived** | One short domain blurb per turn when available |
| **Procedure (active)** | Injected within per-skill budget |

Implementation: `src/services/memory/injection.py`.

## Reader matrix (summary)

| Reader | Inject? | active Belief | due | Procedure | derived | Commitment |
|--------|---------|---------------|-----|-----------|---------|------------|
| **Interviewer** | yes | ≤5 | merged in list | ≤2 | 1 blurb | yes |
| **Librarian** | yes | ≤3 | ≤2 markers | ≤2 | 1 blurb | — |
| **Coach** | **no** | — | — | — | — | — |
| **Reviewer** | yes (future) | ≤5 | merged | — | 1 blurb | yes |

Coach stays at **zero memory** — turn transcript + turn context only.

## Interviewer

**Wiring:** `build_interviewer_runtime_context()` → `render_interviewer_memory_context()` → `# Learner Memory Background` section in `build_turn_input()`, ahead of `# Runtime Context` JSON.

**Scope:** current topic name + plan `scope_note_paths`. Domain beliefs filtered by `domain_relevance_for_current`; universal beliefs always eligible.

**Budget (Charter §8.3–8.4):**

| Slice | Limit |
|-------|-------|
| Active beliefs | ≤5 (due marked inline, no extra slots) |
| Procedures | ≤2 |
| Derived | 1 domain blurb |
| Commitments | ≤3 recent notes |

**Belief line shape:**

```text
- {point} [due] (probe: …; layer: …)
  latest evidence: {most recent evidence_refs summary}
```

`confusion_pair` renders as `{left} vs {right}: {distinction}`.

**Division with `recall_profile`:**

| Preloaded in unified block | On demand via `recall_profile` |
|----------------------------|--------------------------------|
| Top universal + topic-relevant active beliefs (bodies) | Full domain weak-point bodies for **current planned layer** when `profile.current_layer_domain_weak_count > 0` |
| Derived blurb, procedures, commitments | Layer-specific detail beyond the ≤5 budget |

When the unified block is non-empty, runtime JSON **omits** `profile.universal_weak_points` bodies (layer counts remain). `tool_boundaries.preloaded` lists `learner_memory_background` instead of `universal_profile_weak_points`.

**Boundary text (probe-oriented):** use memory quietly to shape probes; do not mention memory unless the user asks; do not turn beliefs into a checklist.

## Librarian

**Wiring:** `AgentTurnRunner.run_answer()` → `render_librarian_memory_context()` → `LibrarianRequest.learner_memory_context`.

**Budget:** active beliefs ≤3, due markers ≤2 (may skip lower-ranked due items), procedures ≤2, derived blurb.

**Boundary text (answer-oriented):** let the user's question and note evidence drive the answer; do not force beliefs into the reply.

## Answer session memory (write, not inject)

Answer memory commits on **`POST /api/answer/sessions/{id}/archive`** only. Archived answer sessions do not inject on their own; extracted beliefs enter the canonical model and may appear in a later Interviewer/Librarian turn after promotion to **active**.

## Selection priority (all injectors)

```text
user Commitment / override
  > due-marked active Belief
  > other active Belief
  > derived blurb
  > Procedure
```

Within each tier, `sort_weak_points_for_prompt` orders by due date, ease factor, and recency.

## Tests

```powershell
$env:PYTHONPATH="src"
python -m pytest tests/test_librarian_memory_injection.py tests/test_interviewer_memory_injection.py tests/test_memory_ui.py tests/test_memory_commitments_api.py -q
```

## `/memory` UI (P2)

| Section | Actions | Evidence |
|---------|---------|----------|
| Candidate Beliefs / Procedures | confirm / deny | `evidence_links` → interview / answer / review URLs |
| Archived Beliefs / Procedures | restore | same evidence links (read-only) |

API: `GET /api/memory/archived` returns the archived panels; candidates already include `evidence_links` via `enrich_memory_item_for_ui`.
