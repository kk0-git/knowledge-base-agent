# Memory Write & Consolidation

记忆 **写入与合并** 细则。实现 [`MEMORY_CHARTER.md`](MEMORY_CHARTER.md) §6–§7；不重复注入矩阵（见 Charter §8）。

**相关代码（Legacy / 待 refactor）**：`interview_profile.py`（`extract_profile_observations`、`apply_profile_observations`）、`review_practice.py`（`advance_review_schedule`）、`interview_memory_commit.py`。

---

## 1. 核心分界

```text
Evidence（只读输入）
    ↓
Extractor → Observation（提议，非 canonical）
    ↓
Commit engine（规则 + 确定性合并）→ Canonical 变更
    ↓
Persist → 标记 Derived 过期
```

| 阶段 | 谁做 | 输出 |
|------|------|------|
| **Extract** | LLM 或规则 | `Observation[]` |
| **Commit** | 代码（铁律 enforced） | 对 Belief / Procedure / Schedule / Commitment 的变更 |
| **Persist** | Store | 磁盘 canonical + `canonical_revision` +1 |

**Extractor 永远不能**：直接改 `lifecycle` 为 active（除 Charter 允许的高置信 answer 标记在 observation 上，由 commit 执行）；写 `domain_anchor`；写 `improved`；Review pass 时改 Belief 正文。

---

## 2. 写入流水线（统一）

所有触发源走同一管道：

```text
1. TriggerGate      — 该事件是否允许写 memory？（Charter §6）
2. GatherEvidence   — 读 messages / reviews / run events；不读 workspace
2b.ExtractionCheckpoint — 该 evidence 包是否已 extract 过？是则跳过 Extract（§2.1）
3. Extract          — 产出 Observation[]（**空数组为正常**）
4. Validate         —  schema、confidence、iron rules 预检
5. ResolveTargets   — find_similar / confusion pair / archived 规则
6. ApplyLifecycle   — candidate | active | archived 迁移
7. ApplySchedule    — 仅允许路径写 sr.*
8. AppendCommitment — 用户操作 / improved / deny / derived-edit
9. PersistCanonical — 原子写盘（含 extraction checkpoint）
10. MarkDerivedStale — derived.generation < canonical_revision
```

### 2.1 Extraction checkpoint（Evidence 侧，非 canonical）

**空 `Observation[]` 是正常结果** — 表示「已检查，本段 evidence 无 durable 信号」，**不是**错误。

为避免同一 session 重复跑 extractor，在 **Evidence 源**（非 Learner Model）写入 checkpoint：

```json
{
  "memory_extraction": {
    "extracted_at": "ISO8601",
    "trigger": "interview.session_ended",
    "evidence_hash": "sha256(messages+reviews 快照)",
    "observation_count": 0,
    "filtered_low_count": 2,
    "commit_revision": null
  }
}
```

| 字段 | 说明 |
|------|------|
| `evidence_hash` | messages/reviews 变更 → hash 变 → 允许 re-extract |
| `observation_count=0` | 正常；仍更新 checkpoint |
| `commit_revision` | 若 commit 成功，写入 canonical_revision；便于审计「检查过但无写入」 |

存放位置（实现选一）：`session.domain.memory_extraction`（面试/问答 session JSON）；review run 顶层 `memory_extraction`。**不**写入 `beliefs[]`。

失败策略：

- Extract LLM 失败 → 按流降级（见 §4–§6），**不** 部分写 active（可写 candidate 降级包，optional）。
- Persist 失败 → 整批回滚，Evidence 不受影响。

---

## 3. Observation 契约

Extractor **只输出 Observation**，不输出 canonical 行。

### 3.1 公共字段

```json
{
  "op": "propose_belief | update_belief | propose_procedure | schedule_pass | schedule_retry | improvement | partial | strong_point | user_commit",
  "confidence": "low | medium | high",
  "source_kind": "interview | answer | review | user",
  "source_ref": {
    "session_id": "",
    "turn_id": "",
    "review_run_id": "",
    "card_id": ""
  },
  "evidence_summary": "一句摘要，供 evidence 链与注入 latest 用"
}
```

`low` confidence：**不进入 candidate**（与现有 profile extractor 一致），但 **非静默**：

1. 计入本次 `memory_extraction.filtered_low_count`；
2. 可选写入 **extraction 诊断 log**（`eval-results/` 或 debug trace，非 canonical）；
3. **预留**（Charter §3.6）：同一 merge 键连续 N 次 low → 人工审核队列（MVP 不做）。

`medium` / `high`：进入 commit 管道；是否 active 由流与铁律决定（§3.2、§7.3）。

### 3.2 `propose_belief`

```json
{
  "op": "propose_belief",
  "kind": "standard | confusion_pair",
  "point": "standard 时必填",
  "left": "confusion_pair",
  "right": "confusion_pair",
  "distinction": "confusion_pair",
  "category": "knowledge_gap | answer_structure | communication | thinking_pattern",
  "scope_suggestion": "domain | universal",
  "topic": "可读标签",
  "planned_layer": "",
  "target_lifecycle": "candidate | active",
  "contradicts_belief_id": "optional，高置信取代旧 active 时填",
  "weak_point_ref": "合并到已有 belief 时的 point/id 提示"
}
```

- **`target_lifecycle=active`**：仅当 `source_kind=answer` 且 `confidence=high`，或 commit 阶段 **多源 merge** 满足条件（§7）。
- 面试 session-end：默认 `target_lifecycle=candidate`；commit 引擎可对 `medium+` 且与现有 candidate 合并的条目升 active（与 Charter「面试结束 candidate/active」一致，implementer 可配置：面试 weak 默认 active 以保持 legacy 行为，见 Schema 迁移双轨期）。

**Legacy 对齐说明**：当前 `apply_profile_observations` 对新 weak **直接 active**（无 lifecycle 字段）。迁移期 commit 引擎对 `source_kind=interview` 可保留 **默认 active**，对新流 **默认 candidate**；见 [`MEMORY_SCHEMA_MIGRATION.md`](MEMORY_SCHEMA_MIGRATION.md) §4。

### 3.3 `schedule_pass` / `schedule_retry`

```json
{
  "op": "schedule_pass | schedule_retry",
  "belief_id": "wp-...",
  "evidence_summary": "review card pass / retry"
}
```

**仅写 `sr.*`**，不修改 `point` / `lifecycle` / `improved`（Decision B）。

### 3.4 `improvement`

```json
{
  "op": "improvement",
  "belief_id": "或 weak_point_ref",
  "evidence_summary": ""
}
```

**仅** 面试 session-end commit 路径调用 → `mark_weak_point_improved` 语义（Schedule + 条件 `improved`）。

### 3.5 `user_commit`

```json
{
  "op": "user_commit",
  "action": "confirm_candidate | deny_belief | restore_archived | edit_domain_mastery | set_procedure",
  "belief_id": "",
  "payload": {}
}
```

写入 **Commitment 事件** + lifecycle / canonical 变更（Derived 路径 B）。

---

## 4. 面试流（Interview）

### 4.1 触发

| 触发 | 条件 |
|------|------|
| `interview.session_ended` | `status=ended`，正常结束 API |

不进行：`turn_complete` 写 profile（Coach 零 write）。

### 4.2 Evidence 包

```text
session.messages[]
session.interview_plan / interview_state
reviews[]（.reviews.json）：feedback.gaps, question_requires, coach_note, context_note_paths
existing canonical（compact）
```

### 4.3 Extract

- 主路径：`extract_profile_observations`（LLM），映射为 Observation[]。
- 降级：gaps + coach_note 规则合成 `propose_belief`（`confidence=medium`）。

### 4.4 Commit 规则

| Observation | Commit |
|-------------|--------|
| `propose_belief` | 合并键命中 **active/candidate** → UPDATE evidence；命中 **archived** → **新建 candidate**（不复活） |
| 新 belief | Legacy：active；目标 schema：`candidate` 或 `active`（迁移开关） |
| `improvement` | `mark_weak_point_improved` |
| `partial` | `mark_weak_point_partial` |
| `strong_point` | `strong_points[]` upsert（非 Belief，MVP 保留） |
| `communication` 块 | 拆分为：`propose_belief`（weak → `category=communication|answer_structure`）或 `propose_procedure`（assistant 风格偏好）；见 Charter §3.3 |

`domain_anchor`：**仅** `prepare_observation_for_write` / `new_weak_point` 代码写入。

### 4.5 scope 升级

保留现有 `promote_repeated_domain_habits()`：domain  habit 多域复现 → `scope=universal`（**非 LLM**）。

---

## 5. 问答流（Answer）

### 5.1 触发

| 触发 | 条件 |
|------|------|
| `answer.session_ended` | 正常结束；或 MVP **每 session 一次** at end |
| `answer.session_idle` | 可选后期：N 轮无新 message 增量 extract（**不** MVP） |

崩溃 / 未结束：**不写**。

### 5.2 Evidence 包

```text
answer.messages[]
assistant.citations[]（note paths）
session.context（scope）
existing canonical compact
```

### 5.3 Extract

- LLM `AnswerMemoryExtractor`（待建）：从 transcript 产 `propose_belief` / `propose_procedure`。
- 规则高置信（无需 LLM 也可 commit）：
  - 用户明确「不懂/不会 X」→ `propose_belief`, `confidence=high`, `target_lifecycle=active`
  - 用户回答与 citation 知识点 **明显矛盾** 且 assistant 已纠正 → `high` + `active`

### 5.4 Commit 规则

| 情况 | 结果 |
|------|------|
| 默认 | `lifecycle=candidate` |
| `high` + `target_lifecycle=active` | active；若与旧 active 矛盾 → 旧 **archived** |
| 与 **candidate** 合并（interview 同源 weak） | **自动 merge → active**（Charter §7.2） |
| 命中 **archived** | **新建 candidate** |
| domain_anchor | 有 citation → `resolve_anchor_from_paths`；无 → `scope=universal`, candidate |

**禁止**：`improvement` op、`schedule_pass` 写 improved。

---

## 6. 复查流（Review）

### 6.1 触发

| 触发 | 事件 |
|------|------|
| Card verify pass | `schedule_pass` |
| Card retry / fail | `schedule_retry` |
| Card verify 发现新混淆 | `propose_belief`, candidate |
| Dialogue turn `suggested_commits` | `propose_belief`, candidate（persist 于 run 完成或 commit API） |
| Dialogue session 结束 | 批量 commit candidate proposals |

### 6.2 Evidence 包

```text
review run: type, topics, cards[], results[], messages[]（dialogue）
weak_point_id 绑定（card）
existing canonical
```

### 6.3 铁律 enforced

```text
Review pass/retry     → 仅 ApplySchedule（advance_review_schedule / retry 规则）
Review 新发现         → 仅 candidate Belief
```

`commit_review_action` / `commit_review_outcome` **不得** 调用 `mark_weak_point_improved`。

### 6.4 Schedule retry 语义

- retry：`sr.repetitions` 不减为负；`next_review` 拉近（与现有 review_practice 一致）；`last_outcome=fail`。
- pass：`advance_review_schedule` only。

---

## 7. 合并（Consolidation）

### 7.1 standard Belief

沿用 `find_similar_weak_point` 逻辑，Charter 三键：

1. `scope` 相同  
2. `domain_anchor` 相容（`domain_anchor_matches` / scope_path 前缀）  
3. `SequenceMatcher(point)` ≥ **0.72**（现有阈值，可配置）

**lifecycle 参与**：

- 合并池：**active + candidate**（不含 archived）
- archived 永不入池

### 7.2 confusion_pair

```text
key = tuple(sorted([normalize(left), normalize(right)]))
```

同 key → UPDATE `distinction`、append evidence；不新建行。

### 7.3 多源升 active

```text
IF belief_id 已有 candidate from interview
AND new observation from answer (any confidence ≥ medium)
AND merge key match
THEN lifecycle = active
```

单源 answer **仅** `high` → active。

### 7.4 Evidence 链

每条 Belief 维护：

```json
"evidence_refs": [
  {
    "at": "ISO8601",
    "source_kind": "interview",
    "session_id": "...",
    "turn_id": "...",
    "summary": "..."
  }
]
```

Legacy 字段 `evidence`（字符串数组）迁移时并入 `evidence_refs`。**注入** 只用 `evidence_refs[-1]`。

### 7.5 冲突

新 active 指定 `contradicts_belief_id` 或 merge 后发现语义对立 → 旧 active **archived**，append Commitment `superseded_by`。

---

## 8. Procedure 写入

**仅** assistant 交互偏好，**不**承载 thinking_pattern / communication **类 weak**（那些是 Belief `category`）。

来源：

1. Interview extractor：明确 **assistant 风格** 的 `communication.suggestions` → `propose_procedure`（candidate）
2. Answer extractor：用户 **要求** 的回答格式（非 weak 诊断）→ candidate
3. 用户 `user_commit.set_procedure` → **active**

**不** 把「听题不仔细」「选型简单归因」写入 Procedure — 应为 `propose_belief` + 对应 `category`。

合并：按 `procedure_key`（如 `answer_format.conclusion_first`）或 `point` 相似；active + candidate 池。

Lifecycle 与 Belief 相同；**默认 candidate 不注入**。

---

## 9. Commitment 事件

Append-only 语义（实现可同文件 `commitments[]` 或 rolling log）：

```json
{
  "id": "cm-...",
  "at": "ISO8601",
  "action": "confirm_candidate | deny_belief | restore_archived | improved | domain_mastery_edit | superseded_by",
  "belief_id": "wp-...",
  "payload": {},
  "source": "user | system"
}
```

| action | Canonical 效应 |
|--------|----------------|
| `confirm_candidate` | belief.lifecycle → active |
| `deny_belief` | → archived |
| `restore_archived` | archived → active（仅用户） |
| `improved` | 面试 improvement 路径；配合 `improved=true` |
| `domain_mastery_edit` | Derived 路径 B：改 belief 或 domain 汇总字段 |
| `superseded_by` | 旧 belief archived，记录新 id |

---

## 10. 与 Derived 的衔接

Commit 步骤 10：**不** 同步重建 Derived。只递增 `canonical_revision` 并将 `derived.stale=true`。

Inject 路径见 Charter §4.1。

---

## 11. 实施阶段建议

| 阶段 | 内容 |
|------|------|
| **W0** | Observation 类型 + Commit 铁律单测（mock canonical） |
| **W1** | Interview 路径挂 lifecycle + evidence_refs；Legacy active 默认 |
| **W2** | Review schedule 与 candidate propose 分离；禁止 review improved |
| **W3** | Answer session-end extractor + candidate/high active |
| **W4** | Procedure 从 communication 拆出；Commitment API |
| **W5** | 多源 merge 升 active；confusion_pair kind |

---

## 12. 验证（实现后）

```powershell
python -m pytest tests/test_memory_commit.py tests/test_review_practice.py tests/test_conversation_schema.py -q
```

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-24 | v0.1：Charter §12-1 首版 |
| 2026-06-24 | v0.2：Belief category vs Procedure；extraction checkpoint；low 非静默 |
