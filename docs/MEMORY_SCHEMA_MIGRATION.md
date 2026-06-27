# Memory Schema & Migration

Learner Model **磁盘形状、ID、版本迁移**。实现 [`MEMORY_CHARTER.md`](MEMORY_CHARTER.md)；写入语义见 [`MEMORY_WRITE_CONSOLIDATION.md`](MEMORY_WRITE_CONSOLIDATION.md)。

---

## 1. 存储位置

| 文件 | 角色 | 阶段 |
|------|------|------|
| `profile/interview_profile.json` | **Legacy canonical**（v3） | 当前生产 |
| `data/profile/learner_model.json` | **Target canonical**（v4） | 迁移目标 |
| `data/profile/derived.json` | **Derived 缓存**（可删可重建） | 新 |
| `data/profile/commitments.jsonl` | **Commitment 事件 log**（optional 分文件） | 新；也可内嵌 v4 |

**原则**：Evidence 仍在 `data/interview-sessions/`、`data/answer-sessions/`、`data/review-runs/`；**不**搬进 profile 文件。

启动时：

1. 若仅有 legacy → 读 legacy，内存 normalize 为 v4 形状，**可选**写回 `learner_model.json`。
2. 若 v4 存在 → v4 为 SSOT；legacy 只读一次 import。

`InterviewProfileStore` 路径逐步改为 `data/profile/learner_model.json`（实现任务，非本文范围）。

---

## 2. Schema 版本

| 版本 | 文件 | 说明 |
|------|------|------|
| **v3** | `interview_profile.json` | `weak_points[]`、`sr` 内嵌、`topic_mastery` derived |
| **v4** | `learner_model.json` | `beliefs[]`、lifecycle、evidence_refs、procedures、commitments、derived 元数据 |

`schema_version: 4` 为 Learner Model canonical。

---

## 3. v4 顶层 Envelope

```json
{
  "schema_version": 4,
  "canonical_revision": 1,
  "updated_at": "ISO8601",
  "beliefs": [],
  "procedures": [],
  "strong_points": [],
  "commitments": [],
  "derived": {
    "generation": 0,
    "stale": true,
    "updated_at": "",
    "domains": []
  },
  "legacy": {
    "communication": { "style": "", "suggestions": [] },
    "topic_mastery": {}
  }
}
```

| 字段 | 说明 |
|------|------|
| `canonical_revision` | 每次 commit +1；derived.generation 对齐此值 |
| `beliefs[]` | 主 Belief 存储（含 Schedule 内嵌） |
| `procedures[]` | Procedure canonical |
| `strong_points[]` | MVP 保留；非 Charter 核心，面试 strengths |
| `commitments[]` | 事件；量大后可迁 `commitments.jsonl` |
| `derived` | 缓存 + stale 标记；完整地图可在 `derived.json` 双存 |
| `legacy.*` | 迁移期只读镜像；稳定后删除 |

**读取兼容**：`normalize_learner_model()` 接受 v3 入参，输出 v4 内存模型。

---

## 4. Belief 记录（`beliefs[]`）

### 4.1 字段

```json
{
  "id": "wp-8f3a2c1d",
  "kind": "standard",
  "lifecycle": "active",
  "point": "chunk 策略与 overlap 的关系说不清",
  "topic": "RAG 基础",
  "category": "knowledge_gap",
  "scope": "domain",
  "planned_layer": "component",
  "domain_anchor": {
    "plan_topic": "RAG 系统",
    "context_note_paths": ["个人/…/rag基础概念.md"],
    "scope_path": "个人/…/RAG"
  },
  "source_note_paths": ["…"],
  "source_session_ids": ["20260624-…"],
  "source_kinds": ["interview"],
  "times_seen": 2,
  "first_seen": "2026-06-20",
  "last_seen": "2026-06-24",
  "improved": false,
  "improved_at": "",
  "sr": {
    "interval_days": 3,
    "ease_factor": 2.5,
    "repetitions": 1,
    "next_review": "2026-06-27",
    "last_outcome": "pass",
    "last_reviewed": "2026-06-24"
  },
  "evidence_refs": [
    {
      "at": "2026-06-24T12:00:00+00:00",
      "source_kind": "interview",
      "session_id": "20260624-…",
      "turn_id": "turn-0003",
      "summary": "将 chunk 大小与语义完整性混为一谈"
    }
  ]
}
```

### 4.2 `kind: confusion_pair`

```json
{
  "id": "wp-cp-…",
  "kind": "confusion_pair",
  "lifecycle": "candidate",
  "left": "BM25",
  "right": "向量检索",
  "distinction": "…",
  "point": "BM25 vs 向量检索",
  "category": "knowledge_gap",
  "scope": "domain",
  "domain_anchor": { "…": "…" },
  "evidence_refs": [],
  "sr": { "…": "…" }
}
```

`point` = 人类可读标题（left vs right）；合并键用 `(left, right)` 无序对。

### 4.3 ID

```text
id = "wp-" + uuid4().hex[:8]   # 新 belief
```

迁移：legacy 无 id → 导入时生成，**稳定后不再变**。

### 4.4 Lifecycle 默认值（迁移）

| 来源 | lifecycle |
|------|-----------|
| 现有 `weak_points[]` | **active** |
| 新 interview write（双轨期） | **active**（与 legacy 行为一致）或配置 `default_interview_lifecycle=candidate` |
| 新 answer / review propose | **candidate** |

### 4.5 `source_kinds`

去重集合：`interview` | `answer` | `review` | `user`；每次 commit append。

Legacy 导入：`["interview"]`。

### 4.6 Schedule（内嵌 `sr`，逻辑独立）

Charter 将 **Schedule** 列为独立 canonical **类型**；Schema v4 **物理上** 将 `sr` 嵌在 Belief 条目中 — ** intentional 妥协**：

| 层 | 设计 |
|----|------|
| **逻辑** | Schedule 是独立概念；**仅** `schedule_pass` / `schedule_retry` / `improvement` 路径可写 `sr.*` |
| **物理** | `beliefs[].sr` 与 belief 同 id 绑定，避免 join 与 orphan schedule |
| **注入** | due 读取 `beliefs[].sr.next_review`；不单独 `schedules[]` 文件 |

**禁止**：extractor observation 直接改 `sr`（复查/面试 improvement 以外）；Belief 正文 (`point`, `category`) 与 `sr` 写入路径分离。

**Extraction checkpoint** 不存本文件：写在 Evidence session / review run 的 `memory_extraction`（见 Write doc §2.1）。

未来若 belief 数 >500 或需跨 belief 调度，可拆 `data/profile/schedules.json` 并以 `belief_id` 关联 — 非 MVP。

### 4.7 `improved` vs Commitment

- `improved=true`：Belief 字段（注入过滤：background 退出）
- 同时 append Commitment `{ action: "improved", belief_id }`
- 用户 deny → `lifecycle=archived`，**不** 删 `evidence_refs`

---

## 5. Procedure 记录（`procedures[]`）

```json
{
  "id": "pr-…",
  "lifecycle": "active",
  "key": "answer_structure.conclusion_first",
  "point": "先给结论，再分点展开",
  "scope": "universal",
  "source_kinds": ["user"],
  "evidence_refs": [],
  "updated_at": "ISO8601"
}
```

迁移 `communication` 块（**启发式，人工可改**）：

- `style` / 明显 **assistant 偏好** → `procedures[]`
- `suggestions[]` 中 **对用户 weak 的描述** → `beliefs[]`，`category=communication` 或 `answer_structure`，`lifecycle=active`

勿将全部 suggestions  blindly 迁入 procedures（见 Charter §3.3）。

---

## 6. Commitment 记录

```json
{
  "id": "cm-…",
  "at": "ISO8601",
  "action": "deny_belief",
  "belief_id": "wp-…",
  "payload": { "reason": "user_ui" },
  "source": "user"
}
```

MVP 可 cap `commitments[]` 最近 500 条；全量迁 jsonl。

---

## 7. Derived（`derived.json` 或内嵌）

```json
{
  "schema_version": 1,
  "generation": 12,
  "updated_at": "ISO8601",
  "domains": [
    {
      "scope_path": "个人/…/RAG",
      "plan_topic": "RAG 系统",
      "coverage": "partial",
      "confidence": 0.55,
      "known_facets": ["hybrid retrieval"],
      "unknown_facets": ["chunk 策略"],
      "active_belief_ids": ["wp-…"],
      "due_belief_ids": ["wp-…"],
      "user_override": null
    }
  ],
  "inject_blurbs": {
    "个人/…/RAG": "RAG 域：chunk 与召回仍有 2 处 active 弱项，1 处 due。"
  }
}
```

| 字段 | 说明 |
|------|------|
| `generation` | 等于 commit 时的 `canonical_revision` |
| `user_override` | 用户 edit 地图时的 Commitment 引用；路径 B |

**重建输入**：`beliefs`（active + due sr）+ `procedures` + `strong_points`；**不**读 archived。

`topic_mastery`（v3）→ 由 `domains[]` **替代**；迁移时 `recompute_topic_mastery` 结果投影进 derived 一次。

---

## 8. v3 → v4 迁移算法

```text
load v3 interview_profile.json
for each weak in weak_points:
  belief = {
    id: new wp-*,
    kind: standard,
    lifecycle: active,
    ...copy weak fields...,
    source_kinds: ["interview"],
    evidence_refs: migrate_evidence(weak.evidence, weak.source_session_ids)
  }
  beliefs.append(belief)

procedures = migrate_communication(v3.communication)
derived.stale = true
derived.domains = project_domains(beliefs)   # 替代 topic_mastery
canonical_revision = 1
write data/profile/learner_model.json
```

### 8.1 `migrate_evidence`

```text
for each legacy string in weak.evidence[]:
  evidence_refs.append({ summary, source_kind: interview, session_id: first source_session_id })
```

无 turn_id 的旧数据：`turn_id: ""`。

### 8.2 双轨读（过渡期）

```text
load():
  if learner_model.json exists:
    return normalize_v4(...)
  else:
    return migrate_v3_to_v4(interview_profile.json)

save():
  write learner_model.json only
  optional: mirror weak_points to interview_profile.json for rollback window
```

双轨窗口建议 **1–2 迭代**，之后只写 v4。

### 8.3 代码 alias

| v3 | v4 |
|----|-----|
| `weak_points` | `beliefs` |
| `weak_points[i].evidence` (string[]) | `evidence_refs` |
| — | `lifecycle` |
| — | `id` |
| `communication` | `procedures` |

`normalize_interview_profile()` 长期保留为 **v4 内存 normalize 的别名** 或 thin wrapper。

---

## 9. API / Store 契约（概念）

| 操作 | 输入 | 输出 |
|------|------|------|
| `LearnerModelStore.load()` | path | v4 dict |
| `LearnerModelStore.save()` | v4 dict | atomic write |
| `commit_observations()` | Observation[] | operations summary + new revision |
| `rebuild_derived()` | canonical | derived.json |
| `GET /api/memory/beliefs` | filter lifecycle | UI canonical |
| `GET /api/memory/derived` | — | derived + stale flag |
| `POST /api/memory/commitments` | user_commit | commitment + canonical |

---

## 10. 注入读取（与 Schema 关系）

Inject 层 **只读**：

```text
beliefs where lifecycle=active
procedures where lifecycle=active
commitments (recent override)
derived.inject_blurbs[scope] if not stale
```

每条 belief 注入字段：`point`, `planned_layer`, `category`, `sr.next_review`, `evidence_refs[-1].summary` — 见 Charter §8.4。

**不**把 `evidence_refs` 全量注入。

---

## 11. 非目标（本 schema）

- 不把 messages 嵌入 profile
- 不在 v4 存 agent trace
- 不在 MVP 拆 beliefs 到 SQLite（单 JSON 直到 >~500 active beliefs 再评估）

---

## 12. 验证（实现后）

```powershell
python -m pytest tests/test_learner_model_migration.py tests/test_memory_commit.py -q
python -m pytest tests/test_review_practice.py -q
```

手工：

1. 备份 `profile/interview_profile.json`
2. 启动迁移 → 检查 `data/profile/learner_model.json` beliefs 条数一致
3. 复查 pass → 仅 `sr` 变，`lifecycle` 不变
4. archived belief 不出现在 inject compact

---

## 13. 文档关系

```text
MEMORY_CHARTER.md          宪法
MEMORY_WRITE_CONSOLIDATION.md   何时写、Observation、合并
MEMORY_SCHEMA_MIGRATION.md      本文：长什么样、怎么迁
INTERVIEW_PROFILE_DESIGN.md     domain_anchor / scope 细节（只读参考，逐步 supersede）
```

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-24 | v0.1：Charter §12-3 首版 |
| 2026-06-24 | v0.2：Schedule 内嵌说明；communication 迁移启发式 |
