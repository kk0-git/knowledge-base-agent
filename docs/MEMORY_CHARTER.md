# Memory Charter

Learner Memory 体系宪法。定义 **记什么、真相源在哪、何时持久化、如何合并、如何注入**。工程 schema 与实现见后续迭代；本文仅顶层契约。

**前置文档**：对话证据层见 [`CONVERSATION_STORAGE.md`](CONVERSATION_STORAGE.md)；面试 weak point 字段细节见 [`INTERVIEW_PROFILE_DESIGN.md`](INTERVIEW_PROFILE_DESIGN.md)（逐步迁入本 Charter 的 Belief/Schedule 模型）。

**范围**：面试、问答、复查（卡片 + 对话）**共用一套 canonical Learner Model**。

---

## 1. 目的与 Non-goals

### 1.1 目的

Memory 系统保存 **关于用户作为学习者/面试者的稳定判断**，供各 Agent 在后续 session 中 **选择性注入**，使追问、讲解、复查选题更贴合个人薄弱与偏好。

### 1.2 Non-goals

- **不是** RAG：笔记里的客观知识仍由检索索引提供。
- **不是** Evidence：对话原文、turn review、card 事件留在 `data/` 会话文件与 artifacts；memory 不复制 transcript。
- **不是** Session 工作区：当前页选中、未提交草稿、sessionStorage workspace 不参与 memory 写入。
- **不** 自动修改用户 Obsidian 笔记。
- **不** 持久化模型 hidden reasoning / 完整 trace（调试 trace 另存，见 Conversation Storage）。

---

## 2. 与相邻系统边界

| 系统 | 职责 | 持久化 |
|------|------|--------|
| **Evidence** | 可审计对话链：`messages[]`、`.reviews.json`、review run events | 长期保留 |
| **Memory (canonical)** | 对用户的 Belief / Procedure / Schedule / Commitment | 可 active / archived，不删 evidence |
| **Derived** | 域摘要、掌握度视图、注入用 compact | 可重建；用户编辑走 Commitment |
| **RAG** |  vault 客观知识 | 索引目录 |
| **Settings** | 用户显式产品配置 | 配置文件 |

**原则**：Memory indexer / extractor **只读** Evidence + 已有 canonical；**不读** workspace 快照、localStorage 聊天草稿。

---

## 3. Canonical 模型

三流共享 **一套** Learner Model（磁盘位置实现阶段再定；当前 legacy 投影为 `profile/interview_profile.json`）。

### 3.1 四类 Canonical

| 类型 | 含义 | 示例 |
|------|------|------|
| **Belief** | 对用户知识/思维/表达的判断 | 「chunk 策略说不清」；混淆对 A/B |
| **Procedure** | **Assistant 应如何呈现/交互**（非对用户能力的判断） | 「讲解时先给结论」；「对比用表格回答」 |
| **Schedule** | 与 Belief 绑定的复习调度（SM-2 等） | due 日期、interval、repetitions |
| **Commitment** | 已确认的状态变更 | `improved`；用户「这不是弱项」；derived 编辑落盘 |

### 3.2 命名迁移

- Legacy `weak_points[]` 在概念上 **改称 Belief**（可含内嵌 Schedule 字段）。
- Charter 层允许 Belief + Schedule **逻辑分离**；实现可仍同条记录，但 **复查 pass 仅写 Schedule，不写 Belief 正文**（见 §6、§7）。

### 3.3 Belief：`kind` × `category`（勿与 Procedure 混淆）

Belief 有两层分类：

| 层 | 字段 | 取值 | 含义 |
|----|------|------|------|
| **结构** | `kind` | `standard` \| `confusion_pair` | 记录形状 |
| **语义** | `category` | 见下表 | 对用户的 **能力/思维/习惯** 判断 |

**`category`（均为 Belief，`propose_belief`）**：

| category | 含义 | 典型 scope | 示例 |
|----------|------|------------|------|
| `knowledge_gap` | 知识点/机制缺口 | 多为 `domain` | 「说不清 chunk 与 overlap」 |
| `answer_structure` | 答题组织弱 | `domain` → 可升 `universal` | 「对比题只列点、不给结论」 |
| `thinking_pattern` | **对用户思维方式的判断** | `domain` 或 `universal` | 「选型时简单归因、不讲 trade-off」 |
| `communication` | **对用户沟通/听题习惯弱** | 多为 `universal` | 「听题不仔细、答非所问」 |

**Procedure vs Belief**：

| | Belief | Procedure |
|---|--------|-----------|
| 主语 | 用户 **怎样**（弱/习惯/思维） | Assistant **应怎样答/问** |
| 例 | 「听题不仔细」→ `category=communication` | 「少追问、直接完整答」→ Procedure |
| 例 | 「选型简单归因」→ `thinking_pattern` | ❌ 不是 Procedure |

Extractor 产出 `communication` 块时 **先判别**：诊断 weak → Belief；指定 assistant 风格 → Procedure。

### 3.4 Belief 子类型 `kind`（MVP）

| `kind` | 说明 |
|--------|------|
| `standard` | 单点 Belief；`category` 见 §3.3 |
| `confusion_pair` | 混淆对；`left` / `right` / `distinction`；合并键 **无序对** |

### 3.5 审计字段

每条 Belief（及 Procedure）应带 **`source_kind`**：`interview` | `answer` | `review` | `user`（实现阶段落 schema）。

**Trajectory** 不单独作为 canonical 类型：由 **evidence 链（全 turn_id）+ Commitment 时间线 + Belief 状态** 表达。

### 3.6 预留：低置信提取（非 MVP）

Extractor 标记 `confidence=low` 的 observation **不写入** canonical，但应 **记 extraction 日志**（见 Write doc）。  
**预留**：同一 merge 键连续 **N 次**（建议 N=3）均为 low → 「人工审核队列」（MVP 不实现）。

---

## 4. Derived

Derived 是 **从 canonical 计算出的视图**，不是第二套真相源。

| 示例 | 用途 |
|------|------|
| 域摘要（epistemic blurb） | 注入时 1–3 句概况 |
| 掌握度 / 边界地图 | UI 与排序；可标注 known / unknown facets |
| inject compact | 按 agent 预算压缩后的 prompt 片段 |

### 4.1 重建策略

1. **Canonical 任意变更后**：标记 derived **过期**。
2. **注入前**：若 derived 版本落后于 canonical → **先重建再注入**。
3. **用户手动刷新**：允许；不替代自动过期机制。

### 4.2 用户编辑 Derived

采用 **路径 B**：

```text
用户编辑 derived → 写入 Commitment → 更新 canonical → 重建 derived
```

禁止「只改 derived、下次重建被 canonical 覆盖」的无效编辑（路径 A）。

### 4.3 可见性

Derived **对用户可见、可操作**；Canonical 与 Evidence 见 §10。

---

## 5. Lifecycle

MVP **三态**（`dormant` 后续迭代再加）：

```text
candidate → active → archived
```

| 状态 | 含义 | 默认注入 |
|------|------|----------|
| **candidate** | 系统怀疑成立，尚未正式启用 | **否** |
| **active** | 参与调度与注入 | 是（按 §8 预算） |
| **archived** | 用户否决、被新 belief 取代、或显式归档 | **否** |

### 5.1 状态迁移（摘要）

- **candidate → active**：用户确认；面试 session 结束 extractor；问答 **高置信** 单源；**多源**（如 interview candidate + answer 同 weak）自动 merge 升 active。
- **active → archived**：用户否认；与新 active **矛盾** 时以新为准旧 archived；Interview verify improved + Commitment（见 §7）。
- **archived → active**：**仅** 用户手动恢复；**禁止** 系统自动复活。
- **archived 后再出现同类**：**新建 candidate**，不合并进 archived 条目。

### 5.2 Candidate 定义

**Candidate memory** = 已持久化但尚未 active 的 Belief/Procedure。默认 **不注入** prompt；可在 UI「待确认记忆」展示（产品阶段）。

---

## 6. 持久化时机

| 事件 | 允许写入 |
|------|----------|
| 面试 session **正常结束** | candidate / active Belief；Procedure 候选 |
| 问答 session **正常结束**（或实现阶段定义的批量点） | 默认 **candidate** Belief；**高置信** → active（见 §7） |
| 复查 pass / retry | **仅 Schedule** |
| 复查新发现（verify / dialogue） | **candidate** Belief |
| 复查 dialogue `suggested_commits` | **candidate** Belief（非仅 UI） |
| 用户确认 / 否认 / 编辑 derived | active / archived / Commitment |
| Session **未正常结束**（崩溃、强关） | **不写** memory；Evidence 已在 messages |

**高置信（问答 → active）示例**：用户明确自承不懂；回答 **明显全错** 且与知识点直接对应。

---

## 7. 四条铁律与升 active 优先级

### 7.1 四条铁律

```text
Review pass/retry     → 仅 Schedule
Review 新发现         → 仅 Candidate Belief
Answer 默认           → Candidate（高置信例外 → Active）
Belief 降级/归档      → 仅 Interview verify | User commit | 多源升 active
```

与 **Decision B** 一致：复查 pass **不** 设置 `improved`；**不** 因复查改善而自动更新 active Belief 正文。

### 7.2 升 active 优先级

```text
问答高置信单源 → active
    高于
双源 candidate 自动 merge → active
```

### 7.3 证据源权重（写入与合并参考）

```text
面试 turn review + transcript  >  问答 transcript  >  复查 card（Schedule 为主）
```

问答 **单条非高置信** 不得 alone 将 candidate 升为 active（须 merge 或用户确认）。

### 7.4 冲突

- 新 **active** 与旧 **active** 矛盾 → **以新为准**，旧 **archived**。
- 已 **archived** 再被 extractor 挖到同类 → **新建 candidate**。

### 7.5 Belief 合并键

**standard Belief** 合并需同时满足：

1. **文本相似**：`point` 表述同一薄弱（实现可用相似度；Charter 不限算法）。
2. **scope 一致**：`universal` vs `domain` 不跨 scope 合并。
3. **domain_anchor 相容**：domain Belief 须同一知识域（legacy 见 `INTERVIEW_PROFILE_DESIGN.md` 三元组）。

**confusion_pair**：无序对 `(left, right)` 相同则 merge，更新 `distinction` 与 evidence。

合并后 **evidence 链保留全部 `turn_id`**；**注入时每条 Belief 仅附最近一次 evidence 要点**（非全链灌 prompt）。

---

## 8. Retrieval & Injection

### 8.1 全局

- **candidate**：默认 **不注入**。
- **archived**：不注入。
- **user Commitment / override**：注入，且优先级最高。

### 8.2 读者矩阵

| 读者 | candidate | active Belief | due（Schedule） | Procedure (active) | derived 摘要 | Commitment |
|------|-----------|---------------|-----------------|---------------------|--------------|------------|
| **Interviewer** | 否 | 是 | 是（与 belief 合并排序） | 是 | 是 | 是 |
| **Coach** | 否 | 否 | 否 | 否 | 否 | 否 |
| **Reviewer** | 否 | 是 | 是 | 否 | 是 | 是 |
| **Librarian** | 否 | 是 | 是（弱） | 是 | 是（短） | 是 |

**Coach**：**永远零 memory** — 仅当轮 transcript + turn 内 context，避免 profile 污染 turn review。

**Librarian due**：due 标记用于 **优先讲透** 近期该巩固的 active Belief（不是安排复习）；与 Schedule 数据同源。

### 8.3 单轮预算

| 读者 | active Belief | due 标记 | Procedure | derived |
|------|---------------|----------|-----------|---------|
| Interviewer | ≤ **5** | 与 belief 合并排序，总数仍 ≤5 | ≤ **2** | 1 段（域摘要） |
| Reviewer | ≤ **5** | 同上 | — | 1 段 |
| Librarian | ≤ **3** | ≤ **2** | ≤ **2** | 1 段（短） |

### 8.4 Interviewer 注入形态（推荐结构）

分层混合，**不以** 纯摘要或纯逐条一种为主：

```text
1. [Derived 域摘要]     1–3 句：当前域概况与关注面
2. [Active Beliefs]     ≤5 条：point + probe hint + planned_layer
3. [Due 优先级]         已 due 条目在 belief 列表内打标，不额外占条数
4. [Latest evidence]    每条仅附最近一次 turn 要点
5. [Commitment]         用户 override / improved 约束
6. [Procedure]          ≤2 条交互偏好
```

### 8.5 注入优先级

```text
user Commitment / override
  > due 标记的 active Belief
  > 其余 active Belief
  > derived 摘要
  > Procedure
```

（candidate 不在此列表 — 默认不注入。）

---

## 9. Domain 与 Scope（Belief）

- **scope**：`universal` | `domain`（legacy 规则见 `INTERVIEW_PROFILE_DESIGN.md`）。
- **问答无 session folder** 时：
  - 有 RAG **citation** → 从 **note path 反推** `domain_anchor`；
  - 无 citation → 先入 **universal** **candidate**。
- **Procedure**：默认 **universal**；走 candidate → active，并支持 **用户手动设置**。

---

## 10. 可见性与 UI 原则

产品层三块（写进 Charter，实现后续）：

| 区块 | 用户权限 | 内容 |
|------|----------|------|
| **Evidence** | 只读 | 会话 transcript、turn review 链接 |
| **Canonical** | 可编辑 | Belief / Procedure；确认 candidate、archived 恢复 |
| **Derived** | 可编辑 | 域地图、摘要；编辑触发 Commitment（§4.2） |

用户「删除」Belief → **archived**，**保留** evidence 链；不硬删 canonical 行（审计可追）。

---

## 11. 迁移与 Legacy

| 项 | 规则 |
|----|------|
| 现有 `interview_profile.weak_points` | 一律视为 **`lifecycle=active`**；迁移补 `source_kind=interview` |
| `improved` / 用户否决 | 纳入 **Commitment** 语义；archived 不注入 |
| Derived 首次生成 | 从现有 profile **一次性投影**；之后按 §4.1 维护 |
| `dormant` 四态、confusion 独立 canonical 文件、append-only event log | 后续迭代，非 MVP |

---

## 12. 后续设计文档

| 文档 | 状态 | 内容 |
|------|------|------|
| [`MEMORY_WRITE_CONSOLIDATION.md`](MEMORY_WRITE_CONSOLIDATION.md) | v0.1 | Observation vs Commit、三流触发、合并、Commitment |
| [`MEMORY_SCHEMA_MIGRATION.md`](MEMORY_SCHEMA_MIGRATION.md) | v0.1 | v4 schema、`data/profile/`、v3→v4 迁移、双轨期 |
| **Injection 细则** | [`MEMORY_INJECTION.md`](MEMORY_INJECTION.md) | 各 skill prompt 片段模板（Charter §8 已含矩阵与预算） |
| **Eval** | metrics slice shipped | `GET /api/memory/metrics` 暴露 inject 预算预览、candidate/archived 积压、derived health；误合并率后续补 |

---

## 13. 验证（实现阶段）

Charter 本身无自动化测试。实现落地后建议：

```powershell
# memory 模块单元测试（待建）
# python -m pytest tests/test_memory_*.py -q

# 回归：现有 profile / review 决策
python -m pytest tests/test_review_practice.py tests/test_conversation_schema.py -q
```

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-24 | v0.1：五步顶层设计讨论收口，首版 Charter |
| 2026-06-24 | v0.1：Write/Consolidation + Schema/Migration 细则 |
| 2026-06-24 | v0.2：Belief kind×category vs Procedure；low 预留；Schedule 内嵌 |
