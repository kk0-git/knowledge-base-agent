# TROUBLESHOOTING.md

历史故障排查记录。从现象 → 原因 → 修复，便于同类问题快速定位。

---

## 复习页：切页/刷新后卡片显示「正在生成复习题…」

**日期**：2026-06-24  
**范围**：`/review` 逐条对照（Phase C 服务端 run + sessionStorage workspace）

### 现象

- 第一次进入逐条对照，已生成题目正常显示。
- 切换到其他页面再回 `/review`，或在卡片页 F5 刷新后，当前卡片长时间显示「正在生成复习题…」（或「第一题准备好后会自动显示」），数秒后才恢复。

### 原因

1. **`mergeCards` 覆盖本地 prompt**  
   `initReviewPage` / `resyncCardReviewRun` 会用 `GET /api/review/plan/{id}` 的结果合并 cards。原逻辑 `{...local, ...server}` 会用服务端 `status: "pending"` 且不含 `review_prompt` 的条目盖掉 sessionStorage 里已有题目的卡片。  
   `isCardViewable` 仍因 `status === "pending"` 为 true，但 `review_prompt.prompt` 为空，UI fallback 为「正在生成复习题…」，直到 poll 拉回服务端 ready 数据。

2. **`__REVIEW_PROMPT_VERSION__` 未替换**  
   `REVIEW_HTML` 中 `CURRENT_REVIEW_PROMPT_VERSION` 占位符未在 `render_web_page` 替换为后端 `REVIEW_PROMPT_VERSION`（`review-prompt-v3`）。  
   恢复 snapshot 时 `inferCardPromptVersion(cards)` 得到 `review-prompt-v3`，与字面量 `__REVIEW_PROMPT_VERSION__` 不等 → 误判 `staleCardPrompts` → 触发 `needsReplan` 整轮重新生成。

### 修复

| 位置 | 改动 |
|------|------|
| `render_web_page` | `content.replace("__REVIEW_PROMPT_VERSION__", REVIEW_PROMPT_VERSION)` |
| `mergeCards` | 合并时保留 local/server 任一侧已有 `review_prompt`；有 prompt 时 status 视为 `ready` |
| `applyReviewSnapshot` | `normalizePromptVersion` 忽略占位符，从 cards 推断真实版本，避免误 replan |
| `resyncCardReviewRun` | 本地当前卡已有 prompt 时不强制重绘 |
| `ReviewRunStore.snapshot_payload` | 返回 `prompt_version` |

### 后续优化（2026-06-24）

- **Workspace 版本迁移**：`KnowledgeAgentWorkspace.migrate()` / `migrateReviewSlice` / `migrateChatSlice`；review snapshot `migrateReviewSnapshot`（v0→v1 补 `version` / `promptVersion`）。
- **总览进度**：总览 tab 与 workspace 摘要显示「已完成 N / 共 M」。
- **自动化**：`tests/test_web_workspace.py` 断言 `KnowledgeAgentWorkspace` 在 `<head>`、早于页面 inline script（防 script 顺序回归）。

### 验证

1. 逐条对照做到第 2 题（题目已显示）→ 去其他页 → 回 `/review` → 题目立即恢复，无长时间「正在生成」。
2. 同上状态 F5 → 题目与进度恢复。
3. 控制台 / sessionStorage：`knowledge_agent.workspace.review.v1` 中 cards 仍含 `review_prompt.prompt`。

### 相关文件

- `src/web/app.py` — `mergeCards`, `applyReviewSnapshot`, `resyncCardReviewRun`, `render_web_page`
- `src/services/workflows/review_runs.py` — snapshot `prompt_version`
- `src/services/workflows/review_practice.py` — `REVIEW_PROMPT_VERSION`
