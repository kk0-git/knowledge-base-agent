# Obsidian Workflow

这份文档记录 Obsidian 侧的使用配置。设计决策写在 `knowledge_agent/DECISIONS.md`，可执行命令写在 `commands.md`。

## 日常路径

```text
在 Obsidian 写笔记
  -> 需要更新某个主题时，运行“更新主题 Wiki”
  -> 查看 wiki_full/_Wiki Report.md
  -> 打开 wiki_full/... 阅读生成结果
```

如果需要更新搜索/对话用的向量索引，再单独运行“同步 RAG 索引”。

## 工作区路径

Web 页面提供工作区设置入口：

```text
http://127.0.0.1:8000/settings
```

这里可以填写：

```text
笔记库路径
Wiki 输出目录
wiki_state.json
workspace_state.json
RAG index / BM25 index
```

配置会保存到 `workspace-config.json`。这是本机路径配置，不提交到 Git。

Obsidian Shell Commands 仍然适合固定常用 vault。如果换 vault，先在 `/settings` 改 Web 工作区；Shell Commands 中写死的路径也需要同步调整。

## Shell Commands 插件

插件地址：

```text
obsidian://show-plugin?id=obsidian-shellcommands
```

建议配置两个命令。

## 命令 1：更新主题 Wiki

用途：

```text
同步 changed notes 的 tag state
生成指定 topic wiki
刷新 wiki_full/_Wiki Report.md
```

该命令会先同步工作区，再更新指定主题 Wiki；如果只想跳过 embedding，可额外追加 `--no-rag`。

Shell command：

```powershell
cd D:\Workspaces\Personal\agent\obsidian_vault\knowledge_agent
uv run python scripts\workspace_sync.py update-topic `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --min-notes-per-tag 2 `
  --overview-note-threshold 12 `
  --tag "{{tag}}" `
  --force
```

推荐使用 Prompt 变量：

```text
变量名: tag
提示文本: Wiki tag
示例输入: 学习/全栈
```

不建议依赖选中文本变量作为第一版入口。Shell Commands 在设置页预览时没有选区，容易出现 `Nothing is selected`；Prompt 变量更稳定。

推荐 Output 设置：

```text
stdout: Notification
stderr: Error balloon
Output handling mode: Wait until finished
Show notification when executing: Default 或 Show
```

成功时应看到类似输出：

```text
OK: synced 1 changed notes, refreshed topic wiki.
Topic wiki updated.
Report: D:\31002\Documents\MyNote\wiki_full\_Wiki Report.md
```

## 命令 2：同步 RAG 索引

用途：

```text
更新 embedding / BM25
同步 tag state
刷新 wiki_full/_Wiki Report.md
不自动合成 wiki 正文
```

Shell command：

```powershell
cd D:\Workspaces\Personal\agent\obsidian_vault\knowledge_agent
uv run python scripts\workspace_sync.py watch `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --min-notes-per-tag 2 `
  --overview-note-threshold 12 `
  --once
```

运行时会输出阶段进度，例如：

```text
RAG: sync started.
RAG: scanned 216 markdown files (0 excluded, 0 failed).
RAG: plan added=0, modified=1, deleted=0, changed=1.
RAG: embedding file 1/1 chunks=6.
RAG: embedding batch 1/1 texts=6.
RAG: rebuilding BM25 for 2819 chunks.
RAG: done embedded_chunks=6, total_chunks=2819.
Wiki tags: sync started.
Wiki tags: done tagged=1, skipped=215, failed=0.
Report: updated D:\31002\Documents\MyNote\wiki_full\_Wiki Report.md.
```

当前增量向量化是顺序执行：按 changed file 逐个处理，每个文件按 `embed_batch_size` 分批调用 embedding API。它是批量请求，不是并发请求。这样更容易断点持久化，也更不容易触发 API 限流。

## Dirty 状态含义

同步流程会维护一份统一文件状态：

```text
wiki-state/workspace_state.json
```

报告中的 Workspace 行含义：

```text
files        当前纳入同步的 markdown 文件数
embed_dirty  需要更新 RAG embedding/BM25 的文件数
tags_dirty   需要重新提取 tag 的文件数
deleted      已从 vault 删除、需要从状态中清理的文件数
```

三类 dirty 不等价：

```text
embed_dirty > 0
  搜索/对话索引还没同步到最新笔记。

tags_dirty > 0
  自动 tag 还没同步到最新笔记。

需要更新 > 0
  某些 topic wiki 的源笔记、来源集合或主题归属变了，需要重新合成 wiki 正文。
```

运行“更新主题 Wiki”会先同步 workspace，再更新指定 wiki。运行“同步 RAG 索引”只更新检索索引、tag state 和 report，不会自动改写 wiki 正文。

## 报告文件

状态面板位置：

```text
D:\31002\Documents\MyNote\wiki_full\_Wiki Report.md
```

它是 Obsidian 内的行动面板：

```text
需要更新
尚未合成
已生成
```

更新主题 Wiki 或同步 RAG 索引后都会刷新这个文件。

## 常见问题

`{{selected_text}}` 没有替换：

Shell Commands 当前没有解析这个变量。使用 Prompt 变量 `{{tag}}`。

`{{selection}}: Nothing is selected`：

这是设置页预览或运行时没有实际选区。使用 Prompt 变量更稳定。

中文输出乱码：

这是 PowerShell 5 / Shell Commands 输出编码问题。当前 CLI 成功输出已改成 ASCII，文件内容仍按 UTF-8 写入，不影响 wiki 文件。

只更新 Wiki 是否会更新 embedding：

默认会同步 embedding、tag state、指定 wiki 和 report。如果只想跳过 embedding，可在命令末尾追加 `--no-rag`。

什么时候运行同步 RAG 索引：

当你希望搜索页、聊天页、RAG eval 使用最新笔记内容时运行。只想更新某个 wiki 时不需要运行。



