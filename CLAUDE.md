# CLAUDE.md

Knowledge Base Agent — 个人 Obsidian 笔记的 RAG + Agent 问答系统。
Python + FastAPI + FAISS + BGE-M3 + BM25 + RRF。

## 每轮对话前

1. 读 docs/ITERATION.md — 当前迭代的任务和方案
2. 读 DECISIONS.md — 历史决策，了解为什么这么做

## 每轮对话后

- 功能完成 → 更新 ITERATION.md 标记状态
- 有设计取舍 → DECISIONS.md 追加记录
- 有新命令 → commands.md 追加

## 调试时

- docs/TROUBLESOLVING.md — 历史故障排查记录

## 工作方式

- 遇到问题或讨论方案时：先排查原因 / 提出方案 / 讨论权衡，**确认后再改代码**
- 不要看到问题就直接修，先讲清楚"为什么"和"怎么修"

## 约束

- reference/ 是参考项目，非本项目代码
- 不做移动端适配
- 不自动修改用户原笔记
