# Librarian Read Tools Design

> 单一 `read_note` 工具：永远返回正文，固定字符预算，section_id 主续读，offset 节内翻页。

## 工具

| 工具 | 用途 |
|------|------|
| `read_note` | 读取笔记正文；截断时附带 `sections` 导航 map |

已删除 `inspect_note`：结构导航合并进 `read_note` 的截断响应。

## 契约

### 参数

| 参数 | 说明 |
|------|------|
| `path` | 必填 |
| `section_id` | 主定位：跳转到指定 section |
| `heading` / `heading_path` | 便捷定位（解析为 section） |
| `offset` | 字符偏移，相对当前阅读窗口 |
| `max_chars` | 默认 4000 |
| `reason` | trace / 前端展示 |

### 响应

- **永远有** `content`
- `truncated=true` 时附带 `sections`、`next_offset`、`hint`
- 无 `mode` 分支（不再 full / outline / section）

### 阅读窗口

1. 有 `section_id` / `heading` / `heading_path` → section 内容
2. 否则 → 整篇笔记

### 分页

在窗口内按字符 `offset` + `max_chars` 切片。截断时 `next_offset` 供续读。

## Agent 决策链

```
read_note(path) → 正文（可能截断）+ sections（若截断）
  → section_id 跳节 / offset 续读
```

## 设计原则

- 工具行为固定、可预测（对齐 Cursor / Claude Code）
- 不在工具层按字数自动切换 outline 模式
- `scope_index`（note 级）+ `sections`（note 内）两级导航

## 延后

- `read_note_summary`（缓存摘要）
- organize ReviewPacket 改用 section excerpt
