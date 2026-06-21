# Librarian 结构化读取工具设计

> 将 vault 读取从「前 N 字符截断」升级为「先结构、再按 section 精读」。

## 工具层

| 工具 | 用途 |
|------|------|
| `inspect_note` | 返回 path/title/char_count/sections（含 preview、line range、section_id） |
| `read_note` | Smart entry：短笔记 `mode=full`；长笔记 `mode=outline`；带 `heading` / `heading_path` / `section_id` 时 `mode=section` |

## 共享解析器

- `src/services/markdown/sections.py`
- RAG chunker 与 agent tools 共用 `split_markdown_sections` / `parse_markdown_sections`
- Section 定位失败返回 error，不再 silent fallback 到全文

## Agent 决策链（SKILL）

```
search/grep 命中
  → inspect_note 或 read_note(path) 获取 outline
  → read_note(section) 精读 1-3 个相关章节
  → 回答
```

## 阈值

- `SHORT_NOTE_CHAR_THRESHOLD = 2500`：低于此值 `read_note` 直接返回全文

## 延后

- `read_note_summary`（缓存摘要）
- organize ReviewPacket 改用 section excerpt
