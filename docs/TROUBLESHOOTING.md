# Troubleshooting

## Web pages show mojibake or render blank

Date observed: 2026-06-18

Affected pages:

- `/chat`
- `/topics`
- `/organize`
- `/search`
- `/wiki`
- `/settings`

Symptoms:

- Browser title displayed mojibake instead of Chinese labels such as "对话", "知识主题", "整理", "检索调试", "维护", "设置".
- Some pages appeared blank or partially blank.
- Several restored pages returned HTTP 200 but the frontend did not work.
- `<title>` tags were corrupted so title extraction failed or returned unreadable text.
- Embedded JavaScript failed `node --check` with errors such as `Invalid or unexpected token`, `missing ) after argument list`, or invalid regular expressions.

Root cause:

- Chinese text inside large Python raw HTML templates in `src/web/app.py` was rewritten through an unsafe encoding path.
- The corrupted text became mojibake inside HTML and JavaScript string literals.
- Some corrupted sequences landed inside JavaScript regular expressions and quoted strings, making the scripts syntactically invalid.
- Because these pages are mostly driven by inline JavaScript, a script parse failure prevented page initialization and made the page look blank even though the backend returned HTML.
- Partial regex-based edits made the problem worse in one pass: static labels were changed, but damaged JavaScript templates and long corrupted strings remained.

Why HTTP 200 was misleading:

- FastAPI served the HTML successfully.
- The failure happened in the browser while parsing/executing inline JavaScript.
- Always validate both the HTTP response and the embedded script syntax when a page is blank.

What fixed it:

- Restored clean original templates for:
  - `WIKI_HTML`
  - `TOPICS_HTML`
  - `AUDIT_HTML`
  - `SETTINGS_HTML`
  - `INDEX_HTML`
- Rebuilt `CHAT_HTML` as a clean template after the original had diverged due to interview-session changes.
- Used HTML entities for Chinese UI labels in newly edited HTML to avoid another unsafe text-encoding write.
- Avoided Chinese literals in JavaScript strings unless the write path is known to preserve UTF-8.
- Restarted the local server after edits so the browser loaded the new templates.

Validation checklist:

```powershell
uv run python -m py_compile src/web/app.py src/services/workflows/interview.py
```

Extract and check inline page scripts:

```powershell
@'
from pathlib import Path
import re

text = Path("src/web/app.py").read_text(encoding="utf-8")
for name in ["TOPICS_HTML", "AUDIT_HTML", "SETTINGS_HTML", "INDEX_HTML", "WIKI_HTML", "CHAT_HTML"]:
    start = text.index(f'{name} = r"""')
    end = text.index('\n"""', start)
    html = text[start:end]
    scripts = list(re.finditer(r"<script>(.*?)</script>", html, re.S))
    print(name, "scripts", len(scripts), "title_match", bool(re.search(r"<title>.*?</title>", html, re.S)))
    for i, match in enumerate(scripts):
        Path(f".tmp_{name}_{i}.js").write_text(match.group(1), encoding="utf-8")
'@ | python -

Get-ChildItem .tmp_*_*.js | ForEach-Object { node --check $_.FullName }
Remove-Item .tmp_*_*.js -Force
```

Check page responses:

```powershell
$pages=@('/chat','/topics','/organize','/search','/wiki','/settings')
foreach($p in $pages){
  $r=Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8003$p" -TimeoutSec 20
  $title=([regex]::Match($r.Content,'<title>(.*?)</title>').Groups[1].Value)
  $h1=([regex]::Match($r.Content,'<h1>(.*?)</h1>').Groups[1].Value)
  [pscustomobject]@{Page=$p;Status=$r.StatusCode;Length=$r.Content.Length;Title=$title;H1=$h1}
}
```

Restart local web service:

```powershell
$connections = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq 8003 }
$connections | ForEach-Object { try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction Stop } catch {} }

Start-Process -FilePath "uv" -ArgumentList @(
  "run","python","scripts\web_search.py",
  "--index","./rag-index/mixed-siliconflow-bge-m3.json",
  "--bm25-index","./rag-index/mixed-siliconflow-bge-m3.bm25.json",
  "--vault","D:\31002\Documents\MyNote",
  "--wiki-state","./wiki-state/wiki_state.full-test.json",
  "--wiki-dir","D:\31002\Documents\MyNote\wiki_full",
  "--embedding-provider","openai_compatible",
  "--model","BAAI/bge-m3",
  "--host","127.0.0.1",
  "--port","8003"
) -WorkingDirectory "d:\Workspaces\Personal\agent\obsidian_vault\knowledge_agent" -WindowStyle Hidden
```

Prevention rules:

- Do not patch large HTML templates with broad regex replacements unless the exact template boundaries and replacements are verified.
- Prefer `apply_patch` for small deterministic edits.
- When adding Chinese UI text into Python raw HTML templates, prefer HTML entities in HTML and Unicode escapes or ASCII text in JavaScript.
- After editing inline scripts, always run `node --check` on extracted scripts.
- If a page returns 200 but is blank, inspect browser console or run script syntax checks before debugging backend routes.
- If many static labels become mojibake at once, treat it as a template encoding corruption issue, not a UI state issue.

