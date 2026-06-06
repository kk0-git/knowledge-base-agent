from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_URL = "https://mineru.net/api/v1/agent"
MAX_LIGHTWEIGHT_FILE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class MinerUAgentResult:
    source_path: str
    output_path: str
    task_id: str
    markdown_url: str
    language: str
    page_range: str | None
    is_ocr: bool
    enable_table: bool
    enable_formula: bool
    markdown_chars: int
    heading_count: int
    elapsed_seconds: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use MinerU Agent Lightweight Extract API to convert one local file to Markdown."
    )
    parser.add_argument("--file", required=True, help="Local PDF/image/Office file to upload.")
    parser.add_argument("--out", required=True, help="Output Markdown path.")
    parser.add_argument(
        "--collection",
        default="textbooks",
        help="Source collection written to Markdown frontmatter.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. Defaults to <out>.mineru-agent.json.",
    )
    parser.add_argument(
        "--language",
        default="ch",
        help="OCR language pack. Use 'ch' for Chinese+English, 'en' for English.",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help="Optional PDF page range, e.g. '1-10'. Lightweight API supports at most 20 pages.",
    )
    parser.add_argument(
        "--ocr",
        dest="is_ocr",
        action="store_true",
        default=True,
        help="Enable OCR. Default: enabled, because this script is for OCR comparison.",
    )
    parser.add_argument(
        "--no-ocr",
        dest="is_ocr",
        action="store_false",
        help="Disable OCR and use embedded text when available.",
    )
    parser.add_argument(
        "--enable-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable table recognition. Default: true.",
    )
    parser.add_argument(
        "--enable-formula",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable formula recognition. Default: true.",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Polling timeout in seconds.")
    parser.add_argument(
        "--transport",
        choices=["sdk", "manual"],
        default="sdk",
        help="Use mineru-open-sdk or this script's manual HTTP flow.",
    )
    parser.add_argument(
        "--upload-no-proxy",
        action="store_true",
        help="Bypass HTTP(S)_PROXY for the pre-signed upload and Markdown download URLs.",
    )
    args = parser.parse_args()

    source_path = Path(args.file).resolve()
    output_path = Path(args.out).resolve()
    report_path = Path(args.report).resolve() if args.report else output_path.with_suffix(
        output_path.suffix + ".mineru-agent.json"
    )

    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Input path is not a file: {source_path}")

    size_bytes = source_path.stat().st_size
    if size_bytes > MAX_LIGHTWEIGHT_FILE_BYTES:
        raise ValueError(
            f"File is larger than MinerU Agent lightweight limit: "
            f"{size_bytes} bytes > {MAX_LIGHTWEIGHT_FILE_BYTES} bytes"
        )

    started_at = time.time()
    print(f"Input: {source_path}")
    print(f"Output: {output_path}")
    print(f"OCR: {args.is_ocr}")
    print(f"Language: {args.language}")
    print(f"Page range: {args.page_range or '(all)'}")

    if args.transport == "manual":
        task_id, markdown_url, markdown = extract_with_manual_http(
            source_path=source_path,
            language=args.language,
            page_range=args.page_range,
            is_ocr=args.is_ocr,
            enable_table=args.enable_table,
            enable_formula=args.enable_formula,
            timeout=args.timeout,
            no_proxy_upload=args.upload_no_proxy,
        )
    else:
        extract_result = extract_with_sdk(
            source_path=source_path,
            language=args.language,
            page_range=args.page_range,
            is_ocr=args.is_ocr,
            enable_table=args.enable_table,
            enable_formula=args.enable_formula,
            timeout=args.timeout,
        )
        if extract_result.state != "done":
            raise RuntimeError(
                f"MinerU task did not finish successfully: "
                f"state={extract_result.state}, error={extract_result.error}, err_code={extract_result.err_code}"
            )
        task_id = extract_result.task_id
        markdown_url = ""
        markdown = (extract_result.markdown or "").strip()
        if not markdown:
            raise RuntimeError(f"MinerU task done but Markdown is empty. task_id={task_id}")

    print(f"Task done: {task_id}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_markdown_document(
            markdown=markdown,
            source_path=source_path,
            source_collection=args.collection,
            task_id=task_id,
            markdown_url=markdown_url,
            language=args.language,
            page_range=args.page_range,
            is_ocr=args.is_ocr,
            enable_table=args.enable_table,
            enable_formula=args.enable_formula,
        ),
        encoding="utf-8",
    )

    elapsed = int(time.time() - started_at)
    result = MinerUAgentResult(
        source_path=source_path.as_posix(),
        output_path=output_path.as_posix(),
        task_id=task_id,
        markdown_url=markdown_url,
        language=args.language,
        page_range=args.page_range,
        is_ocr=args.is_ocr,
        enable_table=args.enable_table,
        enable_formula=args.enable_formula,
        markdown_chars=len(markdown),
        heading_count=count_markdown_headings(markdown),
        elapsed_seconds=elapsed,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Written: {output_path}")
    print(f"Report: {report_path}")
    print(f"Markdown chars: {result.markdown_chars}")
    print(f"Headings: {result.heading_count}")
    print(f"Elapsed: {elapsed}s")
    return 0


def create_upload_task(
    file_name: str,
    language: str,
    page_range: str | None,
    is_ocr: bool,
    enable_table: bool,
    enable_formula: bool,
) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "file_name": file_name,
        "language": language,
        "enable_table": enable_table,
        "is_ocr": is_ocr,
        "enable_formula": enable_formula,
    }
    if page_range:
        payload["page_range"] = page_range

    data = post_json(f"{BASE_URL}/parse/file", payload)
    task_id = require_nested(data, "data", "task_id")
    file_url = require_nested(data, "data", "file_url")
    return str(task_id), str(file_url)


def extract_with_manual_http(
    source_path: Path,
    language: str,
    page_range: str | None,
    is_ocr: bool,
    enable_table: bool,
    enable_formula: bool,
    timeout: int,
    no_proxy_upload: bool,
) -> tuple[str, str, str]:
    task_id, file_url = create_upload_task(
        file_name=source_path.name,
        language=language,
        page_range=page_range,
        is_ocr=is_ocr,
        enable_table=enable_table,
        enable_formula=enable_formula,
    )
    print(f"Task created: {task_id}")
    print(f"Upload URL host: {urllib.parse.urlparse(file_url).netloc}")

    upload_file(file_url, source_path, no_proxy=no_proxy_upload)
    print("Upload finished. Polling result...")

    markdown_url = poll_markdown_url(task_id=task_id, timeout=timeout, interval=5)
    print(f"Markdown URL host: {urllib.parse.urlparse(markdown_url).netloc}")
    markdown = download_text(markdown_url, no_proxy=no_proxy_upload)
    return task_id, markdown_url, markdown


def extract_with_sdk(
    source_path: Path,
    language: str,
    page_range: str | None,
    is_ocr: bool,
    enable_table: bool,
    enable_formula: bool,
    timeout: int,
) -> Any:
    try:
        from mineru import MinerU
    except ImportError as exc:
        raise RuntimeError("mineru-open-sdk is not installed. Install it with: uv add mineru-open-sdk") from exc

    with MinerU() as client:
        return client.flash_extract(
            str(source_path),
            language=language,
            page_range=page_range,
            is_ocr=is_ocr,
            enable_table=enable_table,
            enable_formula=enable_formula,
            timeout=timeout,
        )


def upload_file(file_url: str, source_path: Path, no_proxy: bool) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for MinerU upload.") from exc

    data = source_path.read_bytes()
    try:
        with httpx.Client(trust_env=not no_proxy, timeout=httpx.Timeout(30.0, write=120.0)) as client:
            response = client.put(file_url, content=data)
            response.raise_for_status()
    except Exception as exc:
        response = getattr(exc, "response", None)
        if response is not None:
            raise RuntimeError(f"Upload failed: HTTP {response.status_code}: {response.text}") from exc
        raise RuntimeError(f"Upload failed: {type(exc).__name__}: {exc}") from exc


def poll_markdown_url(task_id: str, timeout: int, interval: int) -> str:
    started_at = time.time()
    while time.time() - started_at < timeout:
        data = get_json(f"{BASE_URL}/parse/{task_id}")
        payload = data.get("data") or {}
        state = payload.get("state")
        elapsed = int(time.time() - started_at)

        if state == "done":
            markdown_url = payload.get("markdown_url")
            if not markdown_url:
                raise RuntimeError(f"MinerU task done but markdown_url is missing: {data}")
            return str(markdown_url)

        if state == "failed":
            err_code = payload.get("err_code")
            err_msg = payload.get("err_msg", "unknown error")
            raise RuntimeError(f"MinerU task failed: {err_code} {err_msg}")

        print(f"[{elapsed}s] state={state}")
        time.sleep(interval)

    raise TimeoutError(f"Polling timed out after {timeout}s. task_id={task_id}")


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return request_json(request)


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    return request_json(request)


def request_json(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    data = json.loads(text)
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU API error: {data}")
    return data


def download_text(url: str, no_proxy: bool) -> str:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for MinerU download.") from exc

    try:
        with httpx.Client(trust_env=not no_proxy, timeout=httpx.Timeout(30.0, read=300.0)) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.text.strip()
    except Exception as exc:
        response = getattr(exc, "response", None)
        if response is not None:
            raise RuntimeError(
                f"Markdown download failed: HTTP {response.status_code}: {response.text}"
            ) from exc
        raise RuntimeError(f"Markdown download failed: {type(exc).__name__}: {exc}") from exc


def build_markdown_document(
    markdown: str,
    source_path: Path,
    source_collection: str,
    task_id: str,
    markdown_url: str,
    language: str,
    page_range: str | None,
    is_ocr: bool,
    enable_table: bool,
    enable_formula: bool,
) -> str:
    converted_at = datetime.now().isoformat(timespec="seconds")
    frontmatter = f"""---
source_type: "pdf"
source_collection: "{yaml_escape(source_collection)}"
converter: "mineru-agent"
source_path: "{yaml_escape(source_path.as_posix())}"
mineru_task_id: "{yaml_escape(task_id)}"
mineru_markdown_url: "{yaml_escape(markdown_url)}"
language: "{yaml_escape(language)}"
page_range: "{yaml_escape(page_range or '')}"
is_ocr: {str(is_ocr).lower()}
enable_table: {str(enable_table).lower()}
enable_formula: {str(enable_formula).lower()}
converted_at: "{converted_at}"
---

"""
    return frontmatter + markdown.strip() + "\n"


def require_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"Missing response field {'.'.join(keys)}: {data}")
        current = current[key]
    return current


def count_markdown_headings(markdown: str) -> int:
    return sum(1 for line in markdown.splitlines() if line.lstrip().startswith("#"))


def yaml_escape(value: str) -> str:
    return value.replace("\\", "/").replace('"', '\\"')


if __name__ == "__main__":
    raise SystemExit(main())
