from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
}


@dataclass(frozen=True)
class ConvertRecord:
    source_path: str
    output_path: str
    status: str
    source_type: str
    source_collection: str
    converter: str | None = None
    source_md5: str | None = None
    markdown_chars: int = 0
    heading_count: int = 0
    error: str | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PDF/Office/HTML documents to Markdown with Docling."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing PDF/DOCX/PPTX/XLSX/HTML files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for generated Markdown files.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Source collection name written to frontmatter. Defaults to input directory name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional conversion report JSON path. Defaults to <output>/_conversion_report.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be converted without writing Markdown files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most N input files. Useful for checking conversion quality first.",
    )
    parser.add_argument(
        "--converter",
        choices=["auto", "docling", "pymupdf4llm", "pymupdf"],
        default="auto",
        help=(
            "Document converter to use. auto tries Docling first, then pymupdf4llm, "
            "then pymupdf for PDF files."
        ),
    )
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    collection = args.collection or input_root.name
    report_path = Path(args.report).resolve() if args.report else output_root / "_conversion_report.json"

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_root}")

    files = collect_input_files(input_root)
    if args.limit is not None:
        files = files[: max(args.limit, 0)]
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Collection: {collection}")
    print(f"Converter: {args.converter}")
    print(f"Input files: {len(files)}")

    docling_converter = None
    if not args.dry_run and args.converter in {"auto", "docling"}:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed yet. Install it with: uv add docling"
            ) from exc
        docling_converter = DocumentConverter()
    records: list[ConvertRecord] = []

    for source_path in files:
        relative_path = source_path.relative_to(input_root)
        output_path = output_root / relative_path.with_suffix(".md")
        source_type = infer_source_type(source_path)

        if args.dry_run:
            print(f"DRY RUN: {source_path} -> {output_path}")
            records.append(
                ConvertRecord(
                    source_path=source_path.as_posix(),
                    output_path=output_path.as_posix(),
                    status="dry_run",
                    source_type=source_type,
                    source_collection=collection,
                )
            )
            continue

        if output_path.exists() and not args.overwrite:
            print(f"SKIP existing: {output_path}")
            records.append(
                ConvertRecord(
                    source_path=source_path.as_posix(),
                    output_path=output_path.as_posix(),
                    status="skipped_existing",
                    source_type=source_type,
                    source_collection=collection,
                )
            )
            continue

        try:
            print(f"Converting: {source_path}")
            source_md5 = file_md5(source_path)
            conversion_result = convert_to_markdown(
                docling_converter=docling_converter,
                source_path=source_path,
                converter_name=args.converter,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                build_markdown_document(
                    markdown=conversion_result.markdown,
                    source_path=source_path,
                    source_type=source_type,
                    source_collection=collection,
                    source_md5=source_md5,
                    converter=conversion_result.converter,
                ),
                encoding="utf-8",
            )
            record = ConvertRecord(
                source_path=source_path.as_posix(),
                output_path=output_path.as_posix(),
                status="converted",
                source_type=source_type,
                source_collection=collection,
                converter=conversion_result.converter,
                source_md5=source_md5,
                markdown_chars=len(conversion_result.markdown),
                heading_count=count_markdown_headings(conversion_result.markdown),
            )
            records.append(record)
            print(
                f"  -> {output_path} "
                f"({record.markdown_chars} chars, {record.heading_count} headings)"
            )
        except Exception as exc:
            print(f"FAIL: {source_path}")
            print(f"  {type(exc).__name__}: {exc}")
            records.append(
                ConvertRecord(
                    source_path=source_path.as_posix(),
                    output_path=output_path.as_posix(),
                    status="failed",
                    source_type=source_type,
                    source_collection=collection,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    write_report(report_path, input_root, output_root, collection, records)
    print_summary(records)
    print(f"Report: {report_path}")

    return 0 if not any(record.status == "failed" for record in records) else 1


def collect_input_files(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


@dataclass(frozen=True)
class MarkdownConversion:
    markdown: str
    converter: str


def convert_to_markdown(
    docling_converter: Any,
    source_path: Path,
    converter_name: str,
) -> MarkdownConversion:
    if converter_name == "docling":
        return convert_with_docling(docling_converter, source_path)

    if converter_name == "pymupdf4llm":
        conversion = convert_pdf_with_pymupdf4llm(source_path)
        if conversion.markdown:
            return conversion
        raise RuntimeError("pymupdf4llm failed or returned empty Markdown.")

    if converter_name == "pymupdf":
        conversion = convert_pdf_with_pymupdf(source_path)
        if conversion.markdown:
            return conversion
        raise RuntimeError("pymupdf failed or returned empty Markdown.")

    if converter_name != "auto":
        raise ValueError(f"Unsupported converter: {converter_name}")

    try:
        return convert_with_docling(docling_converter, source_path)
    except Exception as exc:
        print(f"  Docling failed: {type(exc).__name__}: {exc}")

    if source_path.suffix.lower() != ".pdf":
        raise RuntimeError("Docling failed and fallback is only available for PDF files.")

    fallback = convert_pdf_with_pymupdf4llm(source_path)
    if fallback.markdown:
        return fallback

    fallback = convert_pdf_with_pymupdf(source_path)
    if fallback.markdown:
        return fallback

    raise RuntimeError("All converters failed or returned empty Markdown.")


def convert_with_docling(docling_converter: Any, source_path: Path) -> MarkdownConversion:
    if docling_converter is None:
        raise RuntimeError("Docling converter was not initialized.")

    result = docling_converter.convert(source_path)
    markdown = result.document.export_to_markdown().strip()
    if markdown:
        return MarkdownConversion(markdown=markdown, converter="docling")
    if source_path.suffix.lower() == ".pdf":
        print("  Docling returned empty Markdown; falling back.")
    raise RuntimeError("Docling returned empty Markdown.")


def convert_pdf_with_pymupdf4llm(source_path: Path) -> MarkdownConversion:
    if source_path.suffix.lower() != ".pdf":
        raise RuntimeError("pymupdf4llm only supports PDF input in this script.")

    try:
        import pymupdf4llm  # type: ignore
    except ImportError:
        print("  pymupdf4llm not installed; skipping fallback.")
        return MarkdownConversion(markdown="", converter="pymupdf4llm")

    try:
        markdown = pymupdf4llm.to_markdown(str(source_path)).strip()
        if markdown:
            print("  Fallback succeeded: pymupdf4llm")
        return MarkdownConversion(markdown=markdown, converter="pymupdf4llm")
    except Exception as exc:
        print(f"  pymupdf4llm failed: {type(exc).__name__}: {exc}")
        return MarkdownConversion(markdown="", converter="pymupdf4llm")


def convert_pdf_with_pymupdf(source_path: Path) -> MarkdownConversion:
    if source_path.suffix.lower() != ".pdf":
        raise RuntimeError("pymupdf only supports PDF input in this script.")

    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            print("  pymupdf not installed; skipping fallback.")
            return MarkdownConversion(markdown="", converter="pymupdf")

    try:
        lines: list[str] = []
        with pymupdf.open(str(source_path)) as document:
            for page_index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                lines.append(f"## Page {page_index}\n\n{text}")
        markdown = "\n\n".join(lines).strip()
        if markdown:
            print("  Fallback succeeded: pymupdf")
        return MarkdownConversion(markdown=markdown, converter="pymupdf")
    except Exception as exc:
        print(f"  pymupdf failed: {type(exc).__name__}: {exc}")
        return MarkdownConversion(markdown="", converter="pymupdf")


def build_markdown_document(
    markdown: str,
    source_path: Path,
    source_type: str,
    source_collection: str,
    source_md5: str,
    converter: str,
) -> str:
    converted_at = datetime.now().isoformat(timespec="seconds")
    title = source_path.stem

    frontmatter = f"""---
source_type: {yaml_scalar(source_type)}
source_collection: {yaml_scalar(source_collection)}
converter: {yaml_scalar(converter)}
source_path: {yaml_scalar(source_path.as_posix())}
source_md5: {yaml_scalar(source_md5)}
converted_at: {yaml_scalar(converted_at)}
---

# {title}

"""
    body = markdown.strip()
    if body.startswith(f"# {title}"):
        return frontmatter + body + "\n"
    return frontmatter + body + "\n"


def write_report(
    report_path: Path,
    input_root: Path,
    output_root: Path,
    collection: str,
    records: list[ConvertRecord],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_root": input_root.as_posix(),
        "output_root": output_root.as_posix(),
        "collection": collection,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summarize(records),
        "records": [asdict(record) for record in records],
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize(records: list[ConvertRecord]) -> dict[str, int]:
    summary: dict[str, int] = {
        "total": len(records),
        "converted": 0,
        "skipped_existing": 0,
        "dry_run": 0,
        "failed": 0,
    }
    for record in records:
        summary[record.status] = summary.get(record.status, 0) + 1
    return summary


def print_summary(records: list[ConvertRecord]) -> None:
    summary = summarize(records)
    print("Summary:")
    print(f"  total: {summary.get('total', 0)}")
    print(f"  converted: {summary.get('converted', 0)}")
    print(f"  skipped_existing: {summary.get('skipped_existing', 0)}")
    print(f"  dry_run: {summary.get('dry_run', 0)}")
    print(f"  failed: {summary.get('failed', 0)}")


def infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_markdown_headings(markdown: str) -> int:
    return sum(1 for line in markdown.splitlines() if line.lstrip().startswith("#"))


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "/").replace('"', '\\"')
    return f'"{escaped}"'


if __name__ == "__main__":
    raise SystemExit(main())
