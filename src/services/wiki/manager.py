from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from knowledge_base_agent.llm.client import LLMClient
from services.rag.incremental import calculate_content_hash
from services.wiki.schema import NoteTagRecord, WikiState, WikiTagRecord
from services.wiki.state_store import WikiStateStore
from services.wiki.synthesizer import (
    RelatedWikiPage,
    WikiSynthesisInput,
    WikiSynthesizer,
    render_wiki_markdown,
    slug_from_tag,
    wiki_title_from_tag,
)
from services.wiki.tag_extractor import LLMTagExtractor, build_note_tag_input, extract_user_tags


DEFAULT_EXCLUDED_DIRS = {".git", ".obsidian", ".trash", ".Trash", "node_modules", "wiki"}


class WikiManager:
    def __init__(
        self,
        *,
        vault_root: Path,
        state_store: WikiStateStore,
        llm_client: LLMClient,
        llm_model: str,
        wiki_dir: Path | None = None,
        min_notes_per_tag: int = 3,
        min_tag_depth: int = 2,
        overview_note_threshold: int = 30,
    ) -> None:
        self.vault_root = vault_root
        self.state_store = state_store
        self.wiki_dir = wiki_dir or (vault_root / "wiki")
        self.min_notes_per_tag = min_notes_per_tag
        self.min_tag_depth = min_tag_depth
        self.overview_note_threshold = overview_note_threshold
        self.tag_extractor = LLMTagExtractor(client=llm_client, model=llm_model)
        self.synthesizer = WikiSynthesizer(client=llm_client, model=llm_model)

    def tag_changed_notes(self, *, force: bool = False, limit: int | None = None) -> dict:
        state = self.state_store.load()
        markdown_files = list(scan_markdown_files(self.vault_root))
        current_paths = {file.relative_to(self.vault_root).as_posix(): file for file in markdown_files}
        existing_tag_tree = build_existing_tag_tree(state, markdown_files, self.vault_root)

        files = dict(state.files)
        tagged = 0
        skipped = 0
        failed = 0

        for note_path, file_path in sorted(current_paths.items()):
            if limit is not None and tagged >= limit:
                break

            content_hash = calculate_content_hash(file_path)
            previous = files.get(note_path)
            if not force and previous and previous.content_hash == content_hash:
                skipped += 1
                continue

            try:
                note_input = build_note_tag_input(self.vault_root, file_path)
                extraction = self.tag_extractor.extract_tags(note_input, existing_tag_tree)
                record = NoteTagRecord(
                    note_path=note_path,
                    content_hash=content_hash,
                    user_tags=note_input.user_tags,
                    llm_tags=extraction.tags,
                    candidate_tags=extraction.candidate_tags,
                    title=note_input.title,
                    tagged_at=now_iso(),
                    status="clean",
                    error=None,
                )
                files[note_path] = record
                tagged += 1
            except Exception as exc:
                failed += 1
                files[note_path] = NoteTagRecord(
                    note_path=note_path,
                    content_hash=content_hash,
                    user_tags=extract_user_tags(file_path.read_text(encoding="utf-8", errors="replace")),
                    llm_tags=previous.llm_tags if previous else [],
                    candidate_tags=previous.candidate_tags if previous else [],
                    title=file_path.stem,
                    tagged_at=previous.tagged_at if previous else None,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )

        deleted_paths = sorted(set(files) - set(current_paths))
        for note_path in deleted_paths:
            files.pop(note_path, None)

        new_state = rebuild_tag_index(
            WikiState(files=files, tags=state.tags),
            overview_note_threshold=self.overview_note_threshold,
        )
        self.state_store.save(new_state)
        return {
            "markdown_files": len(markdown_files),
            "tagged": tagged,
            "skipped": skipped,
            "failed": failed,
            "deleted": len(deleted_paths),
            "tags": len(new_state.tags),
        }

    def synthesize_dirty_wikis(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        policy_filter: str | None = None,
    ) -> dict:
        state = rebuild_tag_index(
            self.state_store.load(),
            overview_note_threshold=self.overview_note_threshold,
        )
        tags = dict(state.tags)
        generated = 0
        skipped = 0
        failed = 0

        for tag, tag_record in sorted(
            tags.items(),
            key=lambda item: synthesize_sort_key(item[0], item[1]),
        ):
            if limit is not None and generated >= limit:
                break
            if policy_filter and tag_record.wiki_policy != policy_filter:
                skipped += 1
                continue
            if not force and not tag_record.dirty:
                skipped += 1
                continue
            if tag_record.wiki_policy == "skip":
                skipped += 1
                continue
            if not should_generate_tag(tag_record, self.min_notes_per_tag, self.min_tag_depth):
                skipped += 1
                continue

            try:
                all_source_records = [
                    state.files[path]
                    for path in tag_record.source_paths
                    if path in state.files and (self.vault_root / path).exists()
                ]
                source_records, source_limit_notice = select_source_records_for_synthesis(
                    tag=tag,
                    tag_record=tag_record,
                    source_records=all_source_records,
                )
                source_texts = {
                    record.note_path: (self.vault_root / record.note_path).read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    for record in source_records
                }
                related_wiki_pages = build_related_wiki_pages(
                    current_tag=tag,
                    current_record=tag_record,
                    all_tags=tags,
                )
                synthesis_input = WikiSynthesisInput(
                    tag=tag,
                    source_records=source_records,
                    source_texts=source_texts,
                    related_wiki_pages=related_wiki_pages,
                    source_limit_notice=source_limit_notice,
                )
                if tag_record.wiki_policy == "overview":
                    body = self.synthesizer.synthesize_overview(synthesis_input)
                else:
                    body = self.synthesizer.synthesize(synthesis_input)
                source_hashes = {
                    record.note_path: record.content_hash
                    for record in all_source_records
                }
                wiki_path = wiki_path_for_tag(self.wiki_dir, tag)
                wiki_path.parent.mkdir(parents=True, exist_ok=True)
                new_record = replace(
                    tag_record,
                    wiki_path=display_path(wiki_path, self.vault_root),
                    dirty=False,
                    generated_at=tag_record.generated_at or now_iso(),
                    updated_at=now_iso(),
                    source_hashes=source_hashes,
                )
                wiki_path.write_text(
                    render_wiki_markdown(
                        tag_record=new_record,
                        title=wiki_title_from_tag(tag),
                        body=body,
                        source_hashes=source_hashes,
                        visible_source_paths=[record.note_path for record in source_records],
                        related_wiki_pages=related_wiki_pages,
                    ),
                    encoding="utf-8",
                )
                tags[tag] = new_record
                generated += 1
            except Exception:
                failed += 1

        self.state_store.save(WikiState(files=state.files, tags=tags))
        return {
            "generated": generated,
            "skipped": skipped,
            "failed": failed,
            "tags": len(tags),
        }

    def report(self) -> dict:
        state = rebuild_tag_index(
            self.state_store.load(),
            overview_note_threshold=self.overview_note_threshold,
        )
        tag_rows = []
        for tag, record in sorted(state.tags.items(), key=lambda item: (-len(item[1].source_paths), item[0])):
            tag_rows.append(
                {
                    "tag": tag,
                    "note_count": len(record.source_paths),
                    "candidate_kind": record.candidate_kind,
                    "evidence_counts": {
                        source: len(paths)
                        for source, paths in sorted(record.evidence.items())
                    },
                    "dirty": record.dirty,
                    "wiki_policy": record.wiki_policy,
                    "wiki_policy_source": record.wiki_policy_source,
                    "wiki_path": record.wiki_path,
                    "review_hints": build_review_hints(tag, record),
                    "eligible": record.wiki_policy != "skip"
                    and should_generate_tag(record, self.min_notes_per_tag, self.min_tag_depth),
                }
            )
        return {
            "files": len(state.files),
            "tags": len(state.tags),
            "eligible_tags": sum(1 for row in tag_rows if row["eligible"]),
            "dirty_tags": sum(1 for row in tag_rows if row["dirty"]),
            "tag_rows": tag_rows,
        }


def rebuild_tag_index(state: WikiState, *, overview_note_threshold: int = 30) -> WikiState:
    old_tags = state.tags
    tag_sources: dict[str, list[str]] = {}
    tag_evidence: dict[str, dict[str, list[str]]] = {}
    for note_path, record in state.files.items():
        for tag in sorted(tag for tag in set(record.user_tags) if is_wiki_user_tag(tag)):
            if not tag:
                continue
            tag_sources.setdefault(tag, []).append(note_path)
            tag_evidence.setdefault(tag, {}).setdefault("user_tags", []).append(note_path)
        for tag in sorted(set(record.llm_tags)):
            if not tag:
                continue
            tag_sources.setdefault(tag, []).append(note_path)
            tag_evidence.setdefault(tag, {}).setdefault("llm_tags", []).append(note_path)

    tags: dict[str, WikiTagRecord] = {}
    for tag, source_paths in sorted(tag_sources.items()):
        source_paths = sorted(set(source_paths))
        evidence = normalize_candidate_evidence(tag_evidence.get(tag, {}))
        previous = old_tags.get(tag)
        source_hashes = {
            path: state.files[path].content_hash
            for path in source_paths
            if path in state.files
        }
        previous_evidence = normalize_candidate_evidence(previous.evidence if previous else {})
        dirty = (
            previous is None
            or previous.source_hashes != source_hashes
            or previous_evidence != evidence
        )
        policy, policy_source = resolve_wiki_policy(
            previous=previous,
            tag=tag,
            source_count=len(source_paths),
            evidence=evidence,
            overview_note_threshold=overview_note_threshold,
        )
        tags[tag] = WikiTagRecord(
            tag=tag,
            source_paths=source_paths,
            candidate_kind=previous.candidate_kind if previous else "tag_derived",
            evidence=evidence,
            wiki_path=previous.wiki_path if previous else None,
            dirty=dirty or (previous.dirty if previous else False),
            wiki_policy=policy,
            wiki_policy_source=policy_source,
            generated_at=previous.generated_at if previous else None,
            updated_at=previous.updated_at if previous else None,
            source_hashes=previous.source_hashes if previous else {},
        )

    return WikiState(files=state.files, tags=tags)


def normalize_candidate_evidence(evidence: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        str(source): sorted({str(path) for path in paths if str(path)})
        for source, paths in sorted(evidence.items())
        if paths
    }


def set_tag_policy(state: WikiState, tag: str, wiki_policy: str) -> WikiState:
    if wiki_policy not in {"generate", "overview", "skip"}:
        raise ValueError(f"Unsupported wiki_policy: {wiki_policy}")
    if tag not in state.tags:
        raise ValueError(f"Tag not found in wiki state: {tag}")

    tags = dict(state.tags)
    record = tags[tag]
    tags[tag] = replace(record, wiki_policy=wiki_policy, wiki_policy_source="manual", dirty=True)
    return WikiState(files=state.files, tags=tags)


def import_manual_policies(target_state: WikiState, source_state: WikiState) -> tuple[WikiState, int]:
    tags = dict(target_state.tags)
    imported = 0
    for tag, source_record in source_state.tags.items():
        if source_record.wiki_policy_source != "manual" or tag not in tags:
            continue
        target_record = tags[tag]
        if (
            target_record.wiki_policy == source_record.wiki_policy
            and target_record.wiki_policy_source == "manual"
        ):
            continue
        tags[tag] = replace(
            target_record,
            wiki_policy=source_record.wiki_policy,
            wiki_policy_source="manual",
            dirty=True,
        )
        imported += 1
    return WikiState(files=target_state.files, tags=tags), imported


def resolve_wiki_policy(
    *,
    previous: WikiTagRecord | None,
    tag: str,
    source_count: int,
    evidence: dict[str, list[str]],
    overview_note_threshold: int,
) -> tuple[str, str]:
    if previous is not None:
        source = previous.wiki_policy_source
        if source == "manual":
            return previous.wiki_policy, "manual"

    if tag_depth(tag) == 1:
        return "skip", "auto"

    if not is_auto_generatable_candidate(source_count=source_count, evidence=evidence):
        return "skip", "auto"

    if source_count >= overview_note_threshold:
        return "overview", "auto"

    return "generate", "auto"


def is_auto_generatable_candidate(
    *,
    source_count: int,
    evidence: dict[str, list[str]],
) -> bool:
    user_count = len(evidence.get("user_tags", []))
    llm_count = len(evidence.get("llm_tags", []))
    if user_count >= 2:
        return True
    if user_count >= 1 and llm_count >= 1 and source_count >= 2:
        return True
    return False


def build_review_hints(tag: str, record: WikiTagRecord) -> list[str]:
    hints: list[str] = []
    leaf = tag_leaf(tag).lower()
    source_count = len(record.source_paths)

    if record.wiki_policy == "generate" and source_count >= 8:
        hints.append("large_generate")

    if leaf in {"参考", "资料", "资源", "收藏", "链接"}:
        hints.append("reference_like")

    if leaf in {"路径", "路线", "学习路径", "全栈"} or "路径" in leaf or "路线" in leaf:
        hints.append("navigation_like")

    if record.wiki_policy == "generate" and len(record.evidence.get("user_tags", [])) == 0:
        hints.append("llm_only_generate")

    return hints


def tag_leaf(tag: str) -> str:
    parts = [part.strip() for part in tag.split("/") if part.strip()]
    return parts[-1] if parts else ""


def build_related_wiki_pages(
    *,
    current_tag: str,
    current_record: WikiTagRecord,
    all_tags: dict[str, WikiTagRecord],
    limit: int = 8,
) -> list[RelatedWikiPage]:
    pages: list[tuple[int, RelatedWikiPage]] = []
    current_sources = set(current_record.source_paths)

    for tag, record in all_tags.items():
        if tag == current_tag or not record.wiki_path:
            continue

        overlap = len(current_sources & set(record.source_paths))
        tag_relation = related_tag_relation(current_tag, tag)
        if overlap <= 0 and tag_relation is None:
            continue

        relation_parts: list[str] = []
        score = 0
        if overlap > 0:
            relation_parts.append(f"source_overlap:{overlap}")
            score += 100 + overlap
        if tag_relation is not None:
            relation_parts.append(tag_relation)
            score += 50

        pages.append(
            (
                score,
                RelatedWikiPage(
                    tag=tag,
                    wiki_path=record.wiki_path,
                    relation=", ".join(relation_parts),
                ),
            )
        )

    pages.sort(key=lambda item: (-item[0], item[1].tag))
    return [page for _, page in pages[:limit]]


def related_tag_relation(a: str, b: str) -> str | None:
    a_parts = tag_parts(a)
    b_parts = tag_parts(b)
    if not a_parts or not b_parts:
        return None

    if is_prefix(a_parts, b_parts):
        return "parent_child"
    if is_prefix(b_parts, a_parts):
        return "child_parent"
    if len(a_parts) >= 2 and len(b_parts) >= 2 and a_parts[:-1] == b_parts[:-1]:
        return "shared_parent"
    return None


def synthesize_sort_key(tag: str, record: WikiTagRecord) -> tuple[int, int, str]:
    if record.wiki_policy == "overview":
        return (0, -len(record.source_paths), tag)
    if record.wiki_policy == "generate":
        return (1, len(record.source_paths), tag)
    return (2, len(record.source_paths), tag)


def select_source_records_for_synthesis(
    *,
    tag: str,
    tag_record: WikiTagRecord,
    source_records: list[NoteTagRecord],
) -> tuple[list[NoteTagRecord], str | None]:
    if tag_record.wiki_policy != "overview":
        return source_records, None

    selected = select_representative_source_records(tag=tag, source_records=source_records)
    if len(selected) == len(source_records):
        return selected, None

    notice = (
        f"Overview source packing: this tag has {len(source_records)} source notes. "
        f"Only {len(selected)} representative notes are included below. "
        "Use them to infer subtopics and navigation structure; do not claim the list is exhaustive."
    )
    return selected, notice


def select_representative_source_records(
    *,
    tag: str,
    source_records: list[NoteTagRecord],
    max_records: int = 24,
    max_per_group: int = 4,
) -> list[NoteTagRecord]:
    groups: dict[str, list[NoteTagRecord]] = {}
    for record in sorted(
        source_records,
        key=lambda item: source_record_sort_key(tag, item),
    ):
        groups.setdefault(source_group_key(record.note_path), []).append(record)

    selected: list[NoteTagRecord] = []
    for group, records in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        selected.extend(records[:max_per_group])
        if len(selected) >= max_records:
            break

    return selected[:max_records]


def source_record_sort_key(tag: str, record: NoteTagRecord) -> tuple[int, int, str]:
    manual_match = tag in record.user_tags
    return (
        0 if manual_match else 1,
        len(record.note_path),
        record.note_path,
    )


def source_group_key(note_path: str) -> str:
    parts = [part for part in note_path.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else ""


def tag_parts(tag: str) -> list[str]:
    return [part for part in tag.split("/") if part.strip()]


def is_prefix(prefix: list[str], value: list[str]) -> bool:
    return len(prefix) < len(value) and value[: len(prefix)] == prefix


def scan_markdown_files(vault_root: Path):
    for path in vault_root.rglob("*.md"):
        rel_parts = path.relative_to(vault_root).parts
        if any(part in DEFAULT_EXCLUDED_DIRS for part in rel_parts):
            continue
        yield path


def build_existing_tag_tree(
    state: WikiState,
    markdown_files: list[Path],
    vault_root: Path,
) -> list[str]:
    user_tags = collect_user_tags(state, markdown_files, vault_root)
    accepted_tags = collect_accepted_state_tags(state)
    allowed_roots = {
        tag_root(tag)
        for tag in user_tags | accepted_tags
        if tag_depth(tag) >= 2 and tag_root(tag)
    }

    tags = set()
    for tag in user_tags | accepted_tags:
        if tag_depth(tag) >= 2 and tag_root(tag) in allowed_roots:
            tags.add(tag)

    return sorted(tags)


def collect_user_tags(
    state: WikiState,
    markdown_files: list[Path],
    vault_root: Path,
) -> set[str]:
    tags: set[str] = set()
    for record in state.files.values():
        tags.update(record.user_tags)
    for file_path in markdown_files:
        try:
            tags.update(extract_user_tags(file_path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return {tag for tag in tags if is_wiki_user_tag(tag)}


def collect_accepted_state_tags(state: WikiState) -> set[str]:
    tags: set[str] = set()
    for tag, record in state.tags.items():
        if tag_depth(tag) < 2:
            continue
        if record.wiki_policy == "skip":
            continue
        if record.wiki_path or record.wiki_policy_source == "manual" or not record.dirty:
            tags.add(tag)
    return tags


def tag_root(tag: str) -> str:
    parts = [part.strip() for part in tag.split("/") if part.strip()]
    return parts[0] if parts else ""


def is_wiki_user_tag(tag: str) -> bool:
    return bool(tag) and tag_depth(tag) >= 2


def should_generate_tag(record: WikiTagRecord, min_notes_per_tag: int, min_tag_depth: int) -> bool:
    return len(record.source_paths) >= min_notes_per_tag and tag_depth(record.tag) >= min_tag_depth


def tag_depth(tag: str) -> int:
    return len([part for part in tag.split("/") if part.strip()])


def wiki_path_for_tag(wiki_dir: Path, tag: str) -> Path:
    return wiki_dir / f"{slug_from_tag(tag)}.md"


def display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
