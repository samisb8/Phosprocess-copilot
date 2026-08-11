"""Safe one-command synchronization of the active production corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import pymupdf

from phosprocess.knowledge_base.manifest import (
    KnowledgeBaseManifest,
    ManifestDocument,
)
from phosprocess.knowledge_base.models import (
    KNOWLEDGE_BASE_PIPELINE_VERSION,
    IndexBuildResult,
    ProcessedDocument,
    sha256_file,
)
from phosprocess.knowledge_base.runtime import (
    DEFAULT_KNOWLEDGE_BASE_ROOT,
    PROJECT_ROOT,
    ActiveKnowledgeBase,
    ActiveKnowledgeBaseError,
    load_active_knowledge_base,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk


class KnowledgeBaseSyncError(RuntimeError):
    """Raised when a new version cannot be safely built or activated."""


class DocumentProcessor(Protocol):
    """Dependency boundary used by real and simulated ingestion tests."""

    def process(
        self,
        *,
        pdf_path: Path,
        document_id: str,
        document_sha256: str,
        cache_directory: Path,
    ) -> ProcessedDocument: ...


class IndexBuilder(Protocol):
    """Dependency boundary for immutable version construction."""

    def build(
        self,
        *,
        records: list[dict[str, Any]],
        version_directory: Path,
        previous_version_directory: Path | None,
    ) -> IndexBuildResult: ...


@dataclass(frozen=True, slots=True)
class KnowledgeBasePaths:
    """All production paths rooted outside evaluation artifacts."""

    root: Path
    pdfs: Path
    archive: Path
    rejected: Path
    processed: Path
    corpus: Path
    indexes: Path
    versions: Path
    manifest: Path
    current_index: Path

    @classmethod
    def from_root(cls, root: Path) -> KnowledgeBasePaths:
        resolved = root.resolve()
        indexes = resolved / "indexes"
        return cls(
            root=resolved,
            pdfs=resolved / "pdfs",
            archive=resolved / "archive",
            rejected=resolved / "rejected",
            processed=resolved / "processed",
            corpus=resolved / "corpus",
            indexes=indexes,
            versions=indexes / "versions",
            manifest=resolved / "manifest.sqlite",
            current_index=resolved / "current_index.json",
        )

    def create(self) -> None:
        """Create the requested directory structure."""

        for directory in (
            self.pdfs,
            self.archive,
            self.rejected,
            self.processed,
            self.corpus,
            self.versions,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class ScannedPDF:
    """One detected PDF and its deterministic synchronization status."""

    filename: str
    path: Path
    document_id: str
    sha256: str
    page_count: int
    status: str
    error: str | None = None
    duplicate_of: str | None = None

    @property
    def active_candidate(self) -> bool:
        return self.status in {"new", "modified", "unchanged"}


@dataclass(frozen=True, slots=True)
class SyncPlan:
    """Read-only comparison between disk and the current manifest."""

    entries: tuple[ScannedPDF, ...]
    removed: tuple[ManifestDocument, ...]
    previous_active: Mapping[str, ManifestDocument]

    @property
    def active_entries(self) -> tuple[ScannedPDF, ...]:
        return tuple(entry for entry in self.entries if entry.active_candidate)

    def count(self, status: str) -> int:
        return sum(entry.status == status for entry in self.entries)

    @property
    def summary(self) -> dict[str, int]:
        return {
            "detected": len(self.entries),
            "new": self.count("new"),
            "modified": self.count("modified"),
            "unchanged": self.count("unchanged"),
            "removed": len(self.removed),
            "duplicates": self.count("duplicate"),
            "invalid": self.count("invalid"),
        }

    @property
    def changes_require_rebuild(self) -> bool:
        return bool(self.count("new") or self.count("modified") or self.removed)


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Outcome printed by the CLI and asserted by tests."""

    changed: bool
    dry_run: bool
    version: str | None
    document_count: int
    chunk_count: int
    documents: tuple[ManifestDocument, ...]
    archived: tuple[str, ...]
    rejected: tuple[str, ...]
    plan: SyncPlan
    embedded_chunk_count: int = 0
    reused_embedding_count: int = 0


def document_id_from_filename(filename: str) -> str:
    """Create the same stable filename-derived ID used by existing indexes."""

    raw = Path(filename).stem.casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

    if not normalized:
        raise ValueError(f"Nom de PDF inexploitable : {filename}")

    return normalized


class KnowledgeBaseManager:
    """Synchronize PDFs into a validated immutable production index."""

    def __init__(
        self,
        *,
        root: Path = DEFAULT_KNOWLEDGE_BASE_ROOT,
        project_root: Path = PROJECT_ROOT,
        processor: DocumentProcessor | None = None,
        index_builder: IndexBuilder | None = None,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.paths = KnowledgeBasePaths.from_root(root)
        self.manifest = KnowledgeBaseManifest(self.paths.manifest)
        self._processor = processor
        self._index_builder = index_builder
        self.failure_injector = failure_injector

    def _get_processor(self) -> DocumentProcessor:
        if self._processor is None:
            from phosprocess.knowledge_base.indexing import (
                ProductionDocumentProcessor,
            )

            self._processor = ProductionDocumentProcessor(
                chunking_config_path=(self.project_root / "configs" / "chunking.yaml"),
                postprocessing_config_path=(
                    self.project_root / "configs" / "chunk_postprocessing.yaml"
                ),
            )

        return self._processor

    def _get_index_builder(self) -> IndexBuilder:
        if self._index_builder is None:
            from phosprocess.knowledge_base.indexing import (
                VersionIndexBuilder,
            )

            self._index_builder = VersionIndexBuilder(
                embedding_config_path=(self.project_root / "configs" / "embeddings.yaml"),
                retrieval_config_path=(
                    self.project_root
                    / "data"
                    / "evaluation"
                    / "retrieval"
                    / "v0.1"
                    / "frozen"
                    / "dev_best_v3"
                    / "retrieval_v2.yaml"
                ),
                legacy_dense_directory=(
                    self.project_root / "data" / "indexes" / "dense" / "bge_m3"
                ),
                embedding_cache_path=(self.paths.processed / "embedding_cache.sqlite"),
            )

        return self._index_builder

    @staticmethod
    def _validate_pdf(path: Path) -> tuple[bool, int, str | None]:
        try:
            with pymupdf.open(path) as document:
                if not document.is_pdf:
                    return False, 0, "Le fichier n'est pas un PDF."

                if document.needs_pass:
                    return False, 0, "Le PDF est protégé par mot de passe."

                if document.page_count <= 0:
                    return False, 0, "Le PDF ne contient aucune page."

                sample_size = min(12, document.page_count)
                sample_pages = sorted(
                    {
                        round(index * (document.page_count - 1) / max(1, sample_size - 1))
                        for index in range(sample_size)
                    }
                )
                sampled_text = "\n".join(
                    document.load_page(page_number).get_text("text") for page_number in sample_pages
                )
                visible_count = sum(not character.isspace() for character in sampled_text)
                letter_count = sum(character.isalpha() for character in sampled_text)
                word_count = len(
                    re.findall(
                        r"(?u)[^\W\d_]{3,}",
                        sampled_text,
                    )
                )

                if letter_count == 0:
                    return (
                        False,
                        int(document.page_count),
                        "Le PDF ne contient aucun texte extractible.",
                    )

                if visible_count >= 500 and (
                    letter_count / visible_count < 0.15 or word_count < 10
                ):
                    return (
                        False,
                        int(document.page_count),
                        "La couche texte du PDF est inexploitable "
                        "(encodage de police ou OCR requis).",
                    )

                return True, int(document.page_count), None
        except Exception as error:
            return False, 0, f"{type(error).__name__}: {error}"

    def _current_optional(self) -> ActiveKnowledgeBase | None:
        if not self.paths.current_index.is_file():
            return None

        return load_active_knowledge_base(
            self.paths.current_index,
            project_root=self.project_root,
            knowledge_base_root=self.paths.root,
        )

    def scan(self) -> SyncPlan:
        """Scan active PDFs without creating or modifying any path."""

        previous_active = self.manifest.active_documents()
        paths = (
            sorted(
                (
                    path
                    for path in self.paths.pdfs.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".pdf"
                ),
                key=lambda path: path.name.casefold(),
            )
            if self.paths.pdfs.is_dir()
            else []
        )
        preliminary: list[ScannedPDF] = []

        for path in paths:
            digest = sha256_file(path)
            valid, page_count, error = self._validate_pdf(path)
            preliminary.append(
                ScannedPDF(
                    filename=path.name,
                    path=path,
                    document_id=document_id_from_filename(path.name),
                    sha256=digest,
                    page_count=page_count,
                    status="pending" if valid else "invalid",
                    error=error,
                )
            )

        valid_by_sha: dict[str, list[ScannedPDF]] = {}

        for entry in preliminary:
            if entry.status != "invalid":
                valid_by_sha.setdefault(entry.sha256, []).append(entry)

        canonical_names: dict[str, str] = {}

        for digest, entries in valid_by_sha.items():
            previous_names = {
                entry.filename
                for entry in entries
                if entry.filename in previous_active
                and previous_active[entry.filename].sha256 == digest
            }
            canonical_names[digest] = (
                sorted(previous_names, key=str.casefold)[0]
                if previous_names
                else entries[0].filename
            )

        entries: list[ScannedPDF] = []

        for entry in preliminary:
            if entry.status == "invalid":
                entries.append(entry)
                continue

            canonical = canonical_names[entry.sha256]

            if entry.filename != canonical:
                entries.append(
                    ScannedPDF(
                        filename=entry.filename,
                        path=entry.path,
                        document_id=entry.document_id,
                        sha256=entry.sha256,
                        page_count=entry.page_count,
                        status="duplicate",
                        duplicate_of=canonical,
                    )
                )
                continue

            previous = previous_active.get(entry.filename)
            status = (
                "new"
                if previous is None
                else ("unchanged" if previous.sha256 == entry.sha256 else "modified")
            )
            entries.append(
                ScannedPDF(
                    filename=entry.filename,
                    path=entry.path,
                    document_id=entry.document_id,
                    sha256=entry.sha256,
                    page_count=entry.page_count,
                    status=status,
                )
            )

        active_names = {entry.filename for entry in entries if entry.active_candidate}
        removed = tuple(
            document
            for filename, document in sorted(previous_active.items())
            if filename not in active_names
        )

        document_ids = [entry.document_id for entry in entries if entry.active_candidate]

        if len(document_ids) != len(set(document_ids)):
            raise KnowledgeBaseSyncError("Deux PDF actifs produisent le même document_id.")

        return SyncPlan(
            entries=tuple(entries),
            removed=removed,
            previous_active=previous_active,
        )

    @staticmethod
    def print_plan(plan: SyncPlan) -> None:
        """Print the requested pre-processing summary."""

        summary = plan.summary
        print("Base documentaire active")
        print("-------------------------")
        print(f"PDF détectés       : {summary['detected']}")
        print(f"Nouveaux           : {summary['new']}")
        print(f"Modifiés           : {summary['modified']}")
        print(f"Inchangés          : {summary['unchanged']}")
        print(f"Retirés            : {summary['removed']}")
        print(f"Dupliqués          : {summary['duplicates']}")
        print(f"Invalides          : {summary['invalid']}")

    def _cache_directory(
        self,
        document_id: str,
        digest: str,
    ) -> Path:
        return self.paths.processed / "documents" / document_id / digest

    @staticmethod
    def _read_chunk_records(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    value = json.loads(line)

                    if not isinstance(value, dict):
                        raise ValueError(f"Enregistrement de chunk invalide : {path}")

                    records.append(value)

        return records

    def _load_cached_document(
        self,
        entry: ScannedPDF,
    ) -> ProcessedDocument | None:
        cache = self._cache_directory(
            entry.document_id,
            entry.sha256,
        )
        manifest_path = cache / "manifest.json"
        chunks_path = cache / "chunks.jsonl"

        if not manifest_path.is_file() or not chunks_path.is_file():
            return None

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = self._read_chunk_records(chunks_path)
            chunks = tuple(DocumentChunk.model_validate(record) for record in records)
        except Exception:
            return None

        if (
            not isinstance(payload, dict)
            or payload.get("status") != "success"
            or payload.get("pipeline_version") != KNOWLEDGE_BASE_PIPELINE_VERSION
            or payload.get("document_sha256") != entry.sha256
            or payload.get("filename") != entry.filename
            or not chunks
            or any(
                chunk.document_id != entry.document_id or chunk.source_file != entry.filename
                for chunk in chunks
            )
        ):
            return None

        return ProcessedDocument(
            filename=entry.filename,
            document_id=entry.document_id,
            document_sha256=entry.sha256,
            page_count=int(payload["page_count"]),
            empty_pages=tuple(payload.get("empty_pages", [])),
            chunks=chunks,
            duplicates_removed=int(payload.get("duplicates_removed", 0)),
            ingestion_date=str(payload["ingestion_date"]),
            cache_directory=cache,
        )

    def _load_legacy_document(
        self,
        entry: ScannedPDF,
    ) -> ProcessedDocument | None:
        legacy_source = self.project_root / "data" / "raw" / "public" / entry.filename
        legacy_chunks = (
            self.project_root
            / "data"
            / "processed"
            / "final_chunks"
            / f"{entry.document_id}_chunks.jsonl"
        )

        if (
            not legacy_source.is_file()
            or not legacy_chunks.is_file()
            or sha256_file(legacy_source) != entry.sha256
        ):
            return None

        chunks = tuple(
            DocumentChunk.model_validate(record)
            for record in self._read_chunk_records(legacy_chunks)
        )

        if not chunks or any(
            chunk.document_id != entry.document_id or chunk.source_file != entry.filename
            for chunk in chunks
        ):
            return None

        return ProcessedDocument(
            filename=entry.filename,
            document_id=entry.document_id,
            document_sha256=entry.sha256,
            page_count=entry.page_count,
            empty_pages=(),
            chunks=chunks,
            duplicates_removed=0,
            ingestion_date=datetime.now(UTC).isoformat(),
            cache_directory=self._cache_directory(
                entry.document_id,
                entry.sha256,
            ),
        )

    @staticmethod
    def _format_pages(pages: list[int]) -> str:
        ranges: list[str] = []
        start = pages[0]
        previous = pages[0]

        for page in pages[1:]:
            if page == previous + 1:
                previous = page
                continue

            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = page
            previous = page

        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return ", ".join(ranges)

    def _load_renamed_cached_document(
        self,
        entry: ScannedPDF,
    ) -> ProcessedDocument | None:
        """Reuse extraction/chunk boundaries when identical bytes are renamed."""

        manifests_root = self.paths.processed / "documents"

        if not manifests_root.is_dir():
            return None

        for manifest_path in manifests_root.glob("*/*/manifest.json"):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            if (
                not isinstance(payload, dict)
                or payload.get("status") != "success"
                or payload.get("pipeline_version") != KNOWLEDGE_BASE_PIPELINE_VERSION
                or payload.get("document_sha256") != entry.sha256
            ):
                continue

            records_path = manifest_path.parent / "chunks.jsonl"

            if not records_path.is_file():
                continue

            source_chunks = tuple(
                DocumentChunk.model_validate(record)
                for record in self._read_chunk_records(records_path)
            )

            if not source_chunks:
                continue

            migrated_chunks: list[DocumentChunk] = []

            for chunk in source_chunks:
                pages = list(chunk.source_pages)
                context = [
                    f"Document: {entry.filename}",
                    f"Pages: {self._format_pages(pages)}",
                ]

                if chunk.heading_path:
                    context.append(f"Section: {' > '.join(chunk.heading_path)}")

                embedding_text = "\n".join(context) + "\n\n" + chunk.text
                digest_input = (
                    f"{entry.document_id}|{chunk.chunk_index}|{pages}|{chunk.text}"
                ).encode()
                digest = hashlib.sha256(digest_input).hexdigest()[:12]
                migrated_chunks.append(
                    chunk.model_copy(
                        update={
                            "chunk_id": (f"{entry.document_id}_{chunk.chunk_index:06d}_{digest}"),
                            "document_id": entry.document_id,
                            "source_file": entry.filename,
                            "embedding_text": embedding_text,
                            "source_chunk_ids": [chunk.chunk_id],
                            "postprocessing_actions": [
                                *chunk.postprocessing_actions,
                                "renamed_document_cache_reused",
                            ],
                        }
                    )
                )

            return ProcessedDocument(
                filename=entry.filename,
                document_id=entry.document_id,
                document_sha256=entry.sha256,
                page_count=int(payload["page_count"]),
                empty_pages=tuple(payload.get("empty_pages", [])),
                chunks=tuple(migrated_chunks),
                duplicates_removed=int(payload.get("duplicates_removed", 0)),
                ingestion_date=datetime.now(UTC).isoformat(),
                cache_directory=self._cache_directory(
                    entry.document_id,
                    entry.sha256,
                ),
            )

        return None

    def _save_processed_document(
        self,
        document: ProcessedDocument,
        *,
        source_path: Path,
    ) -> None:
        cache = document.cache_directory
        cache.mkdir(parents=True, exist_ok=True)
        chunks_path = cache / "chunks.jsonl"
        chunks_temporary = chunks_path.with_suffix(".jsonl.tmp")

        with chunks_temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            for record in document.metadata_records():
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(chunks_temporary, chunks_path)
        source_temporary = cache / "source.pdf.tmp"
        shutil.copyfile(source_path, source_temporary)
        os.replace(source_temporary, cache / "source.pdf")
        manifest = {
            "status": "success",
            "pipeline_version": KNOWLEDGE_BASE_PIPELINE_VERSION,
            "filename": document.filename,
            "document_id": document.document_id,
            "document_sha256": document.document_sha256,
            "page_count": document.page_count,
            "empty_pages": list(document.empty_pages),
            "chunk_count": document.chunk_count,
            "duplicates_removed": document.duplicates_removed,
            "ingestion_date": document.ingestion_date,
        }
        manifest_temporary = cache / "manifest.json.tmp"
        manifest_temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(manifest_temporary, cache / "manifest.json")

    def _prepare_document(
        self,
        entry: ScannedPDF,
        *,
        verbose: bool,
    ) -> ProcessedDocument:
        cached = self._load_cached_document(entry)

        if cached is not None:
            if verbose:
                print(f"Statut : cache valide ({cached.chunk_count} chunks)")
            return cached

        legacy = self._load_legacy_document(entry)

        if legacy is not None:
            self._save_processed_document(
                legacy,
                source_path=entry.path,
            )

            if verbose:
                print(f"Migration du corpus existant validé : {legacy.chunk_count} chunks")

            return legacy

        renamed = self._load_renamed_cached_document(entry)

        if renamed is not None:
            self._save_processed_document(
                renamed,
                source_path=entry.path,
            )

            if verbose:
                print(f"Cache de contenu renommé réutilisé : {renamed.chunk_count} chunks")

            return renamed

        processed = self._get_processor().process(
            pdf_path=entry.path,
            document_id=entry.document_id,
            document_sha256=entry.sha256,
            cache_directory=self._cache_directory(
                entry.document_id,
                entry.sha256,
            ),
        )
        self._save_processed_document(
            processed,
            source_path=entry.path,
        )
        return processed

    def _new_version_name(self) -> str:
        base = datetime.now(UTC).strftime("kb_%Y%m%d_%H%M%S")
        candidate = base
        counter = 1

        while (self.paths.versions / candidate).exists() or (
            self.paths.versions / f"{candidate}_tmp"
        ).exists():
            candidate = f"{base}_{counter:02d}"
            counter += 1

        return candidate

    def _pointer_payload(
        self,
        *,
        version: str,
        documents: list[ManifestDocument],
        build: IndexBuildResult,
    ) -> dict[str, Any]:
        version_directory = self.paths.versions / version

        try:
            path_value = version_directory.relative_to(self.project_root).as_posix()
        except ValueError:
            path_value = str(version_directory)

        return {
            "version": version,
            "path": path_value,
            "document_count": len(documents),
            "chunk_count": build.chunk_count,
            "documents": [
                {
                    "filename": document.filename,
                    "document_id": document.document_id,
                    "sha256": document.sha256,
                    "page_count": document.page_count,
                    "chunk_count": document.chunk_count,
                }
                for document in documents
            ],
            "activated_at_utc": datetime.now(UTC).isoformat(),
            "pipeline_version": KNOWLEDGE_BASE_PIPELINE_VERSION,
        }

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(content)

        for attempt in range(3):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 2:
                    temporary.unlink(missing_ok=True)
                    raise

                time.sleep(0.05)

    def _activate_pointer(self, payload: dict[str, Any]) -> None:
        self._atomic_write_bytes(
            self.paths.current_index,
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

    def _safe_remove_version(self, path: Path) -> None:
        resolved = path.resolve()

        if resolved.parent != self.paths.versions.resolve():
            raise KnowledgeBaseSyncError(f"Refus de supprimer un chemin hors versions : {resolved}")

        if resolved.exists():
            shutil.rmtree(resolved)

    @staticmethod
    def _publish_version(
        temporary_version: Path,
        final_version: Path,
    ) -> None:
        """Publish a validated version before the atomic pointer switch.

        A directory rename is preferred.  On Windows, antivirus/indexing
        services can keep short-lived handles on freshly written FAISS or
        NumPy files and make the rename fail with ``WinError 5``.  Copying to
        an as-yet unreachable final directory remains safe: readers only
        discover the version after ``current_index.json`` is replaced.
        """

        for attempt in range(30):
            try:
                os.rename(temporary_version, final_version)
                return
            except PermissionError:
                if attempt < 29:
                    time.sleep(0.1)

        if final_version.exists():
            raise FileExistsError(f"La version finale existe déjà : {final_version}")

        try:
            shutil.copytree(temporary_version, final_version)

            temporary_files = {
                path.relative_to(temporary_version): path.stat().st_size
                for path in temporary_version.rglob("*")
                if path.is_file()
            }
            final_files = {
                path.relative_to(final_version): path.stat().st_size
                for path in final_version.rglob("*")
                if path.is_file()
            }

            if temporary_files != final_files:
                raise KnowledgeBaseSyncError("La copie Windows de la version est incomplète.")

            shutil.rmtree(temporary_version)
        except Exception:
            if final_version.exists():
                shutil.rmtree(final_version)

            raise

    def _restore_pointer(self, previous_content: bytes | None) -> None:
        if previous_content is None:
            self.paths.current_index.unlink(missing_ok=True)
        else:
            self._atomic_write_bytes(
                self.paths.current_index,
                previous_content,
            )

    def _move_rejected(self, entries: tuple[ScannedPDF, ...]) -> list[str]:
        moved: list[str] = []

        for entry in entries:
            if entry.status != "invalid" or not entry.path.exists():
                continue

            target = self.paths.rejected / entry.filename

            if target.exists():
                target = self.paths.rejected / f"{Path(entry.filename).stem}_{entry.sha256[:8]}.pdf"

            entry.path.replace(target)
            moved.append(target.name)

        return moved

    def _archive_retired(
        self,
        plan: SyncPlan,
    ) -> list[str]:
        archived: list[str] = []
        retired = {(document.filename, document.sha256): document for document in plan.removed}

        for entry in plan.entries:
            if entry.status != "modified":
                continue

            previous = plan.previous_active.get(entry.filename)

            if previous is not None:
                retired[(previous.filename, previous.sha256)] = previous

        for document in retired.values():
            cached_source = (
                self._cache_directory(
                    document.document_id,
                    document.sha256,
                )
                / "source.pdf"
            )

            if not cached_source.is_file():
                continue

            target = self.paths.archive / document.filename

            if target.is_file():
                if sha256_file(target) == document.sha256:
                    continue

                target = (
                    self.paths.archive / f"{Path(document.filename).stem}_{document.sha256[:8]}.pdf"
                )

            shutil.copy2(cached_source, target)
            archived.append(target.name)

        return archived

    @staticmethod
    def _observation_documents(
        plan: SyncPlan,
    ) -> list[ManifestDocument]:
        return [
            ManifestDocument(
                filename=entry.filename,
                document_id=entry.document_id,
                sha256=entry.sha256,
                status=entry.status,
                active=False,
                page_count=entry.page_count,
                chunk_count=0,
                version=None,
                error=(
                    entry.error
                    if entry.status == "invalid"
                    else (
                        f"Doublon de {entry.duplicate_of}" if entry.status == "duplicate" else None
                    )
                ),
            )
            for entry in plan.entries
            if entry.status in {"invalid", "duplicate"}
        ]

    def _write_run_report(
        self,
        *,
        run_id: str,
        result: SyncResult,
    ) -> None:
        report = {
            "run_id": run_id,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "version": result.version,
            "document_count": result.document_count,
            "chunk_count": result.chunk_count,
            "embedded_chunk_count": result.embedded_chunk_count,
            "reused_embedding_count": result.reused_embedding_count,
            "plan": result.plan.summary,
            "documents": [
                {
                    "filename": document.filename,
                    "document_id": document.document_id,
                    "sha256": document.sha256,
                    "page_count": document.page_count,
                    "chunk_count": document.chunk_count,
                }
                for document in result.documents
            ],
            "archived": list(result.archived),
            "rejected": list(result.rejected),
        }
        report_path = self.paths.processed / f"sync_{run_id}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def sync(
        self,
        *,
        dry_run: bool = False,
        rebuild: bool = False,
        verbose: bool = False,
    ) -> SyncResult:
        """Synchronize all active PDFs and atomically publish one version."""

        plan = self.scan()
        self.print_plan(plan)

        if dry_run:
            current = self._current_optional()
            return SyncResult(
                changed=False,
                dry_run=True,
                version=current.version if current else None,
                document_count=(current.document_count if current else 0),
                chunk_count=current.chunk_count if current else 0,
                documents=tuple(plan.previous_active.values()),
                archived=(),
                rejected=(),
                plan=plan,
            )

        self.paths.create()
        current = self._current_optional()
        needs_version = rebuild or current is None or plan.changes_require_rebuild
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        version = self._new_version_name() if needs_version else None
        self.manifest.start_run(
            run_id,
            previous_version=current.version if current else None,
            requested_version=version,
            rebuild=rebuild,
            dry_run=False,
            summary=plan.summary,
        )
        rejected = self._move_rejected(plan.entries)
        observations = self._observation_documents(plan)

        if observations:
            self.manifest.record_observations(observations)

        if not needs_version:
            documents = tuple(self.manifest.active_documents().values())
            result = SyncResult(
                changed=False,
                dry_run=False,
                version=current.version if current else None,
                document_count=len(documents),
                chunk_count=current.chunk_count if current else 0,
                documents=documents,
                archived=(),
                rejected=tuple(rejected),
                plan=plan,
            )
            self.manifest.finish_run(
                run_id,
                status="success",
                activated_version=result.version,
                summary=plan.summary,
            )
            self._write_run_report(run_id=run_id, result=result)
            print("Aucun changement actif : index conservé.")
            return result

        if not plan.active_entries:
            error = "La base active ne peut pas être vide. Ajoutez au moins un PDF valide."
            self.manifest.finish_run(
                run_id,
                status="failed",
                error=error,
            )
            raise KnowledgeBaseSyncError(error)

        assert version is not None
        temporary_version = self.paths.versions / f"{version}_tmp"
        final_version = self.paths.versions / version
        previous_pointer = (
            self.paths.current_index.read_bytes() if self.paths.current_index.is_file() else None
        )
        pointer_activated = False

        try:
            prepared_documents: list[ProcessedDocument] = []

            for position, entry in enumerate(
                plan.active_entries,
                start=1,
            ):
                print(f"\n[{position}/{len(plan.active_entries)}] {entry.filename}")
                print(f"Statut : {entry.status}")
                document = self._prepare_document(
                    entry,
                    verbose=verbose,
                )
                prepared_documents.append(document)
                print(f"Pages : {document.page_count}")
                print(f"Chunks : {document.chunk_count}")
                print(f"Doublons ignorés : {document.duplicates_removed}")

            records = [
                record
                for document in sorted(
                    prepared_documents,
                    key=lambda item: item.document_id,
                )
                for record in document.metadata_records()
            ]

            if self.failure_injector is not None:
                self.failure_injector("before_build")

            build = self._get_index_builder().build(
                records=records,
                version_directory=temporary_version,
                previous_version_directory=(
                    current.version_directory if current is not None else None
                ),
            )
            expected_active_files = {
                (entry.filename, entry.sha256) for entry in plan.active_entries
            }
            current_active_files = {
                (entry.filename, entry.sha256) for entry in self.scan().active_entries
            }

            if current_active_files != expected_active_files:
                raise KnowledgeBaseSyncError(
                    "Le dossier pdfs a changé pendant la synchronisation. "
                    "Relancez la commande pour indexer son état courant."
                )

            if self.failure_injector is not None:
                self.failure_injector("after_validation")

            self._publish_version(
                temporary_version,
                final_version,
            )
            documents = [
                ManifestDocument(
                    filename=document.filename,
                    document_id=document.document_id,
                    sha256=document.document_sha256,
                    status="active",
                    active=True,
                    page_count=document.page_count,
                    chunk_count=document.chunk_count,
                    version=version,
                )
                for document in prepared_documents
            ]
            pointer = self._pointer_payload(
                version=version,
                documents=documents,
                build=build,
            )

            if self.failure_injector is not None:
                self.failure_injector("before_activation")

            self._activate_pointer(pointer)
            pointer_activated = True
            retired_statuses = {
                (document.filename, document.sha256): "removed" for document in plan.removed
            }

            for entry in plan.entries:
                if entry.status != "modified":
                    continue

                previous = plan.previous_active.get(entry.filename)

                if previous is not None:
                    retired_statuses[(previous.filename, previous.sha256)] = "superseded"

            self.manifest.activate_documents(
                documents,
                version=version,
                retired_statuses=retired_statuses,
            )
            archived = self._archive_retired(plan)
            result = SyncResult(
                changed=True,
                dry_run=False,
                version=version,
                document_count=len(documents),
                chunk_count=build.chunk_count,
                documents=tuple(documents),
                archived=tuple(archived),
                rejected=tuple(rejected),
                plan=plan,
                embedded_chunk_count=build.embedded_chunk_count,
                reused_embedding_count=build.reused_embedding_count,
            )
            self.manifest.finish_run(
                run_id,
                status="success",
                activated_version=version,
                summary={
                    **plan.summary,
                    "document_count": len(documents),
                    "chunk_count": build.chunk_count,
                    "embedded_chunk_count": (build.embedded_chunk_count),
                    "reused_embedding_count": (build.reused_embedding_count),
                },
            )
            self._write_run_report(run_id=run_id, result=result)
            return result

        except Exception as error:
            if pointer_activated:
                self._restore_pointer(previous_pointer)

            for path in (temporary_version, final_version):
                if path.exists():
                    self._safe_remove_version(path)

            self.manifest.finish_run(
                run_id,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
            raise KnowledgeBaseSyncError(
                "La synchronisation a échoué ; l'ancien index actif a été conservé."
            ) from error

    def list_active(self) -> tuple[ManifestDocument, ...]:
        """Print and return active documents."""

        documents = tuple(self.manifest.active_documents().values())

        if not documents:
            print("Aucun document actif.")
            return documents

        print(f"Documents actifs : {len(documents)}")

        for document in documents:
            print(
                f"- {document.filename} | chunks={document.chunk_count} | sha256={document.sha256}"
            )

        return documents

    def status(self) -> dict[str, Any]:
        """Print current pointer, counts and latest manifest run."""

        try:
            current = self._current_optional()
        except ActiveKnowledgeBaseError as error:
            current = None
            print(f"Base documentaire invalide : {error}")

        latest_run = self.manifest.run_status()

        if current is None:
            print("Base documentaire active : aucune")
        else:
            print(f"Base documentaire : {current.version}")
            print(f"Documents actifs : {current.document_count}")
            print(f"Chunks actifs : {current.chunk_count}")
            print(f"Chemin : {current.version_directory}")

        if latest_run is not None:
            print(f"Dernière synchronisation : {latest_run['status']} ({latest_run['run_id']})")

        return {
            "current": current,
            "latest_run": latest_run,
        }
