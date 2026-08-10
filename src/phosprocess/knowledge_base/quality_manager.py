"""Transactional builder for the independent production quality index."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from phosprocess.knowledge_base.catalog import (
    load_document_catalog,
    verify_catalogue_sources,
)
from phosprocess.knowledge_base.indexing import VersionIndexBuilder
from phosprocess.knowledge_base.models import IndexBuildResult
from phosprocess.knowledge_base.quality_corpus import QualityCorpusProcessor
from phosprocess.knowledge_base.quality_indexing import (
    QUALITY_INDEX_PIPELINE_VERSION,
    QualityIndexBuilder,
)
from phosprocess.knowledge_base.runtime import (
    DEFAULT_KNOWLEDGE_BASE_ROOT,
    PROJECT_ROOT,
    ActiveKnowledgeBaseError,
    load_active_knowledge_base,
)


class QualityKnowledgeBaseError(RuntimeError):
    """Raised before an invalid quality index can become active."""


@dataclass(frozen=True, slots=True)
class QualityDocumentResult:
    filename: str
    document_id: str
    sha256: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class QualitySyncResult:
    changed: bool
    dry_run: bool
    version: str | None
    previous_version: str | None
    document_count: int
    chunk_count: int
    documents: tuple[QualityDocumentResult, ...]
    embedded_chunk_count: int = 0
    reused_embedding_count: int = 0
    archived: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(content)

    for attempt in range(30):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt < 29:
                time.sleep(0.1)

    temporary.unlink(missing_ok=True)
    raise PermissionError(f"Activation atomique impossible : {path}")


def _publish_directory(temporary: Path, final: Path) -> None:
    for attempt in range(30):
        try:
            os.rename(temporary, final)
            return
        except PermissionError:
            if attempt < 29:
                time.sleep(0.1)

    if final.exists():
        raise FileExistsError(f"La version existe déjà : {final}")

    try:
        shutil.copytree(temporary, final)
        source_files = {
            item.relative_to(temporary): item.stat().st_size
            for item in temporary.rglob("*")
            if item.is_file()
        }
        final_files = {
            item.relative_to(final): item.stat().st_size
            for item in final.rglob("*")
            if item.is_file()
        }

        if source_files != final_files:
            raise QualityKnowledgeBaseError("La publication Windows de l'index est incomplète.")

        shutil.rmtree(temporary)
    except Exception:
        if final.exists():
            shutil.rmtree(final)

        raise


class QualityKnowledgeBaseManager:
    """Prepare all eight catalogued books then switch one safe pointer."""

    def __init__(
        self,
        *,
        root: Path = DEFAULT_KNOWLEDGE_BASE_ROOT,
        project_root: Path = PROJECT_ROOT,
        corpus_processor: QualityCorpusProcessor | None = None,
        index_builder: QualityIndexBuilder | None = None,
    ) -> None:
        self.root = root.resolve()
        self.project_root = project_root.resolve()
        self.pdfs = self.root / "pdfs"
        self.versions = self.root / "indexes" / "versions"
        self.pointer = self.root / "current_index.json"
        self.catalog = load_document_catalog()
        self.corpus_processor = corpus_processor or QualityCorpusProcessor(
            knowledge_base_root=self.root
        )
        self.index_builder = index_builder or QualityIndexBuilder(
            VersionIndexBuilder(
                embedding_config_path=self.project_root / "configs" / "embeddings.yaml",
                retrieval_config_path=self.project_root
                / "data"
                / "evaluation"
                / "retrieval"
                / "v0.1"
                / "frozen"
                / "dev_best_v3"
                / "retrieval_v2.yaml",
                embedding_cache_path=self.root / "processed" / "embedding_cache.sqlite",
            )
        )

    def _verified_active_documents(self) -> tuple[Any, ...]:
        if len(self.catalog.documents) != 8:
            raise QualityKnowledgeBaseError("Le catalogue doit contenir exactement huit documents.")

        inactive = [entry.document_id for entry in self.catalog.documents if not entry.active]

        if inactive:
            raise QualityKnowledgeBaseError(
                "Documents inactifs dans le catalogue : " + ", ".join(inactive)
            )

        verification = verify_catalogue_sources(
            self.catalog,
            knowledge_base_root=self.root,
        )
        invalid = {
            document_id: reason
            for document_id, (valid, reason) in verification.items()
            if not valid
        }

        if invalid:
            details = ", ".join(
                f"{document_id}={reason}" for document_id, reason in sorted(invalid.items())
            )
            raise QualityKnowledgeBaseError(
                f"Le corpus physique ne correspond pas au catalogue : {details}"
            )

        sources = {path.name for path in self.pdfs.glob("*.pdf")}
        expected = {entry.source_filename for entry in self.catalog.documents}

        if sources != expected:
            unexpected = sorted(sources - expected)
            missing = sorted(expected - sources)
            raise QualityKnowledgeBaseError(
                "Le dossier actif doit contenir uniquement les huit sources "
                f"cataloguées (inattendus={unexpected}, absents={missing})."
            )

        return tuple(self.catalog.documents)

    def _current_optional(self) -> Any | None:
        try:
            return load_active_knowledge_base(
                self.pointer,
                project_root=self.project_root,
                knowledge_base_root=self.root,
            )
        except ActiveKnowledgeBaseError:
            return None

    def _already_current(self) -> bool:
        current = self._current_optional()

        if current is None or not current.version.startswith("kb_quality_"):
            return False

        manifest_path = current.version_directory / "index_manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("pipeline_version") != QUALITY_INDEX_PIPELINE_VERSION:
            return False

        expected = {entry.document_id: entry.sha256 for entry in self.catalog.documents}
        observed = {
            str(document.get("document_id")): str(document.get("sha256"))
            for document in current.documents
        }
        return expected == observed

    def _version_name(self) -> str:
        base = datetime.now(UTC).strftime("kb_quality_%Y%m%d_%H%M%S")
        candidate = base
        suffix = 1

        while (self.versions / candidate).exists() or (self.versions / f"{candidate}_tmp").exists():
            candidate = f"{base}_{suffix:02d}"
            suffix += 1

        return candidate

    def _pointer_payload(
        self,
        *,
        version: str,
        documents: tuple[QualityDocumentResult, ...],
        build: IndexBuildResult,
        previous_version: str | None,
    ) -> dict[str, Any]:
        version_directory = self.versions / version

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
            "pipeline_version": QUALITY_INDEX_PIPELINE_VERSION,
            "previous_version": previous_version,
        }

    def sync(
        self,
        *,
        dry_run: bool = False,
        rebuild: bool = False,
        verbose: bool = False,
    ) -> QualitySyncResult:
        """Verify, build in a temporary directory, validate, then activate."""

        entries = self._verified_active_documents()
        current = self._current_optional()
        previous_version = current.version if current is not None else None

        if verbose:
            for entry in entries:
                warning = (
                    " [source incomplète déclarée]"
                    if entry.extraction_status.value == "incomplete_source"
                    else ""
                )
                print(
                    f"- {entry.source_filename} | {entry.page_count} pages | SHA vérifié{warning}"
                )

        if dry_run:
            return QualitySyncResult(
                changed=False,
                dry_run=True,
                version=previous_version,
                previous_version=previous_version,
                document_count=len(entries),
                chunk_count=current.chunk_count if current is not None else 0,
                documents=tuple(
                    QualityDocumentResult(
                        filename=entry.source_filename,
                        document_id=entry.document_id,
                        sha256=entry.sha256,
                        page_count=entry.page_count,
                        chunk_count=0,
                    )
                    for entry in entries
                ),
            )

        if not rebuild and self._already_current():
            assert current is not None
            return QualitySyncResult(
                changed=False,
                dry_run=False,
                version=current.version,
                previous_version=current.version,
                document_count=current.document_count,
                chunk_count=current.chunk_count,
                documents=tuple(
                    QualityDocumentResult(
                        filename=str(document["filename"]),
                        document_id=str(document["document_id"]),
                        sha256=str(document["sha256"]),
                        page_count=int(document["page_count"]),
                        chunk_count=int(document["chunk_count"]),
                    )
                    for document in current.documents
                ),
            )

        prepared = []

        try:
            for entry in entries:
                if verbose:
                    print(f"Préparation Docling/chunking : {entry.source_filename}")

                prepared.append(self.corpus_processor.prepare(entry))
        except Exception as error:
            raise QualityKnowledgeBaseError(
                "La préparation structurée a échoué ; l'ancien index reste actif."
            ) from error

        children = [child for document in prepared for child in document.children if child.active]
        parents = [parent for document in prepared for parent in document.parents]
        sections = [section for document in prepared for section in document.sections]
        documents = tuple(
            QualityDocumentResult(
                filename=document.entry.source_filename,
                document_id=document.entry.document_id,
                sha256=document.entry.sha256,
                page_count=document.entry.page_count,
                chunk_count=len(document.children),
            )
            for document in prepared
        )
        version = self._version_name()
        temporary = self.versions / f"{version}_tmp"
        final = self.versions / version
        self.versions.mkdir(parents=True, exist_ok=True)

        try:
            build = self.index_builder.build(
                children=children,
                parents=parents,
                sections=sections,
                documents=list(entries),
                version_directory=temporary,
                previous_version_directory=(
                    current.version_directory if current is not None else None
                ),
            )
            _publish_directory(temporary, final)
            payload = self._pointer_payload(
                version=version,
                documents=documents,
                build=build,
                previous_version=previous_version,
            )
            _atomic_write(
                self.pointer,
                json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        except Exception as error:
            if temporary.exists():
                shutil.rmtree(temporary)

            raise QualityKnowledgeBaseError(
                "La construction qualité a échoué ; l'ancien index reste actif."
            ) from error

        return QualitySyncResult(
            changed=True,
            dry_run=False,
            version=version,
            previous_version=previous_version,
            document_count=len(documents),
            chunk_count=build.chunk_count,
            documents=documents,
            embedded_chunk_count=build.embedded_chunk_count,
            reused_embedding_count=build.reused_embedding_count,
        )

    def status(self) -> dict[str, Any]:
        current = self._current_optional()

        if current is None:
            print("Base documentaire active : aucune")
            return {"current": None}

        print(f"Base documentaire : {current.version}")
        print(f"Pipeline qualité : {current.version.startswith('kb_quality_')}")
        print(f"Documents actifs : {current.document_count}")
        print(f"Chunks actifs : {current.chunk_count}")
        print(f"Chemin : {current.version_directory}")
        return {"current": current}

    def list_active(self) -> tuple[Any, ...]:
        """Print the exact catalogue-backed active production corpus."""

        entries = self._verified_active_documents()
        print(f"Documents actifs : {len(entries)}")

        for entry in entries:
            print(f"- {entry.source_filename} | pages={entry.page_count} | sha256={entry.sha256}")

        return entries
