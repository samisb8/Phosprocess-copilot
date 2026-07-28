"""Tests for safe production knowledge-base synchronization."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from scripts.sync_knowledge_base import build_parser

from phosprocess.knowledge_base.indexing import VersionIndexBuilder
from phosprocess.knowledge_base.manager import (
    KnowledgeBaseManager,
    KnowledgeBaseSyncError,
    document_id_from_filename,
)
from phosprocess.knowledge_base.models import (
    KNOWLEDGE_BASE_PIPELINE_VERSION,
    IndexBuildResult,
    ProcessedDocument,
    chunk_sha256,
    sha256_file,
)
from phosprocess.knowledge_base.runtime import (
    load_active_knowledge_base,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk


def make_pdf(path: Path, text: str) -> None:
    """Create a small valid PDF for filesystem-level tests."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), text)
        document.save(path)


def make_chunk(
    document_id: str,
    filename: str,
    digest: str,
    *,
    number: int = 0,
) -> DocumentChunk:
    """Create one stable fake processed chunk."""

    text = (
        f"Procédé phosphorique {filename} {digest[:16]} "
        "filtration gypse concentration."
    )
    short_digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    embedding_text = f"Document: {filename}\n{text}"
    return DocumentChunk(
        chunk_id=(
            f"{document_id}_{number:06d}_{short_digest}"
        ),
        document_id=document_id,
        source_file=filename,
        chunk_index=number,
        heading_path=["Procédé"],
        source_pages=[1],
        page_start=1,
        page_end=1,
        content_types=["paragraph"],
        text=text,
        embedding_text=embedding_text,
        body_token_count=20,
        token_count=25,
        source_chunk_ids=[],
        postprocessing_actions=[],
    )


class FakeProcessor:
    """Process each changed PDF into one deterministic chunk."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def process(
        self,
        *,
        pdf_path: Path,
        document_id: str,
        document_sha256: str,
        cache_directory: Path,
    ) -> ProcessedDocument:
        self.calls.append((pdf_path.name, document_sha256))
        return ProcessedDocument(
            filename=pdf_path.name,
            document_id=document_id,
            document_sha256=document_sha256,
            page_count=1,
            empty_pages=(),
            chunks=(
                make_chunk(
                    document_id,
                    pdf_path.name,
                    document_sha256,
                ),
            ),
            duplicates_removed=0,
            ingestion_date="2026-07-27T00:00:00+00:00",
            cache_directory=cache_directory,
        )


class FakeIndexBuilder:
    """Write pointer-loadable placeholder artifacts and record rebuilds."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        *,
        records: list[dict[str, Any]],
        version_directory: Path,
        previous_version_directory: Path | None,
    ) -> IndexBuildResult:
        self.calls.append(
            {
                "chunk_ids": [
                    str(record["chunk_id"])
                    for record in records
                ],
                "previous": previous_version_directory,
            }
        )
        dense = version_directory / "dense"
        bm25 = version_directory / "bm25"
        corpus = version_directory / "corpus"

        for directory in (dense, bm25, corpus):
            directory.mkdir(parents=True, exist_ok=True)

        (dense / "index.faiss").write_bytes(b"fake-faiss")
        (dense / "metadata.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {"vector_id": index, **record},
                    ensure_ascii=False,
                )
                for index, record in enumerate(records)
            )
            + "\n",
            encoding="utf-8",
        )
        (bm25 / "metadata.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {"lexical_id": index, **record},
                    ensure_ascii=False,
                )
                for index, record in enumerate(records)
            )
            + "\n",
            encoding="utf-8",
        )

        for path in (
            dense / "manifest.json",
            bm25 / "manifest.json",
            version_directory / "manifest.json",
        ):
            path.write_text("{}", encoding="utf-8")

        document_counts: dict[str, int] = {}

        for record in records:
            document_id = str(record["document_id"])
            document_counts[document_id] = (
                document_counts.get(document_id, 0) + 1
            )

        return IndexBuildResult(
            chunk_count=len(records),
            document_counts=document_counts,
            embedded_chunk_count=len(records),
            reused_embedding_count=0,
            dense_search_ok=True,
            bm25_search_ok=True,
            hybrid_search_ok=True,
        )


def make_manager(
    temporary_path: Path,
    *,
    processor: FakeProcessor | None = None,
    builder: FakeIndexBuilder | None = None,
    failure_injector: Any = None,
) -> tuple[KnowledgeBaseManager, FakeProcessor, FakeIndexBuilder]:
    """Create one isolated manager with no model or evaluation dependency."""

    fake_processor = processor or FakeProcessor()
    fake_builder = builder or FakeIndexBuilder()
    project_root = temporary_path / "project"
    project_root.mkdir(exist_ok=True)
    manager = KnowledgeBaseManager(
        root=project_root / "data" / "knowledge_base",
        project_root=project_root,
        processor=fake_processor,
        index_builder=fake_builder,
        failure_injector=failure_injector,
    )
    manager.paths.create()
    return manager, fake_processor, fake_builder


def test_initial_sync_contains_only_becker_and_report(
    tmp_path: Path,
) -> None:
    manager, processor, builder = make_manager(tmp_path)
    make_pdf(
        manager.paths.pdfs
        / "01_becker_phosphates_and_phosphoric_acid.pdf",
        "Becker",
    )
    make_pdf(
        manager.paths.pdfs
        / "04_rapport_atelier_acide_phosphorique.pdf",
        "Rapport atelier",
    )

    result = manager.sync()
    active = load_active_knowledge_base(
        manager.paths.current_index,
        project_root=manager.project_root,
        knowledge_base_root=manager.paths.root,
    )

    assert result.document_count == 2
    assert result.chunk_count == 2
    assert len(processor.calls) == 2
    assert len(builder.calls) == 1
    assert {
        document["filename"]
        for document in active.documents
    } == {
        "01_becker_phosphates_and_phosphoric_acid.pdf",
        "04_rapport_atelier_acide_phosphorique.pdf",
    }


def test_unchanged_documents_are_not_reprocessed(
    tmp_path: Path,
) -> None:
    manager, processor, builder = make_manager(tmp_path)
    make_pdf(manager.paths.pdfs / "becker.pdf", "Becker")
    first = manager.sync()
    second = manager.sync()

    assert first.changed is True
    assert second.changed is False
    assert len(processor.calls) == 1
    assert len(builder.calls) == 1


def test_new_pdf_uses_previous_version_for_incremental_build(
    tmp_path: Path,
) -> None:
    manager, processor, builder = make_manager(tmp_path)
    make_pdf(manager.paths.pdfs / "becker.pdf", "Becker")
    first = manager.sync()
    make_pdf(manager.paths.pdfs / "nouveau.pdf", "Nouveau procédé")
    second = manager.sync()

    assert second.document_count == 2
    assert len(processor.calls) == 2
    assert builder.calls[1]["previous"] is not None
    assert builder.calls[1]["previous"].name == first.version


def test_duplicate_pdf_is_ignored_without_rebuild(
    tmp_path: Path,
) -> None:
    manager, processor, builder = make_manager(tmp_path)
    original = manager.paths.pdfs / "becker.pdf"
    make_pdf(original, "Becker")
    manager.sync()
    shutil.copyfile(original, manager.paths.pdfs / "copie.pdf")

    result = manager.sync()

    assert result.plan.count("duplicate") == 1
    assert result.changed is False
    assert len(processor.calls) == 1
    assert len(builder.calls) == 1


def test_same_name_with_different_content_is_modified(
    tmp_path: Path,
) -> None:
    manager, processor, builder = make_manager(tmp_path)
    path = manager.paths.pdfs / "becker.pdf"
    make_pdf(path, "Version un")
    manager.sync()
    path.unlink()
    make_pdf(path, "Version deux")

    result = manager.sync()

    assert result.plan.count("modified") == 1
    assert len(processor.calls) == 2
    assert len(builder.calls) == 2
    assert any(name.startswith("becker") for name in result.archived)


def test_renamed_identical_pdf_reuses_processed_chunks(
    tmp_path: Path,
) -> None:
    manager, processor, builder = make_manager(tmp_path)
    original = manager.paths.pdfs / "ancien_nom.pdf"
    make_pdf(original, "Contenu identique")
    manager.sync()
    renamed = manager.paths.pdfs / "nouveau_nom.pdf"
    original.replace(renamed)

    result = manager.sync()

    assert result.document_count == 1
    assert len(processor.calls) == 1
    assert len(builder.calls) == 2
    assert builder.calls[-1]["chunk_ids"][0].startswith(
        "nouveau_nom_"
    )


def test_removed_document_leaves_no_chunk_in_new_version(
    tmp_path: Path,
) -> None:
    manager, _, builder = make_manager(tmp_path)
    becker = manager.paths.pdfs / "becker.pdf"
    report = manager.paths.pdfs / "rapport.pdf"
    make_pdf(becker, "Becker")
    make_pdf(report, "Rapport")
    manager.sync()
    report.unlink()

    result = manager.sync()

    assert result.plan.summary["removed"] == 1
    assert len(builder.calls[-1]["chunk_ids"]) == 1
    assert all(
        "rapport" not in chunk_id
        for chunk_id in builder.calls[-1]["chunk_ids"]
    )
    assert "rapport.pdf" in result.archived


def test_rebuild_creates_new_version_without_reprocessing(
    tmp_path: Path,
) -> None:
    manager, processor, builder = make_manager(tmp_path)
    make_pdf(manager.paths.pdfs / "becker.pdf", "Becker")
    first = manager.sync()
    second = manager.sync(rebuild=True)

    assert second.changed is True
    assert second.version != first.version
    assert len(processor.calls) == 1
    assert len(builder.calls) == 2


def test_invalid_pdf_is_moved_to_rejected(
    tmp_path: Path,
) -> None:
    manager, _, _ = make_manager(tmp_path)
    make_pdf(manager.paths.pdfs / "becker.pdf", "Becker")
    manager.sync()
    invalid = manager.paths.pdfs / "invalide.pdf"
    invalid.write_bytes(b"not a pdf")

    result = manager.sync()

    assert result.plan.count("invalid") == 1
    assert not invalid.exists()
    assert (
        manager.paths.rejected / "invalide.pdf"
    ).is_file()


def test_unusable_encoded_text_layer_is_rejected(
    tmp_path: Path,
) -> None:
    manager, _, _ = make_manager(tmp_path)
    make_pdf(manager.paths.pdfs / "becker.pdf", "Becker")
    encoded = manager.paths.pdfs / "encoded.pdf"

    with pymupdf.open() as document:
        for _page_number in range(12):
            page = document.new_page()
            page.insert_textbox(
                page.rect + (36, 36, -36, -36),
                ("! # $ % & * + - / 1 2 3 4 5\n" * 25),
                fontsize=8,
            )

        document.save(encoded)

    result = manager.sync()

    assert result.plan.count("invalid") == 1
    assert not encoded.exists()
    assert (manager.paths.rejected / "encoded.pdf").is_file()


def test_failure_preserves_previous_pointer_and_removes_temporary_version(
    tmp_path: Path,
) -> None:
    manager, processor, builder = make_manager(tmp_path)
    make_pdf(manager.paths.pdfs / "becker.pdf", "Becker")
    manager.sync()
    original_pointer = manager.paths.current_index.read_bytes()
    make_pdf(manager.paths.pdfs / "nouveau.pdf", "Nouveau")

    def fail(phase: str) -> None:
        if phase == "before_activation":
            raise RuntimeError("failure injection")

    failing = KnowledgeBaseManager(
        root=manager.paths.root,
        project_root=manager.project_root,
        processor=processor,
        index_builder=builder,
        failure_injector=fail,
    )

    with pytest.raises(KnowledgeBaseSyncError):
        failing.sync()

    assert manager.paths.current_index.read_bytes() == original_pointer
    assert not list(manager.paths.versions.glob("*_tmp"))


def test_windows_publish_falls_back_to_verified_directory_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = tmp_path / "kb_tmp"
    final = tmp_path / "kb_final"
    (temporary / "dense").mkdir(parents=True)
    (temporary / "dense" / "index.faiss").write_bytes(b"index")
    (temporary / "manifest.json").write_text("{}", encoding="utf-8")

    def deny_rename(_source: Path, _target: Path) -> None:
        raise PermissionError("WinError 5")

    monkeypatch.setattr(
        "phosprocess.knowledge_base.manager.os.rename",
        deny_rename,
    )
    monkeypatch.setattr(
        "phosprocess.knowledge_base.manager.time.sleep",
        lambda _seconds: None,
    )

    KnowledgeBaseManager._publish_version(temporary, final)

    assert not temporary.exists()
    assert (final / "dense" / "index.faiss").read_bytes() == b"index"
    assert (final / "manifest.json").read_text(encoding="utf-8") == "{}"


def test_dry_run_performs_no_write(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    pdf_directory = project_root / "data" / "knowledge_base" / "pdfs"
    make_pdf(pdf_directory / "becker.pdf", "Becker")
    manager = KnowledgeBaseManager(
        root=pdf_directory.parent,
        project_root=project_root,
        processor=FakeProcessor(),
        index_builder=FakeIndexBuilder(),
    )
    before = {
        path.relative_to(project_root)
        for path in project_root.rglob("*")
    }

    result = manager.sync(dry_run=True)
    after = {
        path.relative_to(project_root)
        for path in project_root.rglob("*")
    }

    assert result.dry_run is True
    assert before == after
    assert not manager.paths.manifest.exists()
    assert not manager.paths.current_index.exists()


def test_list_status_and_cli_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager, _, _ = make_manager(tmp_path)
    make_pdf(manager.paths.pdfs / "becker.pdf", "Becker")
    manager.sync()
    documents = manager.list_active()
    status = manager.status()
    arguments = build_parser().parse_args(
        ["--dry-run", "--verbose"]
    )

    assert len(documents) == 1
    assert status["current"] is not None
    assert arguments.dry_run is True
    assert arguments.verbose is True
    output = capsys.readouterr().out
    assert "Documents actifs : 1" in output
    assert "Base documentaire : kb_" in output


def test_stable_sha_and_chunk_id() -> None:
    first = make_chunk("becker", "becker.pdf", "A" * 64)
    second = make_chunk("becker", "becker.pdf", "A" * 64)

    assert first.chunk_id == second.chunk_id
    assert chunk_sha256(first) == chunk_sha256(second)
    assert document_id_from_filename("Becker test.pdf") == "becker_test"


class TinyEmbedder:
    """Deterministic normalized embeddings for real mini-index tests."""

    calls: list[list[str]] = []

    def __init__(self, config: Any) -> None:
        self.dimension = config.embedding_dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = np.zeros(
            (len(texts), self.dimension),
            dtype=np.float32,
        )

        for index, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vectors[index, seed % self.dimension] = 1.0

        return vectors


def write_embedding_config(path: Path) -> None:
    path.write_text(
        """
model:
  name: fake-bge-m3
  dimension: 8
  device: cpu
  use_fp16: false
  normalize_embeddings: true
  trust_remote_code: false
  cache_dir: null
inference:
  batch_size: 4
  passage_max_length: 128
  query_max_length: 64
data:
  chunks_directory: unused
index:
  type: IndexFlatIP
  metric: inner_product
  output_directory: unused
  index_filename: index.faiss
  metadata_filename: metadata.jsonl
  manifest_filename: manifest.json
pipeline_version: "test"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def make_records(
    specifications: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for document_id, filename in specifications:
        document = ProcessedDocument(
            filename=filename,
            document_id=document_id,
            document_sha256="A" * 64,
            page_count=1,
            empty_pages=(),
            chunks=(make_chunk(document_id, filename, "A" * 64),),
            duplicates_removed=0,
            ingestion_date="2026-07-27T00:00:00+00:00",
            cache_directory=Path("unused"),
        )
        records.extend(document.metadata_records())

    return records


def test_real_faiss_bm25_incremental_embeddings_and_removal(
    tmp_path: Path,
) -> None:
    TinyEmbedder.calls.clear()
    embedding_config = tmp_path / "embeddings.yaml"
    write_embedding_config(embedding_config)
    project_root = Path(__file__).resolve().parents[1]
    builder = VersionIndexBuilder(
        embedding_config_path=embedding_config,
        retrieval_config_path=(
            project_root / "configs" / "retrieval_v2.yaml"
        ),
        embedder_factory=TinyEmbedder,
        runtime_validation=False,
    )
    first_version = tmp_path / "kb_first"
    first_records = make_records([("becker", "becker.pdf")])
    first = builder.build(
        records=first_records,
        version_directory=first_version,
        previous_version_directory=None,
    )
    second_version = tmp_path / "kb_second"
    second_records = make_records(
        [
            ("becker", "becker.pdf"),
            ("nouveau", "nouveau.pdf"),
        ]
    )
    second = builder.build(
        records=second_records,
        version_directory=second_version,
        previous_version_directory=first_version,
    )
    third_version = tmp_path / "kb_third"
    third = builder.build(
        records=first_records,
        version_directory=third_version,
        previous_version_directory=second_version,
    )
    third_metadata = (
        third_version / "dense" / "metadata.jsonl"
    ).read_text(encoding="utf-8")

    assert first.embedded_chunk_count == 1
    assert second.embedded_chunk_count == 1
    assert second.reused_embedding_count == 1
    assert third.embedded_chunk_count == 0
    assert third.reused_embedding_count == 1
    assert "nouveau" not in third_metadata
    assert first.dense_search_ok and first.bm25_search_ok
    assert len(TinyEmbedder.calls) == 2


def test_inference_code_has_no_test_or_gold_dependency() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_paths = (
        project_root
        / "src"
        / "phosprocess"
        / "knowledge_base"
        / "manager.py",
        project_root
        / "src"
        / "phosprocess"
        / "knowledge_base"
        / "indexing.py",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8").casefold()
        for path in source_paths
    )

    for forbidden in (
        "test_pool",
        "gold_evidence",
        "reference_answer",
        "evaluation_test",
    ):
        assert forbidden not in source

    assert KNOWLEDGE_BASE_PIPELINE_VERSION == "1.0.0"


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"phosprocess")

    assert sha256_file(path) == sha256_file(path)
