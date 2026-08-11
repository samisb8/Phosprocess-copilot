"""SQLite history for active documents and knowledge-base sync runs."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return one ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    """One active or historical document version."""

    filename: str
    document_id: str
    sha256: str
    status: str
    active: bool
    page_count: int
    chunk_count: int
    version: str | None
    error: str | None = None


class KnowledgeBaseManifest:
    """Persist synchronization state without mixing it with evaluation data."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def initialize(self) -> None:
        """Create the schema when a mutating command starts."""

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS document_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0
                        CHECK(active IN (0, 1)),
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    version TEXT,
                    first_seen_utc TEXT NOT NULL,
                    last_seen_utc TEXT NOT NULL,
                    error TEXT,
                    UNIQUE(filename, sha256)
                );

                CREATE INDEX IF NOT EXISTS idx_documents_active
                ON document_versions(active, filename);

                CREATE TABLE IF NOT EXISTS sync_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at_utc TEXT NOT NULL,
                    finished_at_utc TEXT,
                    status TEXT NOT NULL,
                    previous_version TEXT,
                    requested_version TEXT,
                    activated_version TEXT,
                    rebuild INTEGER NOT NULL DEFAULT 0,
                    dry_run INTEGER NOT NULL DEFAULT 0,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def active_documents(self) -> dict[str, ManifestDocument]:
        """Return active documents keyed by filename without creating the DB."""

        if not self.path.is_file():
            return {}

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT filename, document_id, sha256, status, active,
                       page_count, chunk_count, version, error
                FROM document_versions
                WHERE active = 1
                ORDER BY filename
                """
            ).fetchall()

        return {
            str(row["filename"]): ManifestDocument(
                filename=str(row["filename"]),
                document_id=str(row["document_id"]),
                sha256=str(row["sha256"]),
                status=str(row["status"]),
                active=bool(row["active"]),
                page_count=int(row["page_count"]),
                chunk_count=int(row["chunk_count"]),
                version=(str(row["version"]) if row["version"] is not None else None),
                error=(str(row["error"]) if row["error"] is not None else None),
            )
            for row in rows
        }

    def start_run(
        self,
        run_id: str,
        *,
        previous_version: str | None,
        requested_version: str | None,
        rebuild: bool,
        dry_run: bool,
        summary: Mapping[str, Any],
    ) -> None:
        """Record a mutating synchronization attempt."""

        self.initialize()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sync_runs (
                    run_id, started_at_utc, status, previous_version,
                    requested_version, rebuild, dry_run, summary_json
                ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    utc_now(),
                    previous_version,
                    requested_version,
                    int(rebuild),
                    int(dry_run),
                    json.dumps(dict(summary), ensure_ascii=False),
                ),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        activated_version: str | None = None,
        error: str | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        """Finish an existing run with diagnostics."""

        if not self.path.is_file():
            return

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sync_runs
                SET finished_at_utc = ?, status = ?,
                    activated_version = ?, error = ?,
                    summary_json = COALESCE(?, summary_json)
                WHERE run_id = ?
                """,
                (
                    utc_now(),
                    status,
                    activated_version,
                    error,
                    (
                        json.dumps(dict(summary), ensure_ascii=False)
                        if summary is not None
                        else None
                    ),
                    run_id,
                ),
            )

    def record_observations(
        self,
        observations: Iterable[ManifestDocument],
    ) -> None:
        """Store rejected or duplicate files without changing active rows."""

        now = utc_now()

        with self._connect() as connection:
            for document in observations:
                connection.execute(
                    """
                    INSERT INTO document_versions (
                        filename, document_id, sha256, status, active,
                        page_count, chunk_count, version,
                        first_seen_utc, last_seen_utc, error
                    ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(filename, sha256) DO UPDATE SET
                        status = excluded.status,
                        active = 0,
                        last_seen_utc = excluded.last_seen_utc,
                        error = excluded.error
                    """,
                    (
                        document.filename,
                        document.document_id,
                        document.sha256,
                        document.status,
                        document.page_count,
                        document.chunk_count,
                        document.version,
                        now,
                        now,
                        document.error,
                    ),
                )

    def activate_documents(
        self,
        documents: Iterable[ManifestDocument],
        *,
        version: str,
        retired_statuses: Mapping[tuple[str, str], str],
    ) -> None:
        """Atomically switch the manifest's active document set."""

        now = utc_now()

        with self._connect() as connection:
            connection.execute("UPDATE document_versions SET active = 0 WHERE active = 1")

            for (filename, sha256), status in retired_statuses.items():
                connection.execute(
                    """
                    UPDATE document_versions
                    SET status = ?, active = 0, last_seen_utc = ?
                    WHERE filename = ? AND sha256 = ?
                    """,
                    (status, now, filename, sha256),
                )

            for document in documents:
                connection.execute(
                    """
                    INSERT INTO document_versions (
                        filename, document_id, sha256, status, active,
                        page_count, chunk_count, version,
                        first_seen_utc, last_seen_utc, error
                    ) VALUES (?, ?, ?, 'active', 1, ?, ?, ?, ?, ?, NULL)
                    ON CONFLICT(filename, sha256) DO UPDATE SET
                        document_id = excluded.document_id,
                        status = 'active',
                        active = 1,
                        page_count = excluded.page_count,
                        chunk_count = excluded.chunk_count,
                        version = excluded.version,
                        last_seen_utc = excluded.last_seen_utc,
                        error = NULL
                    """,
                    (
                        document.filename,
                        document.document_id,
                        document.sha256,
                        document.page_count,
                        document.chunk_count,
                        version,
                        now,
                        now,
                    ),
                )

    def run_status(self) -> dict[str, Any] | None:
        """Return the latest synchronization run."""

        if not self.path.is_file():
            return None

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, started_at_utc, finished_at_utc, status,
                       previous_version, requested_version,
                       activated_version, rebuild, dry_run,
                       summary_json, error
                FROM sync_runs
                ORDER BY started_at_utc DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["summary"] = json.loads(result.pop("summary_json"))
        result["rebuild"] = bool(result["rebuild"])
        result["dry_run"] = bool(result["dry_run"])
        return result
