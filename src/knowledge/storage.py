from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.knowledge.schemas import KnowledgeChunk, KnowledgeDocument, KnowledgeIndexStatus


class KnowledgeBaseStore:
    def __init__(self, db_path: str | Path = "data/knowledge/knowledge.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def upsert_document(
        self,
        *,
        document_id: str,
        filename: str,
        file_type: str,
        file_hash: str,
        source_path: str,
        parsed_path: str,
        status: str,
        chunk_count: int,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        now = _utc_now()
        existing = self.get_document(document_id)
        created_at = existing.created_at if existing else now
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                  document_id, filename, file_type, file_hash, source_path,
                  parsed_path, status, chunk_count, error_message,
                  created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                  filename=excluded.filename,
                  file_type=excluded.file_type,
                  file_hash=excluded.file_hash,
                  source_path=excluded.source_path,
                  parsed_path=excluded.parsed_path,
                  status=excluded.status,
                  chunk_count=excluded.chunk_count,
                  error_message=excluded.error_message,
                  updated_at=excluded.updated_at,
                  metadata_json=excluded.metadata_json
                """,
                (
                    document_id,
                    filename,
                    file_type,
                    file_hash,
                    source_path,
                    parsed_path,
                    status,
                    chunk_count,
                    error_message,
                    created_at,
                    now,
                    _json_dumps(metadata or {}),
                ),
            )
        loaded = self.get_document(document_id)
        if loaded is None:
            raise sqlite3.DatabaseError(f"Failed to load document after upsert: {document_id}")
        return loaded

    def get_document(self, document_id: str) -> KnowledgeDocument | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return _document_from_row(row) if row else None

    def find_document_by_hash(self, file_hash: str) -> KnowledgeDocument | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM documents
                WHERE file_hash = ?
                ORDER BY
                  CASE WHEN status = 'failed' THEN 1 ELSE 0 END,
                  created_at DESC
                LIMIT 1
                """,
                (file_hash,),
            ).fetchone()
        return _document_from_row(row) if row else None

    def list_documents(self) -> list[KnowledgeDocument]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [_document_from_row(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def replace_chunks(self, document_id: str, chunks: list[KnowledgeChunk]) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (
                      chunk_id, document_id, chunk_index, text, page_number,
                      section_title, token_count, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.page_number,
                        chunk.section_title,
                        chunk.token_count,
                        _json_dumps(chunk.metadata),
                        chunk.created_at or now,
                    ),
                )
            conn.execute(
                "UPDATE documents SET chunk_count = ?, updated_at = ? WHERE document_id = ?",
                (len(chunks), now, document_id),
            )

    def load_document_chunks(self, document_id: str) -> list[KnowledgeChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def delete_chunks(self, document_id: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute(
                "UPDATE documents SET chunk_count = 0, updated_at = ? WHERE document_id = ?",
                (now, document_id),
            )

    def load_chunks(self) -> list[KnowledgeChunk]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY document_id, chunk_index").fetchall()
        return [_chunk_from_row(row) for row in rows]

    def save_index_status(self, status: KnowledgeIndexStatus) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_versions (
                  index_id, faiss_path, chunks_path, embedding_provider,
                  embedding_model, chunk_count, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"index_{_utc_now_compact()}",
                    status.faiss_path,
                    status.chunks_path,
                    status.embedding_provider,
                    status.embedding_model,
                    status.chunk_count,
                    status.updated_at or _utc_now(),
                    _json_dumps({"available": status.available, "error": status.error}),
                ),
            )

    def latest_index_status(self) -> KnowledgeIndexStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM index_versions ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        metadata = _json_loads(row["metadata_json"])
        return KnowledgeIndexStatus(
            available=bool(metadata.get("available", True)),
            faiss_path=row["faiss_path"],
            chunks_path=row["chunks_path"],
            chunk_count=int(row["chunk_count"]),
            embedding_provider=row["embedding_provider"],
            embedding_model=row["embedding_model"],
            updated_at=row["created_at"],
            error=str(metadata.get("error", "")),
        )

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                  document_id TEXT PRIMARY KEY,
                  filename TEXT NOT NULL,
                  file_type TEXT NOT NULL,
                  file_hash TEXT NOT NULL,
                  source_path TEXT NOT NULL,
                  parsed_path TEXT NOT NULL,
                  status TEXT NOT NULL,
                  chunk_count INTEGER NOT NULL DEFAULT 0,
                  error_message TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_documents_file_hash
                  ON documents(file_hash);

                CREATE TABLE IF NOT EXISTS chunks (
                  chunk_id TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL,
                  chunk_index INTEGER NOT NULL,
                  text TEXT NOT NULL,
                  page_number INTEGER,
                  section_title TEXT,
                  token_count INTEGER NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_index
                  ON chunks(document_id, chunk_index);

                CREATE TABLE IF NOT EXISTS index_versions (
                  index_id TEXT PRIMARY KEY,
                  faiss_path TEXT NOT NULL,
                  chunks_path TEXT NOT NULL,
                  embedding_provider TEXT NOT NULL,
                  embedding_model TEXT NOT NULL,
                  chunk_count INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _document_from_row(row: sqlite3.Row) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=row["document_id"],
        filename=row["filename"],
        file_type=row["file_type"],
        file_hash=row["file_hash"],
        source_path=row["source_path"],
        parsed_path=row["parsed_path"],
        status=row["status"],
        chunk_count=int(row["chunk_count"]),
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=_json_loads(row["metadata_json"]),
    )


def _chunk_from_row(row: sqlite3.Row) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        chunk_index=int(row["chunk_index"]),
        text=row["text"],
        page_number=row["page_number"],
        section_title=row["section_title"],
        token_count=int(row["token_count"]),
        metadata=_json_loads(row["metadata_json"]),
        created_at=row["created_at"],
    )


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str) -> dict[str, Any]:
    loaded = json.loads(value or "{}")
    return loaded if isinstance(loaded, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
