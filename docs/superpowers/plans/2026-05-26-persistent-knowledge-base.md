# Persistent Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PDF/DOCX/TXT/Markdown upload, parsing, chunking, SQLite metadata persistence, persistent FAISS indexing, and RAG integration for uploaded operational documents.

**Architecture:** Add a focused `src/knowledge/` package for dynamic knowledge ingestion. SQLite stores document and chunk metadata as the source of truth; FAISS stores dense vectors with a `chunks.jsonl` sidecar. The API exposes upload/list/status/reindex/delete endpoints and refreshes the orchestrator RAG pipeline after index changes.

**Tech Stack:** Python 3.12, FastAPI `UploadFile`, SQLite, FAISS, sentence-transformers/BGE, deterministic hash embeddings for tests, pypdf, python-docx, Streamlit.

---

## File Structure

- Create: `src/knowledge/__init__.py`
  - Package marker and public exports.
- Create: `src/knowledge/schemas.py`
  - Dataclasses/Pydantic-compatible records for parsed documents, stored documents, chunks, and index status.
- Create: `src/knowledge/storage.py`
  - SQLite schema initialization and document/chunk/index metadata operations.
- Create: `src/knowledge/parsers.py`
  - TXT/Markdown/PDF/DOCX text extraction with dependency-aware errors.
- Create: `src/knowledge/chunking.py`
  - Metadata-preserving word chunking for parsed pages/sections.
- Create: `src/knowledge/indexer.py`
  - Full FAISS rebuild, atomic file replacement, sidecar writing, and index status creation.
- Create: `src/knowledge/retriever.py`
  - Persistent FAISS retriever that returns existing RAG context shape.
- Create: `src/knowledge/service.py`
  - Upload, parse, chunk, persist, rebuild, delete, list, and status orchestration.
- Modify: `pyproject.toml`
  - Add parser/upload dependencies and ensure dense extras remain available.
- Modify: `Dockerfile`
  - Install `.[dev,dense]` so FAISS + sentence-transformers work in the demo container.
- Modify: `src/api/schemas.py`
  - Add response models for knowledge endpoints.
- Modify: `src/api/demo_factory.py`
  - Build a persistent FAISS RAG pipeline when index files exist, with static documents as fallback.
- Modify: `src/api/app.py`
  - Add knowledge endpoints and refresh orchestrator instances after reindex.
- Modify: `app/api_client.py`
  - Add client helpers for knowledge endpoints.
- Modify: `app/streamlit_app.py`
  - Add a `Knowledge Base` tab for upload/status/list/reindex.
- Modify: `README.md`
  - Document the new upload workflow, storage layout, and dense dependency.
- Test: `tests/test_knowledge_parsers.py`
- Test: `tests/test_knowledge_storage.py`
- Test: `tests/test_knowledge_chunking.py`
- Test: `tests/test_knowledge_indexer.py`
- Test: `tests/test_knowledge_api.py`
- Test: `tests/test_api_app.py`
- Test: `tests/test_streamlit_client.py`

---

### Task 1: Dependencies And Package Skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `Dockerfile`
- Create: `src/knowledge/__init__.py`

- [ ] **Step 1: Write the failing dependency expectation test**

Create `tests/test_knowledge_parsers.py` with this initial import test:

```python
from __future__ import annotations


def test_knowledge_package_imports():
    import src.knowledge as knowledge

    assert knowledge.__all__ == [
        "KnowledgeBaseService",
        "KnowledgeBaseStore",
        "PersistentKnowledgeRetriever",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_parsers.py::test_knowledge_package_imports -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.knowledge'`.

- [ ] **Step 3: Add dependencies and package exports**

Modify `pyproject.toml` dependencies:

```toml
dependencies = [
    "fastapi>=0.110",
    "langgraph>=1.2",
    "matplotlib>=3.8",
    "numpy>=1.26",
    "pandas>=2.1",
    "pydantic>=2.5",
    "python-docx>=1.1",
    "python-multipart>=0.0.9",
    "pypdf>=4.2",
    "streamlit>=1.37",
    "uvicorn>=0.27",
]
```

Modify `Dockerfile` install line:

```dockerfile
RUN pip install --no-cache-dir -e ".[dev,dense]"
```

Create `src/knowledge/__init__.py`:

```python
from __future__ import annotations

from src.knowledge.retriever import PersistentKnowledgeRetriever
from src.knowledge.service import KnowledgeBaseService
from src.knowledge.storage import KnowledgeBaseStore

__all__ = [
    "KnowledgeBaseService",
    "KnowledgeBaseStore",
    "PersistentKnowledgeRetriever",
]
```

Create temporary empty class shells so imports resolve:

```python
# src/knowledge/storage.py
from __future__ import annotations


class KnowledgeBaseStore:
    pass
```

```python
# src/knowledge/retriever.py
from __future__ import annotations


class PersistentKnowledgeRetriever:
    pass
```

```python
# src/knowledge/service.py
from __future__ import annotations


class KnowledgeBaseService:
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_parsers.py::test_knowledge_package_imports -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Dockerfile src/knowledge tests/test_knowledge_parsers.py
git commit -m "feat: add knowledge base package skeleton"
```

---

### Task 2: Knowledge Schemas

**Files:**
- Create: `src/knowledge/schemas.py`
- Test: `tests/test_knowledge_storage.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_knowledge_storage.py`:

```python
from __future__ import annotations

from src.knowledge.schemas import KnowledgeChunk, KnowledgeDocument, ParsedPage


def test_knowledge_document_serializes_to_dict():
    document = KnowledgeDocument(
        document_id="doc_abc",
        filename="manual.pdf",
        file_type=".pdf",
        file_hash="hash123",
        source_path="data/knowledge/uploads/doc_abc_manual.pdf",
        parsed_path="data/knowledge/parsed/doc_abc.json",
        status="indexed",
        chunk_count=3,
        error_message="",
        created_at="2026-05-26T00:00:00+00:00",
        updated_at="2026-05-26T00:00:00+00:00",
        metadata={"uploaded_by": "operator"},
    )

    assert document.to_dict()["document_id"] == "doc_abc"
    assert document.to_dict()["metadata"]["uploaded_by"] == "operator"


def test_knowledge_chunk_serializes_with_citation_metadata():
    chunk = KnowledgeChunk(
        chunk_id="doc_abc::chunk_0000",
        document_id="doc_abc",
        chunk_index=0,
        text="Alarm handling procedure",
        page_number=12,
        section_title="Alarm Handling",
        token_count=3,
        metadata={"filename": "manual.pdf"},
        created_at="2026-05-26T00:00:00+00:00",
    )

    assert chunk.to_citation()["source_id"] == "doc_abc"
    assert chunk.to_citation()["page_number"] == 12
    assert chunk.to_citation()["section"] == "Alarm Handling"


def test_parsed_page_rejects_empty_text():
    page = ParsedPage(page_number=1, text="   ", section_title=None)

    assert page.normalized_text() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_storage.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.knowledge.schemas'`.

- [ ] **Step 3: Implement schemas**

Create `src/knowledge/schemas.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedPage:
    page_number: int | None
    text: str
    section_title: str | None = None

    def normalized_text(self) -> str:
        return " ".join(self.text.split())


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    filename: str
    file_type: str
    file_hash: str
    source_path: str
    pages: list[ParsedPage]
    metadata: dict[str, Any] = field(default_factory=dict)

    def text_length(self) -> int:
        return sum(len(page.normalized_text()) for page in self.pages)


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    filename: str
    file_type: str
    file_hash: str
    source_path: str
    parsed_path: str
    status: str
    chunk_count: int
    error_message: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_hash": self.file_hash,
            "source_path": self.source_path,
            "parsed_path": self.parsed_path,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int | None
    section_title: str | None
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "token_count": self.token_count,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    def to_citation(self) -> dict[str, Any]:
        return {
            "source_id": self.document_id,
            "title": self.metadata.get("filename", self.document_id),
            "source_path": self.metadata.get("source_path", ""),
            "page_number": self.page_number,
            "section": self.section_title,
        }


@dataclass(frozen=True)
class KnowledgeIndexStatus:
    available: bool
    faiss_path: str
    chunks_path: str
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    updated_at: str | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "faiss_path": self.faiss_path,
            "chunks_path": self.chunks_path,
            "chunk_count": self.chunk_count,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "updated_at": self.updated_at,
            "error": self.error,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_storage.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/schemas.py tests/test_knowledge_storage.py
git commit -m "feat: add knowledge base schemas"
```

---

### Task 3: SQLite Knowledge Store

**Files:**
- Modify: `src/knowledge/storage.py`
- Test: `tests/test_knowledge_storage.py`

- [ ] **Step 1: Add failing storage tests**

Append to `tests/test_knowledge_storage.py`:

```python
from pathlib import Path

from src.knowledge.storage import KnowledgeBaseStore


def test_store_saves_and_loads_document_and_chunks(tmp_path: Path):
    store = KnowledgeBaseStore(tmp_path / "knowledge.db")
    document = store.upsert_document(
        document_id="doc_1",
        filename="ops.md",
        file_type=".md",
        file_hash="hash1",
        source_path=str(tmp_path / "ops.md"),
        parsed_path=str(tmp_path / "doc_1.json"),
        status="parsed",
        chunk_count=0,
        error_message="",
        metadata={"category": "sop"},
    )

    store.replace_chunks(
        "doc_1",
        [
            KnowledgeChunk(
                chunk_id="doc_1::chunk_0000",
                document_id="doc_1",
                chunk_index=0,
                text="Cooling alarm SOP",
                page_number=None,
                section_title="SOP",
                token_count=3,
                metadata={"filename": "ops.md"},
            )
        ],
    )

    loaded = store.get_document("doc_1")
    chunks = store.load_chunks()

    assert loaded is not None
    assert document.document_id == "doc_1"
    assert loaded.metadata["category"] == "sop"
    assert chunks[0].text == "Cooling alarm SOP"


def test_store_deduplicates_by_file_hash(tmp_path: Path):
    store = KnowledgeBaseStore(tmp_path / "knowledge.db")
    store.upsert_document(
        document_id="doc_1",
        filename="ops.md",
        file_type=".md",
        file_hash="same",
        source_path="ops.md",
        parsed_path="doc_1.json",
        status="indexed",
        chunk_count=1,
        error_message="",
    )

    assert store.find_document_by_hash("same").document_id == "doc_1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_storage.py::test_store_saves_and_loads_document_and_chunks -q`

Expected: FAIL with `AttributeError: 'KnowledgeBaseStore' object has no attribute 'upsert_document'`.

- [ ] **Step 3: Implement SQLite store**

Replace `src/knowledge/storage.py` with a real implementation:

```python
from __future__ import annotations

import json
import sqlite3
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
        return self.get_document(document_id)  # type: ignore[return-value]

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
                "SELECT * FROM documents WHERE file_hash = ? ORDER BY created_at LIMIT 1",
                (file_hash,),
            ).fetchone()
        return _document_from_row(row) if row else None

    def list_documents(self) -> list[KnowledgeDocument]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
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

    def load_chunks(self) -> list[KnowledgeChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks ORDER BY document_id, chunk_index"
            ).fetchall()
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
                  FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
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
        return conn


def _document_from_row(row: sqlite3.Row) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=str(row["document_id"]),
        filename=str(row["filename"]),
        file_type=str(row["file_type"]),
        file_hash=str(row["file_hash"]),
        source_path=str(row["source_path"]),
        parsed_path=str(row["parsed_path"]),
        status=str(row["status"]),
        chunk_count=int(row["chunk_count"]),
        error_message=str(row["error_message"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _chunk_from_row(row: sqlite3.Row) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        page_number=row["page_number"],
        section_title=row["section_title"],
        token_count=int(row["token_count"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
        created_at=str(row["created_at"]),
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact() -> str:
    return _utc_now().replace(":", "").replace("-", "").replace(".", "")
```

- [ ] **Step 4: Run storage tests**

Run: `pytest tests/test_knowledge_storage.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/storage.py tests/test_knowledge_storage.py
git commit -m "feat: persist knowledge document metadata"
```

---

### Task 4: Document Parsers

**Files:**
- Create: `src/knowledge/parsers.py`
- Test: `tests/test_knowledge_parsers.py`

- [ ] **Step 1: Add failing parser tests**

Append to `tests/test_knowledge_parsers.py`:

```python
from pathlib import Path

import pytest

from src.knowledge.parsers import UnsupportedDocumentTypeError, parse_document


def test_parse_markdown_document_preserves_heading(tmp_path: Path):
    path = tmp_path / "ops.md"
    path.write_text("# Cooling SOP\n\nCheck rack delta T before reset.", encoding="utf-8")

    parsed = parse_document(path, document_id="doc_md")

    assert parsed.filename == "ops.md"
    assert parsed.file_type == ".md"
    assert parsed.pages[0].section_title == "Cooling SOP"
    assert "rack delta T" in parsed.pages[0].text


def test_parse_txt_document(tmp_path: Path):
    path = tmp_path / "manual.txt"
    path.write_text("Alarm response procedure", encoding="utf-8")

    parsed = parse_document(path, document_id="doc_txt")

    assert parsed.pages[0].page_number is None
    assert parsed.pages[0].text == "Alarm response procedure"


def test_parse_rejects_unsupported_file_type(tmp_path: Path):
    path = tmp_path / "manual.xlsx"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match=".xlsx"):
        parse_document(path, document_id="doc_bad")
```

- [ ] **Step 2: Run parser tests to verify failure**

Run: `pytest tests/test_knowledge_parsers.py::test_parse_markdown_document_preserves_heading -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.knowledge.parsers'`.

- [ ] **Step 3: Implement parsers**

Create `src/knowledge/parsers.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from src.knowledge.schemas import ParsedDocument, ParsedPage

SUPPORTED_DOCUMENT_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}


class UnsupportedDocumentTypeError(ValueError):
    pass


class DocumentParseError(RuntimeError):
    pass


def parse_document(path: str | Path, *, document_id: str) -> ParsedDocument:
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))
        raise UnsupportedDocumentTypeError(f"Unsupported document type '{suffix}'. Supported: {supported}")

    if suffix in {".md", ".txt"}:
        pages = [_parse_text_page(document_path)]
    elif suffix == ".pdf":
        pages = _parse_pdf_pages(document_path)
    else:
        pages = _parse_docx_pages(document_path)

    if not any(page.normalized_text() for page in pages):
        raise DocumentParseError(f"Parsed document '{document_path.name}' did not contain extractable text.")

    return ParsedDocument(
        document_id=document_id,
        filename=document_path.name,
        file_type=suffix,
        file_hash=file_sha256(document_path),
        source_path=str(document_path),
        pages=pages,
        metadata={"filename": document_path.name, "source_path": str(document_path)},
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_text_page(path: Path) -> ParsedPage:
    text = path.read_text(encoding="utf-8")
    return ParsedPage(
        page_number=None,
        text=text,
        section_title=_first_heading(text),
    )


def _parse_pdf_pages(path: Path) -> list[ParsedPage]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise DocumentParseError("PDF parsing requires pypdf. Install project dependencies.") from exc

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(ParsedPage(page_number=index, text=text, section_title=None))
    return pages


def _parse_docx_pages(path: Path) -> list[ParsedPage]:
    try:
        from docx import Document
    except Exception as exc:
        raise DocumentParseError("DOCX parsing requires python-docx. Install project dependencies.") from exc

    document = Document(str(path))
    lines = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    text = "\n".join(lines)
    return [ParsedPage(page_number=None, text=text, section_title=_first_heading(text))]


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        if stripped and len(stripped.split()) <= 12:
            return stripped
    return None
```

- [ ] **Step 4: Run parser tests**

Run: `pytest tests/test_knowledge_parsers.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/parsers.py tests/test_knowledge_parsers.py
git commit -m "feat: parse uploaded knowledge documents"
```

---

### Task 5: Metadata-Preserving Chunking

**Files:**
- Create: `src/knowledge/chunking.py`
- Test: `tests/test_knowledge_chunking.py`

- [ ] **Step 1: Write failing chunking tests**

Create `tests/test_knowledge_chunking.py`:

```python
from __future__ import annotations

from src.knowledge.chunking import chunk_parsed_document
from src.knowledge.schemas import ParsedDocument, ParsedPage


def test_chunk_parsed_document_preserves_page_and_source_metadata():
    parsed = ParsedDocument(
        document_id="doc_1",
        filename="manual.pdf",
        file_type=".pdf",
        file_hash="hash1",
        source_path="manual.pdf",
        pages=[
            ParsedPage(
                page_number=2,
                section_title="Alarm Handling",
                text="one two three four five six seven eight nine ten",
            )
        ],
        metadata={"filename": "manual.pdf", "source_path": "manual.pdf"},
    )

    chunks = chunk_parsed_document(parsed, chunk_size_words=5, overlap_words=1)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert chunks[0].chunk_id == "doc_1::chunk_0000"
    assert chunks[0].page_number == 2
    assert chunks[0].section_title == "Alarm Handling"
    assert chunks[0].metadata["filename"] == "manual.pdf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_chunking.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.knowledge.chunking'`.

- [ ] **Step 3: Implement chunking**

Create `src/knowledge/chunking.py`:

```python
from __future__ import annotations

from src.knowledge.schemas import KnowledgeChunk, ParsedDocument


def chunk_parsed_document(
    document: ParsedDocument,
    *,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> list[KnowledgeChunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive.")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative.")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    chunks: list[KnowledgeChunk] = []
    for page in document.pages:
        words = page.normalized_text().split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            text = " ".join(words[start:end])
            chunk_index = len(chunks)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.document_id}::chunk_{chunk_index:04d}",
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=text,
                    page_number=page.page_number,
                    section_title=page.section_title,
                    token_count=len(text.split()),
                    metadata={
                        **document.metadata,
                        "filename": document.filename,
                        "source_path": document.source_path,
                        "file_type": document.file_type,
                        "file_hash": document.file_hash,
                    },
                )
            )
            if end == len(words):
                break
            start = end - overlap_words
    return chunks
```

- [ ] **Step 4: Run chunking tests**

Run: `pytest tests/test_knowledge_chunking.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/chunking.py tests/test_knowledge_chunking.py
git commit -m "feat: chunk parsed knowledge documents"
```

---

### Task 6: Persistent FAISS Indexer And Retriever

**Files:**
- Modify: `src/knowledge/indexer.py`
- Modify: `src/knowledge/retriever.py`
- Test: `tests/test_knowledge_indexer.py`

- [ ] **Step 1: Write failing FAISS persistence tests**

Create `tests/test_knowledge_indexer.py`:

```python
from __future__ import annotations

from pathlib import Path

from src.knowledge.indexer import KnowledgeFaissIndexer
from src.knowledge.retriever import PersistentKnowledgeRetriever
from src.knowledge.schemas import KnowledgeChunk
from src.retrieval.embeddings import DeterministicHashEmbeddingProvider


def _chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    document_id = chunk_id.split("::")[0]
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        page_number=1,
        section_title="Ops",
        token_count=len(text.split()),
        metadata={"filename": f"{document_id}.md", "source_path": f"{document_id}.md"},
    )


def test_indexer_rebuilds_and_retriever_loads_persistent_index(tmp_path: Path):
    indexer = KnowledgeFaissIndexer(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    chunks = [
        _chunk("doc_a::chunk_0000", "cooling alarm response"),
        _chunk("doc_b::chunk_0000", "employee access policy"),
    ]

    status = indexer.rebuild(chunks)
    retriever = PersistentKnowledgeRetriever(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    results = retriever.search("cooling alarm", top_k=1)

    assert status.available is True
    assert status.chunk_count == 2
    assert results[0]["chunk_id"] == "doc_a::chunk_0000"
    assert results[0]["citation"]["title"] == "doc_a.md"
    assert results[0]["retrieval_mode"] == "persistent_faiss"


def test_indexer_keeps_existing_index_when_rebuild_fails(tmp_path: Path, monkeypatch):
    indexer = KnowledgeFaissIndexer(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    indexer.rebuild([_chunk("doc_a::chunk_0000", "cooling alarm response")])
    original_chunks = (tmp_path / "faiss" / "chunks.jsonl").read_text(encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise RuntimeError("replace failed")

    monkeypatch.setattr("src.knowledge.indexer._atomic_replace", fail_replace)

    try:
        indexer.rebuild([_chunk("doc_b::chunk_0000", "new text")])
    except RuntimeError:
        pass

    assert (tmp_path / "faiss" / "chunks.jsonl").read_text(encoding="utf-8") == original_chunks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_indexer.py::test_indexer_rebuilds_and_retriever_loads_persistent_index -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.knowledge.indexer'`.

- [ ] **Step 3: Implement indexer**

Create `src/knowledge/indexer.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.knowledge.schemas import KnowledgeChunk, KnowledgeIndexStatus
from src.retrieval.embeddings import EmbeddingProvider


class KnowledgeFaissIndexer:
    def __init__(
        self,
        *,
        index_dir: str | Path = "data/knowledge/faiss",
        embedding_provider: EmbeddingProvider,
        embedding_provider_name: str,
        embedding_model: str,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.embedding_provider = embedding_provider
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model = embedding_model

    def rebuild(self, chunks: list[KnowledgeChunk]) -> KnowledgeIndexStatus:
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp_index = self.index_dir / "index.faiss.tmp"
        tmp_chunks = self.index_dir / "chunks.jsonl.tmp"

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_provider.embed_texts(texts) if texts else []
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.size == 0:
            matrix = np.zeros((0, 1), dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        index = faiss.IndexFlatIP(matrix.shape[1])
        if len(chunks) > 0:
            index.add(matrix)
        faiss.write_index(index, str(tmp_index))
        _write_chunks_sidecar(tmp_chunks, chunks)

        _atomic_replace(tmp_index, self.index_path)
        _atomic_replace(tmp_chunks, self.chunks_path)

        return KnowledgeIndexStatus(
            available=True,
            faiss_path=str(self.index_path),
            chunks_path=str(self.chunks_path),
            chunk_count=len(chunks),
            embedding_provider=self.embedding_provider_name,
            embedding_model=self.embedding_model,
            updated_at=_utc_now(),
        )


def load_sidecar_chunks(path: str | Path) -> list[dict[str, Any]]:
    sidecar = Path(path)
    if not sidecar.exists():
        return []
    records = []
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_chunks_sidecar(path: Path, chunks: list[KnowledgeChunk]) -> None:
    lines = [json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) for chunk in chunks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _atomic_replace(src: Path, dst: Path) -> None:
    src.replace(dst)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
```

- [ ] **Step 4: Implement persistent retriever**

Replace `src/knowledge/retriever.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.knowledge.indexer import load_sidecar_chunks
from src.knowledge.schemas import KnowledgeChunk, KnowledgeIndexStatus
from src.retrieval.embeddings import EmbeddingProvider


class PersistentKnowledgeRetriever:
    def __init__(
        self,
        *,
        index_dir: str | Path = "data/knowledge/faiss",
        embedding_provider: EmbeddingProvider,
        embedding_provider_name: str,
        embedding_model: str,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.embedding_provider = embedding_provider
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model = embedding_model
        self._index = None
        self._chunks = []
        self.status = self._load()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self._index is None or not self._chunks or not query.strip():
            return []
        query_vector = np.asarray(self.embedding_provider.embed_texts([query]), dtype="float32")
        scores, indices = self._index.search(query_vector, min(top_k, len(self._chunks)))
        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self._chunks[int(index)]
            citation = {
                "source_id": chunk["document_id"],
                "title": chunk.get("metadata", {}).get("filename", chunk["document_id"]),
                "source_path": chunk.get("metadata", {}).get("source_path", ""),
                "page_number": chunk.get("page_number"),
                "section": chunk.get("section_title"),
            }
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "text": chunk["text"],
                    "citation": citation,
                    "retrieval_mode": "persistent_faiss",
                    "metadata": chunk.get("metadata", {}),
                }
            )
        return results

    def _load(self) -> dict[str, Any]:
        if not self.index_path.exists() or not self.chunks_path.exists():
            return KnowledgeIndexStatus(
                available=False,
                faiss_path=str(self.index_path),
                chunks_path=str(self.chunks_path),
                chunk_count=0,
                embedding_provider=self.embedding_provider_name,
                embedding_model=self.embedding_model,
                error="persistent FAISS index not found",
            ).to_dict()
        try:
            import faiss

            self._index = faiss.read_index(str(self.index_path))
            self._chunks = load_sidecar_chunks(self.chunks_path)
            return KnowledgeIndexStatus(
                available=True,
                faiss_path=str(self.index_path),
                chunks_path=str(self.chunks_path),
                chunk_count=len(self._chunks),
                embedding_provider=self.embedding_provider_name,
                embedding_model=self.embedding_model,
            ).to_dict()
        except Exception as exc:
            self._index = None
            self._chunks = []
            return KnowledgeIndexStatus(
                available=False,
                faiss_path=str(self.index_path),
                chunks_path=str(self.chunks_path),
                chunk_count=0,
                embedding_provider=self.embedding_provider_name,
                embedding_model=self.embedding_model,
                error=str(exc),
            ).to_dict()
```

- [ ] **Step 5: Run indexer tests**

Run: `pytest tests/test_knowledge_indexer.py -q`

Expected: PASS when `faiss-cpu` is installed. If the environment lacks FAISS, install with `pip install -e ".[dense]"` and rerun.

- [ ] **Step 6: Commit**

```bash
git add src/knowledge/indexer.py src/knowledge/retriever.py tests/test_knowledge_indexer.py
git commit -m "feat: persist knowledge FAISS index"
```

---

### Task 7: Knowledge Base Service

**Files:**
- Modify: `src/knowledge/service.py`
- Test: `tests/test_knowledge_indexer.py`

- [ ] **Step 1: Add failing service integration test**

Append to `tests/test_knowledge_indexer.py`:

```python
from src.knowledge.service import KnowledgeBaseService


def test_service_ingests_document_and_rebuilds_index(tmp_path: Path):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    result = service.ingest_existing_file(source)
    results = service.retriever().search("cooling alarm", top_k=1)

    assert result["document"]["filename"] == "cooling.md"
    assert result["index_status"]["available"] is True
    assert results[0]["citation"]["title"] == "cooling.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_indexer.py::test_service_ingests_document_and_rebuilds_index -q`

Expected: FAIL with `AttributeError: 'KnowledgeBaseService' object has no attribute 'ingest_existing_file'`.

- [ ] **Step 3: Implement service**

Replace `src/knowledge/service.py`:

```python
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from src.knowledge.chunking import chunk_parsed_document
from src.knowledge.indexer import KnowledgeFaissIndexer
from src.knowledge.parsers import file_sha256, parse_document
from src.knowledge.retriever import PersistentKnowledgeRetriever
from src.knowledge.schemas import KnowledgeDocument
from src.knowledge.storage import KnowledgeBaseStore
from src.retrieval.embeddings import (
    DeterministicHashEmbeddingProvider,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


class KnowledgeBaseService:
    def __init__(
        self,
        *,
        knowledge_dir: str | Path = "data/knowledge",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_provider_name: str = "sentence-transformers",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.uploads_dir = self.knowledge_dir / "uploads"
        self.parsed_dir = self.knowledge_dir / "parsed"
        self.index_dir = self.knowledge_dir / "faiss"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.store = KnowledgeBaseStore(self.knowledge_dir / "knowledge.db")
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider or _build_embedding_provider(
            embedding_provider_name,
            embedding_model,
        )

    def ingest_existing_file(self, source_path: str | Path) -> dict[str, Any]:
        source = Path(source_path)
        file_hash = file_sha256(source)
        existing = self.store.find_document_by_hash(file_hash)
        if existing is not None:
            return {
                "document": existing.to_dict(),
                "deduplicated": True,
                "index_status": self.status()["index"],
            }

        document_id = f"doc_{uuid.uuid4().hex}"
        stored_path = self.uploads_dir / f"{document_id}_{_safe_filename(source.name)}"
        shutil.copyfile(source, stored_path)
        parsed_path = self.parsed_dir / f"{document_id}.json"

        try:
            parsed = parse_document(stored_path, document_id=document_id)
            chunks = chunk_parsed_document(parsed)
            parsed_path.write_text(
                json.dumps(
                    {
                        "document_id": parsed.document_id,
                        "filename": parsed.filename,
                        "file_type": parsed.file_type,
                        "file_hash": parsed.file_hash,
                        "source_path": parsed.source_path,
                        "pages": [
                            {
                                "page_number": page.page_number,
                                "section_title": page.section_title,
                                "text": page.text,
                            }
                            for page in parsed.pages
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            document = self.store.upsert_document(
                document_id=document_id,
                filename=source.name,
                file_type=source.suffix.lower(),
                file_hash=file_hash,
                source_path=str(stored_path),
                parsed_path=str(parsed_path),
                status="indexed",
                chunk_count=len(chunks),
                error_message="",
                metadata={"filename": source.name},
            )
            self.store.replace_chunks(document_id, chunks)
            index_status = self.reindex()
            return {
                "document": document.to_dict(),
                "deduplicated": False,
                "index_status": index_status,
            }
        except Exception as exc:
            document = self.store.upsert_document(
                document_id=document_id,
                filename=source.name,
                file_type=source.suffix.lower(),
                file_hash=file_hash,
                source_path=str(stored_path),
                parsed_path=str(parsed_path),
                status="failed",
                chunk_count=0,
                error_message=str(exc),
                metadata={"filename": source.name},
            )
            return {
                "document": document.to_dict(),
                "deduplicated": False,
                "index_status": self.status()["index"],
            }

    def list_documents(self) -> list[dict[str, Any]]:
        return [document.to_dict() for document in self.store.list_documents()]

    def delete_document(self, document_id: str) -> dict[str, Any]:
        document = self.store.get_document(document_id)
        if document is not None:
            _unlink_if_exists(document.source_path)
            _unlink_if_exists(document.parsed_path)
        self.store.delete_document(document_id)
        return {"deleted": document_id, "index_status": self.reindex()}

    def reindex(self) -> dict[str, Any]:
        chunks = self.store.load_chunks()
        status = KnowledgeFaissIndexer(
            index_dir=self.index_dir,
            embedding_provider=self.embedding_provider,
            embedding_provider_name=self.embedding_provider_name,
            embedding_model=self.embedding_model,
        ).rebuild(chunks)
        self.store.save_index_status(status)
        return status.to_dict()

    def retriever(self) -> PersistentKnowledgeRetriever:
        return PersistentKnowledgeRetriever(
            index_dir=self.index_dir,
            embedding_provider=self.embedding_provider,
            embedding_provider_name=self.embedding_provider_name,
            embedding_model=self.embedding_model,
        )

    def status(self) -> dict[str, Any]:
        retriever = self.retriever()
        return {
            "document_count": len(self.store.list_documents()),
            "chunk_count": len(self.store.load_chunks()),
            "index": retriever.status,
        }


def _build_embedding_provider(provider: str, model: str) -> EmbeddingProvider:
    if provider == "deterministic":
        return DeterministicHashEmbeddingProvider()
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(model)
    raise ValueError(f"Unsupported knowledge embedding provider: {provider}")


def _safe_filename(filename: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in filename)


def _unlink_if_exists(path: str) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
```

- [ ] **Step 4: Run service test**

Run: `pytest tests/test_knowledge_indexer.py::test_service_ingests_document_and_rebuilds_index -q`

Expected: PASS.

- [ ] **Step 5: Run all knowledge tests**

Run: `pytest tests/test_knowledge_*.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/knowledge/service.py tests/test_knowledge_indexer.py
git commit -m "feat: orchestrate knowledge ingestion"
```

---

### Task 8: API Schemas And Knowledge Endpoints

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `src/api/app.py`
- Test: `tests/test_knowledge_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_knowledge_api.py`:

```python
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.retrieval.embeddings import DeterministicHashEmbeddingProvider


def test_upload_document_indexes_and_lists_document(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(use_env_answer_generator=False, use_env_intent_classifier=False, use_dropt_policy=False)
    client = TestClient(app)

    response = client.post(
        "/knowledge/documents/upload",
        files={"file": ("cooling.md", b"# Cooling SOP\n\nCooling alarm response.", "text/markdown")},
    )
    listing = client.get("/knowledge/documents")
    status = client.get("/knowledge/status")

    assert response.status_code == 200
    assert response.json()["document"]["filename"] == "cooling.md"
    assert response.json()["index_status"]["available"] is True
    assert listing.json()["documents"][0]["filename"] == "cooling.md"
    assert status.json()["chunk_count"] >= 1


def test_delete_document_rebuilds_index(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(use_env_answer_generator=False, use_env_intent_classifier=False, use_dropt_policy=False)
    client = TestClient(app)
    uploaded = client.post(
        "/knowledge/documents/upload",
        files={"file": ("ops.txt", b"employee cooling procedure", "text/plain")},
    ).json()

    document_id = uploaded["document"]["document_id"]
    response = client.delete(f"/knowledge/documents/{document_id}")

    assert response.status_code == 200
    assert client.get("/knowledge/documents").json()["documents"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_api.py::test_upload_document_indexes_and_lists_document -q`

Expected: FAIL with 404 for `/knowledge/documents/upload`.

- [ ] **Step 3: Add API response schemas**

Append to `src/api/schemas.py`:

```python
class KnowledgeUploadResponse(BaseModel):
    document: dict
    deduplicated: bool = False
    index_status: dict = Field(default_factory=dict)


class KnowledgeDocumentListResponse(BaseModel):
    documents: list[dict] = Field(default_factory=list)


class KnowledgeStatusResponse(BaseModel):
    document_count: int
    chunk_count: int
    index: dict = Field(default_factory=dict)


class KnowledgeDeleteResponse(BaseModel):
    deleted: str
    index_status: dict = Field(default_factory=dict)
```

- [ ] **Step 4: Add env-driven service and endpoints**

Modify imports in `src/api/app.py`:

```python
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from src.knowledge.service import KnowledgeBaseService
```

Add response models to imports from `src.api.schemas`:

```python
KnowledgeDeleteResponse,
KnowledgeDocumentListResponse,
KnowledgeStatusResponse,
KnowledgeUploadResponse,
```

Inside `create_app`, add a nonlocal service:

```python
    knowledge_service: KnowledgeBaseService | None = None
```

Add helper inside `create_app`:

```python
    def _get_knowledge_service() -> KnowledgeBaseService:
        nonlocal knowledge_service
        if knowledge_service is None:
            knowledge_service = KnowledgeBaseService(
                knowledge_dir=os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge"),
                embedding_provider_name=os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "sentence-transformers"),
                embedding_model=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            )
        return knowledge_service
```

Add endpoints before `return app`:

```python
    @app.post("/knowledge/documents/upload", response_model=KnowledgeUploadResponse)
    def upload_knowledge_document(file: UploadFile = File(...)) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".md", ".txt", ".pdf", ".docx"}:
            raise HTTPException(status_code=400, detail="Supported file types: .md, .txt, .pdf, .docx")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            return _get_knowledge_service().ingest_existing_file(tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    @app.get("/knowledge/documents", response_model=KnowledgeDocumentListResponse)
    def list_knowledge_documents() -> dict:
        return {"documents": _get_knowledge_service().list_documents()}

    @app.get("/knowledge/status", response_model=KnowledgeStatusResponse)
    def knowledge_status() -> dict:
        return _get_knowledge_service().status()

    @app.post("/knowledge/reindex", response_model=KnowledgeStatusResponse)
    def reindex_knowledge() -> dict:
        _get_knowledge_service().reindex()
        return _get_knowledge_service().status()

    @app.delete("/knowledge/documents/{document_id}", response_model=KnowledgeDeleteResponse)
    def delete_knowledge_document(document_id: str) -> dict:
        return _get_knowledge_service().delete_document(document_id)
```

- [ ] **Step 5: Run API tests**

Run: `pytest tests/test_knowledge_api.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/api/app.py src/api/schemas.py tests/test_knowledge_api.py
git commit -m "feat: expose knowledge base API"
```

---

### Task 9: Use Persistent Knowledge In `/ask`

**Files:**
- Modify: `src/api/demo_factory.py`
- Modify: `src/api/app.py`
- Test: `tests/test_knowledge_api.py`

- [ ] **Step 1: Add failing `/ask` integration test**

Append to `tests/test_knowledge_api.py`:

```python
def test_ask_uses_uploaded_knowledge_without_restart(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(use_env_answer_generator=False, use_env_intent_classifier=False, use_dropt_policy=False)
    client = TestClient(app)
    client.post(
        "/knowledge/documents/upload",
        files={"file": ("custom_sop.md", b"# Custom SOP\n\nBlue coolant valve inspection is required.", "text/markdown")},
    )

    response = client.post(
        "/ask",
        json={
            "question": "What does the custom SOP say about blue coolant valve inspection?",
            "task_type": "document_qa",
            "workflow_engine": "deterministic",
            "memory_enabled": False,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert any("Blue coolant valve" in context["text"] for context in body["retrieved_contexts"])
    assert any(citation["title"] == "custom_sop.md" for citation in body["citations"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_api.py::test_ask_uses_uploaded_knowledge_without_restart -q`

Expected: FAIL because `/ask` still uses the orchestrator built before upload or static documents only.

- [ ] **Step 3: Add persistent RAG builder**

Modify `src/api/demo_factory.py` imports:

```python
import os
from src.knowledge.service import KnowledgeBaseService
```

Add function:

```python
def build_rag_pipeline(project_root: Path | None = None) -> ExtractiveRAGPipeline:
    project_root = project_root or Path(__file__).resolve().parents[2]
    knowledge_dir = Path(os.getenv("KNOWLEDGE_BASE_DIR", project_root / "data" / "knowledge"))
    provider = os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "sentence-transformers")
    model = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    service = KnowledgeBaseService(
        knowledge_dir=knowledge_dir,
        embedding_provider_name=provider,
        embedding_model=model,
    )
    retriever = service.retriever()
    if retriever.status.get("available") and retriever.status.get("chunk_count", 0) > 0:
        return ExtractiveRAGPipeline(retriever)
    chunks = []
    for document in _load_demo_documents(project_root):
        chunks.extend(chunk_document(document, chunk_size=45, overlap=5))
    return ExtractiveRAGPipeline(HybridRetriever(chunks))
```

Change `build_demo_orchestrator()`:

```python
    rag = build_rag_pipeline(project_root)
```

- [ ] **Step 4: Refresh orchestrators after upload/reindex/delete**

In `src/api/app.py`, add helper inside `create_app`:

```python
    def _refresh_orchestrators() -> None:
        nonlocal orchestrator, langgraph_orchestrator
        orchestrator = build_demo_orchestrator(
            use_env_answer_generator=use_env_answer_generator,
            use_dropt_policy=use_dropt_policy,
        )
        langgraph_orchestrator = LangGraphOrchestrator(
            orchestrator,
            route_planner=(
                build_route_planner_from_env()
                if use_env_intent_classifier
                else DeterministicRoutePlanner()
            ),
        )
```

Call `_refresh_orchestrators()` after successful upload, reindex, and delete:

```python
            result = _get_knowledge_service().ingest_existing_file(tmp_path)
            _refresh_orchestrators()
            return result
```

```python
        _get_knowledge_service().reindex()
        _refresh_orchestrators()
        return _get_knowledge_service().status()
```

```python
        result = _get_knowledge_service().delete_document(document_id)
        _refresh_orchestrators()
        return result
```

- [ ] **Step 5: Run `/ask` integration test**

Run: `pytest tests/test_knowledge_api.py::test_ask_uses_uploaded_knowledge_without_restart -q`

Expected: PASS.

- [ ] **Step 6: Run API app tests**

Run: `pytest tests/test_knowledge_api.py tests/test_api_app.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/api/demo_factory.py src/api/app.py tests/test_knowledge_api.py
git commit -m "feat: ground ask responses in uploaded knowledge"
```

---

### Task 10: Streamlit API Client And Knowledge Tab

**Files:**
- Modify: `app/api_client.py`
- Modify: `app/streamlit_app.py`
- Test: `tests/test_streamlit_client.py`

- [ ] **Step 1: Add failing client tests**

Append to `tests/test_streamlit_client.py` and extend `FakeHttpClient` with a `get()` method:

```python
from app.api_client import get_knowledge_status_api, list_knowledge_documents_api


def _fake_get(self, url: str, timeout: float) -> FakeResponse:
    self.last_url = url
    self.last_timeout = timeout
    return self.response


FakeHttpClient.get = _fake_get


def test_list_knowledge_documents_api_gets_documents():
    http_client = FakeHttpClient(
        FakeResponse(200, {"documents": [{"filename": "ops.md"}]})
    )

    result = list_knowledge_documents_api("http://localhost:8000", http_client=http_client)

    assert http_client.last_url == "http://localhost:8000/knowledge/documents"
    assert result["documents"][0]["filename"] == "ops.md"


def test_get_knowledge_status_api_gets_status():
    http_client = FakeHttpClient(
        FakeResponse(200, {"document_count": 1, "chunk_count": 2, "index": {"available": True}})
    )

    result = get_knowledge_status_api("http://localhost:8000", http_client=http_client)

    assert http_client.last_url == "http://localhost:8000/knowledge/status"
    assert result["index"]["available"] is True
```

- [ ] **Step 2: Run client tests to verify failure**

Run: `pytest tests/test_streamlit_client.py -q`

Expected: FAIL with missing import for `list_knowledge_documents_api`.

- [ ] **Step 3: Implement API client helpers**

Add to `app/api_client.py`:

Because `app/api_client.py` currently calls `httpx` directly, first add a shared status-code helper:

```python
def _raise_for_bad_status(response: Any) -> None:
    if response.status_code != 200:
        raise ApiClientError(f"API request failed with status {response.status_code}: {response.text}")
```

Then implement the helpers with injectable `http_client` arguments:

```python
def list_knowledge_documents_api(
    api_base_url: str,
    http_client: Any = httpx,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response = http_client.get(_join_url(api_base_url, "/knowledge/documents"), timeout=timeout)
    _raise_for_bad_status(response)
    return response.json()


def get_knowledge_status_api(
    api_base_url: str,
    http_client: Any = httpx,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response = http_client.get(_join_url(api_base_url, "/knowledge/status"), timeout=timeout)
    _raise_for_bad_status(response)
    return response.json()


def reindex_knowledge_api(
    api_base_url: str,
    http_client: Any = httpx,
    timeout: float = 60.0,
) -> dict[str, Any]:
    response = http_client.post(_join_url(api_base_url, "/knowledge/reindex"), timeout=timeout)
    _raise_for_bad_status(response)
    return response.json()
```

Add upload helper:

```python
def upload_knowledge_document_api(
    api_base_url: str,
    filename: str,
    content: bytes,
    http_client: Any = httpx,
    timeout: float = 120.0,
) -> dict[str, Any]:
    response = http_client.post(
        _join_url(api_base_url, "/knowledge/documents/upload"),
        files={"file": (filename, content)},
        timeout=timeout,
    )
    _raise_for_bad_status(response)
    return response.json()
```

- [ ] **Step 4: Add Streamlit tab**

In `app/streamlit_app.py`, import the helpers:

```python
from app.api_client import (
    ApiClientError,
    ask_api,
    get_knowledge_status_api,
    list_knowledge_documents_api,
    reindex_knowledge_api,
    run_eval_api,
    upload_knowledge_document_api,
)
```

Update tabs from two tabs to three:

```python
copilot_tab, knowledge_tab, eval_tab = st.tabs(["Copilot", "Knowledge Base", "Evaluation"])
```

Add a render function:

```python
def render_knowledge_base_tab(api_base_url: str) -> None:
    st.subheader("Knowledge Base")
    uploaded = st.file_uploader(
        "Upload SOP, manual, or operation guide",
        type=["md", "txt", "pdf", "docx"],
    )
    if uploaded is not None and st.button("Index document"):
        try:
            result = upload_knowledge_document_api(
                api_base_url,
                uploaded.name,
                uploaded.getvalue(),
            )
            st.success(f"Indexed {result['document']['filename']}")
            st.json(result["index_status"])
        except ApiClientError as exc:
            st.error(str(exc))

    columns = st.columns(2)
    with columns[0]:
        if st.button("Refresh knowledge status"):
            st.session_state["knowledge_status"] = get_knowledge_status_api(api_base_url)
    with columns[1]:
        if st.button("Rebuild FAISS index"):
            st.session_state["knowledge_status"] = reindex_knowledge_api(api_base_url)

    try:
        status = st.session_state.get("knowledge_status") or get_knowledge_status_api(api_base_url)
        documents = list_knowledge_documents_api(api_base_url)["documents"]
        st.metric("Documents", status.get("document_count", 0))
        st.metric("Chunks", status.get("chunk_count", 0))
        st.json(status.get("index", {}))
        if documents:
            st.dataframe(documents, use_container_width=True)
        else:
            st.info("No uploaded knowledge documents yet.")
    except ApiClientError as exc:
        st.warning(str(exc))
```

Call it under the new tab:

```python
with knowledge_tab:
    render_knowledge_base_tab(api_base_url)
```

- [ ] **Step 5: Run Streamlit client tests**

Run: `pytest tests/test_streamlit_client.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/api_client.py app/streamlit_app.py tests/test_streamlit_client.py
git commit -m "feat: add knowledge base upload UI"
```

---

### Task 11: README And Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README section**

Add a concise section near the RAG or startup documentation:

```markdown
## Persistent Knowledge Base

The Copilot supports uploading operational documents into a persistent FAISS-backed RAG knowledge base.

Supported formats:

- `.md`
- `.txt`
- `.pdf`
- `.docx`

Storage layout:

```text
data/knowledge/
  uploads/       original uploaded files
  parsed/        parsed document text JSON
  faiss/         index.faiss and chunks.jsonl
  knowledge.db   SQLite metadata store
```

API:

```bash
curl -F "file=@manual.pdf" http://localhost:8000/knowledge/documents/upload
curl http://localhost:8000/knowledge/status
curl http://localhost:8000/knowledge/documents
```

The first version rebuilds the full FAISS index after upload/delete/reindex and atomically replaces the previous index only after rebuild succeeds.
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
pytest tests/test_knowledge_*.py tests/test_api_app.py tests/test_streamlit_client.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 4: Run lint**

Run:

```bash
ruff check .
```

Expected: PASS.

- [ ] **Step 5: Manual smoke test**

Start API:

```bash
uvicorn src.api.app:app --port 8000
```

In another shell:

```bash
curl -F "file=@data/documents/sample_hvac_guidance.md" http://localhost:8000/knowledge/documents/upload
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What does the uploaded HVAC guidance say?\",\"task_type\":\"document_qa\",\"workflow_engine\":\"deterministic\",\"memory_enabled\":false}"
```

Expected: upload returns `"available": true`; `/ask` returns at least one citation from uploaded knowledge.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document persistent knowledge base"
```

---

## Self-Review

- Spec coverage:
  - Upload API: Task 8.
  - PDF/DOCX/TXT/MD parsing: Task 4.
  - SQLite metadata: Task 3.
  - Chunk metadata: Task 5.
  - Persistent FAISS and atomic replacement: Task 6.
  - `/ask` integration: Task 9.
  - Streamlit UI: Task 10.
  - Docs and verification: Task 11.
- Type consistency:
  - `KnowledgeDocument`, `KnowledgeChunk`, `KnowledgeIndexStatus`, and `ParsedDocument` are defined before use.
  - `KnowledgeBaseService.ingest_existing_file()`, `reindex()`, `retriever()`, `status()`, `list_documents()`, and `delete_document()` are introduced before API usage.
  - `PersistentKnowledgeRetriever.search()` returns the existing RAG context shape.
- Scope check:
  - The plan intentionally excludes async queues, OCR, Qdrant/Milvus, permissions, table extraction, and version history.
