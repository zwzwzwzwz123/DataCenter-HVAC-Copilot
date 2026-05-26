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
