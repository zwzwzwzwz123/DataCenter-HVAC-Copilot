from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source_id: str
    title: str
    source_path: str
    published_at: str | None = None
    category: str | None = None

    def to_chunk(
        self,
        chunk_id: str,
        text: str,
        section: str | None,
        start_word: int,
        end_word: int,
    ) -> "DocumentChunk":
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=self,
            section=section,
            start_word=start_word,
            end_word=end_word,
        )


class SourceDocument(BaseModel):
    text: str
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    section: str | None = None
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)

    @property
    def citation(self) -> dict[str, str | int | None]:
        return {
            "source_id": self.metadata.source_id,
            "title": self.metadata.title,
            "source_path": self.metadata.source_path,
            "published_at": self.metadata.published_at,
            "category": self.metadata.category,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "start_word": self.start_word,
            "end_word": self.end_word,
        }

