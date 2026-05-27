from __future__ import annotations

from src.core.text_chunking import detokenize_chunk, tokenize_for_chunking
from src.retrieval.schemas import DocumentChunk, SourceDocument


def chunk_document(
    document: SourceDocument,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = tokenize_for_chunking(document.text)
    if not words:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0
    section = _first_heading(document.text)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        text = detokenize_chunk(words[start:end])
        chunk_id = f"{document.metadata.source_id}::chunk_{chunk_index:04d}"
        chunks.append(
            document.metadata.to_chunk(
                chunk_id=chunk_id,
                text=text,
                section=section,
                start_word=start,
                end_word=end,
            )
        )
        if end == len(words):
            break
        start = end - overlap
        chunk_index += 1
    return chunks


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None

