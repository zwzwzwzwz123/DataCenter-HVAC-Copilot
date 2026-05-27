from __future__ import annotations

from src.core.text_chunking import detokenize_chunk, tokenize_for_chunking
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
        words = tokenize_for_chunking(page.normalized_text())
        start = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            text = detokenize_chunk(words[start:end])
            chunk_index = len(chunks)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.document_id}::chunk_{chunk_index:04d}",
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=text,
                    page_number=page.page_number,
                    section_title=page.section_title,
                    token_count=end - start,
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
