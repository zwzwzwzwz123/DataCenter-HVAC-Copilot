from __future__ import annotations

from pathlib import Path

from src.retrieval.schemas import DocumentMetadata, SourceDocument


SUPPORTED_TEXT_SUFFIXES = {".md", ".txt"}


def load_markdown_document(
    path: str | Path,
    source_id: str | None = None,
    title: str | None = None,
    published_at: str | None = None,
    category: str | None = None,
) -> SourceDocument:
    document_path = Path(path)
    text = document_path.read_text(encoding="utf-8")
    metadata = DocumentMetadata(
        source_id=source_id or document_path.stem,
        title=title or _infer_title(text, document_path.stem),
        source_path=str(document_path),
        published_at=published_at,
        category=category,
    )
    return SourceDocument(text=text, metadata=metadata)


def load_text_documents(directory: str | Path) -> list[SourceDocument]:
    root = Path(directory)
    documents: list[SourceDocument] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES:
            documents.append(load_markdown_document(path))
    return documents


def _infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback

