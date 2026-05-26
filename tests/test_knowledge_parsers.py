from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge.parsers import (
    DocumentParseError,
    UnsupportedDocumentTypeError,
    parse_document,
)


def test_knowledge_package_imports():
    import src.knowledge as knowledge

    assert knowledge.__all__ == [
        "KnowledgeBaseService",
        "KnowledgeBaseStore",
        "PersistentKnowledgeRetriever",
    ]


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


def test_parse_rejects_empty_text_document(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(DocumentParseError, match="extractable text"):
        parse_document(path, document_id="doc_empty")
