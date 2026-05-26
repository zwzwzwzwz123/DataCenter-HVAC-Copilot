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
    assert chunks[1].text.startswith("five six")


def test_chunk_parsed_document_rejects_invalid_overlap():
    parsed = ParsedDocument(
        document_id="doc_1",
        filename="manual.txt",
        file_type=".txt",
        file_hash="hash1",
        source_path="manual.txt",
        pages=[ParsedPage(page_number=None, text="one two")],
    )

    try:
        chunk_parsed_document(parsed, chunk_size_words=5, overlap_words=5)
    except ValueError as exc:
        assert "overlap_words" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid overlap")
