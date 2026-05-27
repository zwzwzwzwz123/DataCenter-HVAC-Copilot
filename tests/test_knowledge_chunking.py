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


def test_chunk_parsed_document_splits_long_chinese_text_without_spaces():
    parsed = ParsedDocument(
        document_id="doc_cn",
        filename="manual.md",
        file_type=".md",
        file_hash="hash_cn",
        source_path="manual.md",
        pages=[
            ParsedPage(
                page_number=1,
                section_title="冷却策略",
                text="冷却策略需要结合回风温度机柜温差告警状态进行判断" * 6,
            )
        ],
        metadata={"filename": "manual.md", "source_path": "manual.md"},
    )

    chunks = chunk_parsed_document(parsed, chunk_size_words=20, overlap_words=4)

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.token_count <= 20 for chunk in chunks)
    assert "冷却策略" in chunks[0].text
