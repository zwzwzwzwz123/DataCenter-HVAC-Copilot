from __future__ import annotations

from pathlib import Path

from src.knowledge.schemas import KnowledgeChunk, KnowledgeDocument, ParsedPage
from src.knowledge.storage import KnowledgeBaseStore


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


def test_knowledge_code_uses_python_310_compatible_timezone_imports():
    for path in Path("src/knowledge").glob("*.py"):
        assert "datetime import UTC" not in path.read_text(encoding="utf-8")


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

    found = store.find_document_by_hash("same")

    assert found is not None
    assert found.document_id == "doc_1"
