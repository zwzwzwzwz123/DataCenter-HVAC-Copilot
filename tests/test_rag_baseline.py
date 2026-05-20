from pathlib import Path

from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.retriever import KeywordRetriever


def test_extractive_rag_answer_contains_citations_and_contexts():
    document = load_markdown_document(
        Path("data/documents/sample_hvac_guidance.md"),
        source_id="sample_hvac_guidance",
        title="Sample HVAC Guidance",
        published_at="2026",
        category="internal_note",
    )
    chunks = chunk_document(document, chunk_size=35, overlap=5)
    retriever = KeywordRetriever(chunks)
    pipeline = ExtractiveRAGPipeline(retriever)

    answer = pipeline.answer("Why should PUE-like values not be invented?", top_k=2)

    assert answer.question == "Why should PUE-like values not be invented?"
    assert "PUE-like values require an explicit calculation method" in answer.answer
    assert len(answer.citations) >= 1
    assert answer.citations[0]["source_id"] == "sample_hvac_guidance"
    assert len(answer.retrieved_contexts) == 2


def test_extractive_rag_returns_uncertain_answer_without_context():
    pipeline = ExtractiveRAGPipeline(KeywordRetriever([]))

    answer = pipeline.answer("What is the chiller schedule?", top_k=2)

    assert answer.answer.startswith("未找到足够的检索证据")
    assert answer.citations == []
    assert answer.retrieved_contexts == []

