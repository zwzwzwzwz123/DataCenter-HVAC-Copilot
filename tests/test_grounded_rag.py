from __future__ import annotations

from pathlib import Path

from src.agent.answer_generator import AnswerGeneratorInput, GeneratedAnswer
from src.evaluation.metrics import grounding_rate
from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.rag import GroundedRAGPipeline
from src.retrieval.retriever import KeywordRetriever


class RecordingGenerator:
    def __init__(self) -> None:
        self.payloads: list[AnswerGeneratorInput] = []

    def generate(self, payload: AnswerGeneratorInput) -> GeneratedAnswer:
        self.payloads.append(payload)
        return GeneratedAnswer(
            answer=(
                "回答：基于 retrieved_contexts 生成。\n"
                "引用：\n"
                "- sample_hvac_guidance: Sample HVAC Guidance"
            ),
            generator="recording",
        )


def test_grounded_rag_pipeline_calls_generator_with_contexts() -> None:
    document = load_markdown_document(
        Path("data/documents/sample_hvac_guidance.md"),
        source_id="sample_hvac_guidance",
        title="Sample HVAC Guidance",
        published_at="2026",
        category="internal_note",
    )
    chunks = chunk_document(document, chunk_size=35, overlap=5)
    generator = RecordingGenerator()
    pipeline = GroundedRAGPipeline(KeywordRetriever(chunks), answer_generator=generator)

    answer = pipeline.answer("Why should PUE-like values not be invented?", top_k=2)

    assert answer.question == "Why should PUE-like values not be invented?"
    assert answer.answer.startswith("回答：")
    assert answer.citations[0]["source_id"] == "sample_hvac_guidance"
    assert len(answer.retrieved_contexts) == 2
    assert len(generator.payloads) == 1
    assert generator.payloads[0].route == "document_qa"
    assert generator.payloads[0].retrieved_contexts == answer.retrieved_contexts


def test_grounding_rate_scores_explicit_citation_overlap() -> None:
    records = []
    predictions = {
        "doc_1": {
            "answer": "回答。\n引用：\n- sample_hvac_guidance: Sample HVAC Guidance",
            "retrieved_contexts": [
                {
                    "citation": {
                        "source_id": "sample_hvac_guidance",
                        "title": "Sample HVAC Guidance",
                    }
                }
            ],
        },
        "doc_2": {
            "answer": "回答。\n引用：\n- unrelated_source: Unrelated",
            "retrieved_contexts": [
                {
                    "citation": {
                        "source_id": "sample_hvac_guidance",
                        "title": "Sample HVAC Guidance",
                    }
                }
            ],
        },
    }

    class Record:
        def __init__(self, record_id: str) -> None:
            self.id = record_id
            self.required_documents = ["sample_hvac_guidance"]
            self.required_tools = []

    records = [Record("doc_1"), Record("doc_2")]

    assert grounding_rate(records, predictions) == 0.5
