from __future__ import annotations

from typing import Protocol

from src.agent.answer_generator import AnswerGenerator
from src.agent.answer_generator import AnswerGeneratorInput
from src.agent.answer_generator import DeterministicAnswerGenerator
from pydantic import BaseModel


class Searcher(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        ...


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    retrieved_contexts: list[dict]


class ExtractiveRAGPipeline:
    """Minimal extractive RAG baseline with citation-preserving output."""

    def __init__(self, retriever: Searcher) -> None:
        self.retriever = retriever

    def answer(self, question: str, top_k: int = 3) -> RAGAnswer:
        contexts = self.retriever.search(question, top_k=top_k)
        if not contexts:
            return RAGAnswer(
                question=question,
                answer="未找到足够的检索证据，无法给出可靠回答。",
                citations=[],
                retrieved_contexts=[],
            )

        answer_text = " ".join(context["text"] for context in contexts)
        citations = [context["citation"] for context in contexts]
        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_contexts=contexts,
        )


class GroundedRAGPipeline:
    """Grounded RAG that keeps retrieval separate from answer generation."""

    def __init__(
        self,
        retriever: Searcher,
        *,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()

    def answer(self, question: str, top_k: int = 3) -> RAGAnswer:
        contexts = self.retriever.search(question, top_k=top_k)
        if not contexts:
            return RAGAnswer(
                question=question,
                answer="未找到足够的检索证据，无法给出可靠回答。",
                citations=[],
                retrieved_contexts=[],
            )

        grounded_contexts = [_with_top_level_citation_metadata(context) for context in contexts]
        citations = [context["citation"] for context in grounded_contexts]
        generated = self.answer_generator.generate(
            AnswerGeneratorInput(
                question=question,
                route="document_qa",
                route_reason="grounded_rag_pipeline",
                retrieved_contexts=grounded_contexts,
                citations=citations,
            )
        )
        return RAGAnswer(
            question=question,
            answer=generated.answer,
            citations=citations,
            retrieved_contexts=grounded_contexts,
        )


def _with_top_level_citation_metadata(context: dict) -> dict:
    citation = context.get("citation")
    if not isinstance(citation, dict):
        return context
    return {
        **context,
        "source_id": context.get("source_id") or citation.get("source_id"),
        "title": context.get("title") or citation.get("title"),
    }

