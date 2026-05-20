from __future__ import annotations

from typing import Protocol

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

