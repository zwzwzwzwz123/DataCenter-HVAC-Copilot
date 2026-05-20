from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Iterable

from src.retrieval.schemas import DocumentChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")


class KeywordRetriever:
    """Small lexical retriever used as a replaceable baseline."""

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self._chunk_tokens = [Counter(_tokenize(chunk.text)) for chunk in self.chunks]
        self._doc_frequency = Counter()
        for token_counts in self._chunk_tokens:
            self._doc_frequency.update(token_counts.keys())

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        query_tokens = Counter(_tokenize(query))
        if not query_tokens:
            return []

        scored = []
        for chunk, token_counts in zip(self.chunks, self._chunk_tokens):
            score = self._score(query_tokens, token_counts)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            {
                "chunk_id": chunk.chunk_id,
                "score": score,
                "text": chunk.text,
                "citation": chunk.citation,
            }
            for score, chunk in scored[:top_k]
        ]

    def _score(self, query_tokens: Counter, chunk_tokens: Counter) -> float:
        score = 0.0
        total_chunks = max(len(self.chunks), 1)
        for token, query_count in query_tokens.items():
            if token not in chunk_tokens:
                continue
            idf = log((1 + total_chunks) / (1 + self._doc_frequency[token])) + 1
            score += query_count * chunk_tokens[token] * idf
        return score


class HybridRetriever:
    """Lightweight BM25-style lexical retriever for early retrieval experiments."""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._chunk_tokens = [Counter(_tokenize(chunk.text)) for chunk in self.chunks]
        self._doc_frequency = Counter()
        for token_counts in self._chunk_tokens:
            self._doc_frequency.update(token_counts.keys())
        lengths = [sum(token_counts.values()) for token_counts in self._chunk_tokens]
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        query_tokens = Counter(_tokenize(query))
        if not query_tokens:
            return []

        scored = []
        for chunk, token_counts in zip(self.chunks, self._chunk_tokens):
            score = self._score(query_tokens, token_counts)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            {
                "chunk_id": chunk.chunk_id,
                "score": score,
                "text": chunk.text,
                "citation": chunk.citation,
                "retrieval_mode": "hybrid_bm25",
            }
            for score, chunk in scored[:top_k]
        ]

    def _score(self, query_tokens: Counter, chunk_tokens: Counter) -> float:
        score = 0.0
        total_chunks = max(len(self.chunks), 1)
        chunk_length = max(sum(chunk_tokens.values()), 1)
        average_length = self._average_length or chunk_length
        for token, query_count in query_tokens.items():
            term_frequency = chunk_tokens.get(token, 0)
            if term_frequency == 0:
                continue
            doc_frequency = self._doc_frequency[token]
            idf = log(1 + (total_chunks - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * (chunk_length / average_length)
            )
            score += query_count * idf * (
                term_frequency * (self.k1 + 1) / denominator
            )
        return score


class RerankingRetriever:
    """Second-stage lexical reranker that can wrap an existing retriever."""

    def __init__(
        self,
        base_retriever,
        *,
        candidate_k: int = 10,
        phrase_weight: float = 2.0,
        coverage_weight: float = 3.0,
        base_score_weight: float = 0.1,
        metadata_weight: float = 1.0,
    ) -> None:
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.phrase_weight = phrase_weight
        self.coverage_weight = coverage_weight
        self.base_score_weight = base_score_weight
        self.metadata_weight = metadata_weight
        self.chunks = getattr(base_retriever, "chunks", [])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")

        query_tokens = _tokenize(query)
        candidates = self.base_retriever.search(query, top_k=self.candidate_k)
        reranked = []
        for candidate in candidates:
            score = self._score(query, query_tokens, candidate)
            metadata_score = self._metadata_score(query_tokens, candidate)
            score += self.metadata_weight * metadata_score
            updated = dict(candidate)
            updated["base_score"] = candidate.get("score", 0.0)
            updated["base_retrieval_mode"] = candidate.get("retrieval_mode", "keyword")
            updated["metadata_score"] = metadata_score
            updated["score"] = score
            updated["rerank_score"] = score
            updated["retrieval_mode"] = "rerank_keyword_overlap"
            reranked.append(updated)

        reranked.sort(key=lambda item: (-item["rerank_score"], item["chunk_id"]))
        return reranked[:top_k]

    def _score(self, query: str, query_tokens: list[str], candidate: dict) -> float:
        text = str(candidate.get("text", ""))
        text_tokens = set(_tokenize(text))
        query_token_set = set(query_tokens)
        if not query_token_set:
            return 0.0
        coverage = len(query_token_set & text_tokens) / len(query_token_set)
        phrase_hits = _phrase_hit_count(query, text)
        base_score = float(candidate.get("score", 0.0))
        return (
            self.coverage_weight * coverage
            + self.phrase_weight * phrase_hits
            + self.base_score_weight * base_score
        )

    def _metadata_score(self, query_tokens: list[str], candidate: dict) -> float:
        query_token_set = set(query_tokens)
        if not query_token_set:
            return 0.0
        citation = candidate.get("citation") or {}
        metadata_text = " ".join(
            str(citation.get(key, ""))
            for key in ("source_id", "title", "section")
            if citation.get(key) is not None
        )
        metadata_tokens = set(_tokenize(metadata_text))
        return len(query_token_set & metadata_tokens) / len(query_token_set)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _phrase_hit_count(query: str, text: str) -> int:
    query_tokens = _tokenize(query)
    text_lower = text.lower()
    count = 0
    for length in range(min(5, len(query_tokens)), 1, -1):
        for start in range(0, len(query_tokens) - length + 1):
            phrase = " ".join(query_tokens[start : start + length])
            if phrase in text_lower:
                count += 1
    return count
