from __future__ import annotations

from typing import Protocol


class CrossEncoderScorer(Protocol):
    model_name: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        ...


class SentenceTransformersCrossEncoderScorer:
    """Cross-encoder scorer backed by sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                'Install it with: pip install -e ".[dense]"'
            ) from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        raw_scores = self.model.predict(pairs)
        return [float(score) for score in raw_scores]


class CrossEncoderRerankingRetriever:
    """Second-stage reranker that scores query-document pairs with a cross-encoder."""

    def __init__(
        self,
        base_retriever,
        *,
        scorer: CrossEncoderScorer,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")
        self.base_retriever = base_retriever
        self.scorer = scorer
        self.candidate_k = candidate_k
        self.chunks = getattr(base_retriever, "chunks", [])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")

        candidates = self.base_retriever.search(query, top_k=self.candidate_k)
        if not candidates:
            return []

        texts = [str(candidate.get("text", "")) for candidate in candidates]
        scores = self.scorer.score(query, texts)
        if len(scores) != len(candidates):
            raise ValueError("cross-encoder scorer returned a score count mismatch.")

        reranked = []
        for index, (candidate, score) in enumerate(zip(candidates, scores), start=1):
            updated = dict(candidate)
            updated["base_score"] = float(candidate.get("score", 0.0))
            updated["base_retrieval_mode"] = candidate.get("retrieval_mode", "unknown")
            updated["candidate_rank"] = index
            updated["score"] = float(score)
            updated["cross_encoder_score"] = float(score)
            updated["cross_encoder_model"] = self.scorer.model_name
            updated["retrieval_mode"] = "cross_encoder_rerank"
            reranked.append(updated)

        reranked.sort(
            key=lambda item: (
                -float(item["cross_encoder_score"]),
                int(item["candidate_rank"]),
                str(item.get("chunk_id", "")),
            )
        )
        return reranked[:top_k]
