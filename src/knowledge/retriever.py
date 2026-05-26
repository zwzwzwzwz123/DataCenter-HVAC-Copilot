from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.knowledge.indexer import file_sha256, load_index_manifest, load_sidecar_chunks
from src.knowledge.schemas import KnowledgeIndexStatus
from src.retrieval.embeddings import EmbeddingProvider


class PersistentKnowledgeRetriever:
    def __init__(
        self,
        *,
        index_dir: str | Path = "data/knowledge/faiss",
        embedding_provider: EmbeddingProvider,
        embedding_provider_name: str,
        embedding_model: str,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.manifest_path = self.index_dir / "manifest.json"
        self.embedding_provider = embedding_provider
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model = embedding_model
        self._index = None
        self._chunks: list[dict[str, Any]] = []
        self.status = self._load()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self._index is None or not self._chunks or not query.strip():
            return []
        query_vector = np.asarray(self.embedding_provider.embed_texts([query]), dtype="float32")
        scores, indices = self._index.search(query_vector, min(top_k, len(self._chunks)))
        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            if int(index) >= len(self._chunks):
                continue
            chunk = self._chunks[int(index)]
            metadata = chunk.get("metadata", {})
            citation = {
                "source_id": chunk["document_id"],
                "title": metadata.get("filename", chunk["document_id"]),
                "source_path": metadata.get("source_path", ""),
                "page_number": chunk.get("page_number"),
                "section": chunk.get("section_title"),
            }
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "text": chunk["text"],
                    "citation": citation,
                    "retrieval_mode": "persistent_faiss",
                    "metadata": metadata,
                }
            )
        return results

    def _load(self) -> dict[str, Any]:
        if not self.index_path.exists() or not self.chunks_path.exists() or not self.manifest_path.exists():
            return KnowledgeIndexStatus(
                available=False,
                faiss_path=str(self.index_path),
                chunks_path=str(self.chunks_path),
                chunk_count=0,
                embedding_provider=self.embedding_provider_name,
                embedding_model=self.embedding_model,
                error="index files not found",
            ).to_dict()
        try:
            import faiss

            manifest = load_index_manifest(self.manifest_path)
            _validate_manifest_file_hashes(
                manifest=manifest,
                index_path=self.index_path,
                chunks_path=self.chunks_path,
            )
            self._index = faiss.read_index(str(self.index_path))
            self._chunks = load_sidecar_chunks(self.chunks_path)
            _validate_row_counts(
                manifest=manifest,
                index_total=int(self._index.ntotal),
                sidecar_count=len(self._chunks),
            )
        except Exception as exc:
            self._index = None
            self._chunks = []
            return KnowledgeIndexStatus(
                available=False,
                faiss_path=str(self.index_path),
                chunks_path=str(self.chunks_path),
                chunk_count=0,
                embedding_provider=self.embedding_provider_name,
                embedding_model=self.embedding_model,
                error=str(exc),
            ).to_dict()
        return KnowledgeIndexStatus(
            available=True,
            faiss_path=str(self.index_path),
            chunks_path=str(self.chunks_path),
            chunk_count=len(self._chunks),
            embedding_provider=self.embedding_provider_name,
            embedding_model=self.embedding_model,
        ).to_dict()


def _validate_manifest_file_hashes(
    *,
    manifest: dict[str, Any],
    index_path: Path,
    chunks_path: Path,
) -> None:
    expected_index_hash = manifest.get("index_sha256")
    expected_chunks_hash = manifest.get("chunks_sha256")
    if not expected_index_hash or not expected_chunks_hash:
        raise ValueError("manifest is missing index_sha256 or chunks_sha256")
    if file_sha256(index_path) != expected_index_hash:
        raise ValueError("index.faiss sha256 does not match manifest")
    if file_sha256(chunks_path) != expected_chunks_hash:
        raise ValueError("chunks.jsonl sha256 does not match manifest")


def _validate_row_counts(
    *,
    manifest: dict[str, Any],
    index_total: int,
    sidecar_count: int,
) -> None:
    manifest_count = int(manifest.get("chunk_count", -1))
    if index_total != sidecar_count:
        raise ValueError(
            f"FAISS row count mismatch: index has {index_total}, sidecar has {sidecar_count}"
        )
    if manifest_count != sidecar_count:
        raise ValueError(
            f"manifest row count mismatch: manifest has {manifest_count}, sidecar has {sidecar_count}"
        )
