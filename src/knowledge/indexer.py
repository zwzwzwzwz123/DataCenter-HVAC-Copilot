from __future__ import annotations

import json
import shutil
import uuid
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.knowledge.schemas import KnowledgeChunk, KnowledgeIndexStatus
from src.retrieval.embeddings import EmbeddingProvider


class KnowledgeFaissIndexer:
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

    def rebuild(self, chunks: list[KnowledgeChunk]) -> KnowledgeIndexStatus:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                'faiss-cpu is required for persistent knowledge indexing. Install it with: pip install -e ".[dev,dense]"'
            ) from exc

        self.index_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        tmp_index = self.index_dir / f"index.faiss.{token}.tmp"
        tmp_chunks = self.index_dir / f"chunks.jsonl.{token}.tmp"
        tmp_manifest = self.index_dir / f"manifest.json.{token}.tmp"

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_provider.embed_texts(texts) if texts else []
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.size == 0:
            matrix = np.zeros((0, 1), dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        index = faiss.IndexFlatIP(matrix.shape[1])
        if len(chunks) > 0:
            index.add(matrix)
        faiss.write_index(index, str(tmp_index))
        _write_chunks_sidecar(tmp_chunks, chunks)
        _write_manifest(
            tmp_manifest,
            index_path=tmp_index,
            chunks_path=tmp_chunks,
            chunk_count=len(chunks),
            embedding_provider=self.embedding_provider_name,
            embedding_model=self.embedding_model,
        )

        backup_index = None
        backup_chunks = None
        backup_manifest = None
        replaced_index = False
        replaced_chunks = False
        replaced_manifest = False
        try:
            backup_index = _backup_existing(self.index_path)
            backup_chunks = _backup_existing(self.chunks_path)
            backup_manifest = _backup_existing(self.manifest_path)
            _atomic_replace(tmp_index, self.index_path)
            replaced_index = True
            _atomic_replace(tmp_chunks, self.chunks_path)
            replaced_chunks = True
            _atomic_replace(tmp_manifest, self.manifest_path)
            replaced_manifest = True
        except Exception:
            _restore_backup(backup_index, self.index_path, remove_dst=replaced_index)
            _restore_backup(backup_chunks, self.chunks_path, remove_dst=replaced_chunks)
            _restore_backup(backup_manifest, self.manifest_path, remove_dst=replaced_manifest)
            raise
        finally:
            _unlink_if_exists(tmp_index)
            _unlink_if_exists(tmp_chunks)
            _unlink_if_exists(tmp_manifest)
            _unlink_if_exists(backup_index)
            _unlink_if_exists(backup_chunks)
            _unlink_if_exists(backup_manifest)

        return KnowledgeIndexStatus(
            available=True,
            faiss_path=str(self.index_path),
            chunks_path=str(self.chunks_path),
            chunk_count=len(chunks),
            embedding_provider=self.embedding_provider_name,
            embedding_model=self.embedding_model,
            updated_at=_utc_now(),
        )


def load_sidecar_chunks(path: str | Path) -> list[dict[str, Any]]:
    sidecar = Path(path)
    if not sidecar.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                records.append(loaded)
    return records


def load_index_manifest(path: str | Path) -> dict[str, Any]:
    manifest = Path(path)
    if not manifest.exists():
        return {}
    loaded = json.loads(manifest.read_text(encoding="utf-8") or "{}")
    return loaded if isinstance(loaded, dict) else {}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_chunks_sidecar(path: Path, chunks: list[KnowledgeChunk]) -> None:
    lines = [json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) for chunk in chunks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    index_path: Path,
    chunks_path: Path,
    chunk_count: int,
    embedding_provider: str,
    embedding_model: str,
) -> None:
    manifest = {
        "index_path": "index.faiss",
        "chunks_path": "chunks.jsonl",
        "index_sha256": file_sha256(index_path),
        "chunks_sha256": file_sha256(chunks_path),
        "chunk_count": chunk_count,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "created_at": _utc_now(),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _atomic_replace(src: Path, dst: Path) -> None:
    src.replace(dst)


def _backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.{uuid.uuid4().hex}.bak")
    shutil.copy2(path, backup)
    return backup


def _restore_backup(backup: Path | None, dst: Path, *, remove_dst: bool) -> None:
    if backup is None:
        if remove_dst:
            _unlink_if_exists(dst)
        return
    shutil.copy2(backup, dst)


def _unlink_if_exists(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
