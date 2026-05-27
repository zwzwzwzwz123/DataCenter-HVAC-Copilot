from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.knowledge.chunking import chunk_parsed_document
from src.knowledge.indexer import (
    KnowledgeFaissIndexer,
    file_sha256 as index_file_sha256,
    load_index_manifest,
    load_sidecar_chunks,
)
from src.knowledge.parsers import file_sha256, parse_document
from src.knowledge.retriever import PersistentKnowledgeRetriever
from src.knowledge.storage import KnowledgeBaseStore
from src.retrieval.embeddings import (
    DeterministicHashEmbeddingProvider,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


class KnowledgeBaseService:
    def __init__(
        self,
        *,
        knowledge_dir: str | Path = "data/knowledge",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_provider_name: str = "sentence-transformers",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.uploads_dir = self.knowledge_dir / "uploads"
        self.parsed_dir = self.knowledge_dir / "parsed"
        self.index_dir = self.knowledge_dir / "faiss"
        self.lock_dir = self.knowledge_dir / ".mutation.lock"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.store = KnowledgeBaseStore(self.knowledge_dir / "knowledge.db")
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model = embedding_model
        self._embedding_provider = embedding_provider
        self._mutation_lock = threading.RLock()

    def ingest_existing_file(self, source_path: str | Path) -> dict[str, Any]:
        with self._mutation_lock:
            with _process_lock(self.lock_dir):
                return self._ingest_existing_file(source_path)

    def _ingest_existing_file(self, source_path: str | Path) -> dict[str, Any]:
        source = Path(source_path)
        file_hash = file_sha256(source)
        existing = self.store.find_document_by_hash(file_hash)
        if existing is not None and existing.status != "failed":
            return {
                "document": existing.to_dict(),
                "deduplicated": True,
                "index_status": self.status()["index"],
            }

        document_id = f"doc_{uuid.uuid4().hex}"
        stored_path = self.uploads_dir / f"{document_id}_{_safe_filename(source.name)}"
        shutil.copyfile(source, stored_path)
        parsed_path = self.parsed_dir / f"{document_id}.json"

        try:
            parsed = parse_document(stored_path, document_id=document_id)
            parsed = replace(
                parsed,
                filename=source.name,
                metadata={
                    **parsed.metadata,
                    "filename": source.name,
                    "source_path": str(stored_path),
                },
            )
            chunks = chunk_parsed_document(parsed)
            parsed_path.write_text(
                json.dumps(
                    {
                        "document_id": parsed.document_id,
                        "filename": parsed.filename,
                        "file_type": parsed.file_type,
                        "file_hash": parsed.file_hash,
                        "source_path": parsed.source_path,
                        "pages": [
                            {
                                "page_number": page.page_number,
                                "section_title": page.section_title,
                                "text": page.text,
                            }
                            for page in parsed.pages
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            document = self.store.upsert_document(
                document_id=document_id,
                filename=source.name,
                file_type=source.suffix.lower(),
                file_hash=file_hash,
                source_path=str(stored_path),
                parsed_path=str(parsed_path),
                status="indexed",
                chunk_count=len(chunks),
                error_message="",
                metadata={"filename": source.name},
            )
            self.store.replace_chunks(document_id, chunks)
            index_status = self._reindex_unlocked()
            return {
                "document": document.to_dict(),
                "deduplicated": False,
                "index_status": index_status,
            }
        except Exception as exc:
            self.store.delete_chunks(document_id)
            document = self.store.upsert_document(
                document_id=document_id,
                filename=source.name,
                file_type=source.suffix.lower(),
                file_hash=file_hash,
                source_path=str(stored_path),
                parsed_path=str(parsed_path),
                status="failed",
                chunk_count=0,
                error_message=str(exc),
                metadata={"filename": source.name},
            )
            return {
                "document": document.to_dict(),
                "deduplicated": False,
                "index_status": self.status()["index"],
            }

    def list_documents(self) -> list[dict[str, Any]]:
        return [document.to_dict() for document in self.store.list_documents()]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        document = self.store.get_document(document_id)
        return document.to_dict() if document is not None else None

    def delete_document(self, document_id: str) -> dict[str, Any]:
        with self._mutation_lock:
            with _process_lock(self.lock_dir):
                return self._delete_document(document_id)

    def _delete_document(self, document_id: str) -> dict[str, Any]:
        document = self.store.get_document(document_id)
        if document is None:
            return {"deleted": document_id, "index_status": self._reindex_unlocked()}
        chunks = self.store.load_document_chunks(document_id)
        self.store.delete_document(document_id)
        try:
            index_status = self._reindex_unlocked()
        except Exception:
            self.store.upsert_document(
                document_id=document.document_id,
                filename=document.filename,
                file_type=document.file_type,
                file_hash=document.file_hash,
                source_path=document.source_path,
                parsed_path=document.parsed_path,
                status=document.status,
                chunk_count=document.chunk_count,
                error_message=document.error_message,
                metadata=document.metadata,
            )
            self.store.replace_chunks(document_id, chunks)
            raise
        cleanup_errors = _cleanup_document_files(document)
        result: dict[str, Any] = {"deleted": document_id, "index_status": index_status}
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
        return result

    def reindex(self) -> dict[str, Any]:
        with self._mutation_lock:
            with _process_lock(self.lock_dir):
                return self._reindex_unlocked()

    def _reindex_unlocked(self) -> dict[str, Any]:
        chunks = self.store.load_chunks()
        status = KnowledgeFaissIndexer(
            index_dir=self.index_dir,
            embedding_provider=self._get_embedding_provider(),
            embedding_provider_name=self.embedding_provider_name,
            embedding_model=self.embedding_model,
        ).rebuild(chunks)
        result = status.to_dict()
        try:
            self.store.save_index_status(status)
        except Exception as exc:
            result["metadata_error"] = str(exc)
        return result

    def retriever(self) -> PersistentKnowledgeRetriever:
        return PersistentKnowledgeRetriever(
            index_dir=self.index_dir,
            embedding_provider=self._get_embedding_provider(),
            embedding_provider_name=self.embedding_provider_name,
            embedding_model=self.embedding_model,
        )

    def status(self) -> dict[str, Any]:
        index_status = _read_index_status(
            index_dir=self.index_dir,
            embedding_provider_name=self.embedding_provider_name,
            embedding_model=self.embedding_model,
        )
        return {
            "document_count": len(self.store.list_documents()),
            "chunk_count": len(self.store.load_chunks()),
            "index": index_status,
        }

    def _get_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = _build_embedding_provider(
                self.embedding_provider_name,
                self.embedding_model,
            )
        return self._embedding_provider


def _build_embedding_provider(provider: str, model: str) -> EmbeddingProvider:
    if provider == "deterministic":
        return DeterministicHashEmbeddingProvider()
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(model)
    raise ValueError(f"Unsupported knowledge embedding provider: {provider}")


def _safe_filename(filename: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in filename)


def _read_index_status(
    *,
    index_dir: Path,
    embedding_provider_name: str,
    embedding_model: str,
) -> dict[str, Any]:
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.jsonl"
    manifest_path = index_dir / "manifest.json"
    base_status = {
        "faiss_path": str(index_path),
        "chunks_path": str(chunks_path),
        "embedding_provider": embedding_provider_name,
        "embedding_model": embedding_model,
        "updated_at": None,
    }
    if not index_path.exists() or not chunks_path.exists() or not manifest_path.exists():
        return {
            **base_status,
            "available": False,
            "chunk_count": 0,
            "error": "index files not found",
        }
    try:
        manifest = load_index_manifest(manifest_path)
        chunks = load_sidecar_chunks(chunks_path)
        _validate_status_manifest(
            manifest=manifest,
            index_path=index_path,
            chunks_path=chunks_path,
            sidecar_count=len(chunks),
        )
    except Exception as exc:
        return {
            **base_status,
            "available": False,
            "chunk_count": 0,
            "error": str(exc),
        }
    return {
        **base_status,
        "available": True,
        "chunk_count": len(chunks),
        "updated_at": str(manifest.get("created_at") or ""),
        "error": "",
    }


def _validate_status_manifest(
    *,
    manifest: dict[str, Any],
    index_path: Path,
    chunks_path: Path,
    sidecar_count: int,
) -> None:
    if not manifest.get("index_sha256") or not manifest.get("chunks_sha256"):
        raise ValueError("manifest is missing index_sha256 or chunks_sha256")
    if index_file_sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("index.faiss sha256 does not match manifest")
    if index_file_sha256(chunks_path) != manifest["chunks_sha256"]:
        raise ValueError("chunks.jsonl sha256 does not match manifest")
    if int(manifest.get("chunk_count", -1)) != sidecar_count:
        raise ValueError(
            f"manifest row count mismatch: manifest has {manifest.get('chunk_count')}, sidecar has {sidecar_count}"
        )


def _unlink_if_exists(path: str) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _cleanup_document_files(document) -> list[str]:
    errors: list[str] = []
    for path in (document.source_path, document.parsed_path):
        try:
            _unlink_if_exists(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return errors


@contextmanager
def _process_lock(lock_dir: Path, *, timeout_seconds: float = 30.0, stale_seconds: float = 300.0):
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    while not acquired:
        try:
            lock_dir.mkdir()
            acquired = True
        except FileExistsError as exc:
            if _is_stale_lock(lock_dir, stale_seconds=stale_seconds):
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for knowledge mutation lock: {lock_dir}"
                ) from exc
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except FileNotFoundError:
            pass


def _is_stale_lock(lock_dir: Path, *, stale_seconds: float) -> bool:
    try:
        age = time.time() - lock_dir.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > stale_seconds
