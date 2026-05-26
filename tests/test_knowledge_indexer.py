from __future__ import annotations

from pathlib import Path

from src.knowledge.indexer import KnowledgeFaissIndexer
from src.knowledge.retriever import PersistentKnowledgeRetriever
from src.knowledge.schemas import KnowledgeChunk
from src.knowledge.service import KnowledgeBaseService
from src.retrieval.embeddings import DeterministicHashEmbeddingProvider


def _chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    document_id = chunk_id.split("::")[0]
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        page_number=1,
        section_title="Ops",
        token_count=len(text.split()),
        metadata={"filename": f"{document_id}.md", "source_path": f"{document_id}.md"},
    )


def test_indexer_rebuilds_and_retriever_loads_persistent_index(tmp_path: Path):
    indexer = KnowledgeFaissIndexer(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    chunks = [
        _chunk("doc_a::chunk_0000", "cooling alarm response"),
        _chunk("doc_b::chunk_0000", "employee access policy"),
    ]

    status = indexer.rebuild(chunks)
    retriever = PersistentKnowledgeRetriever(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    results = retriever.search("cooling alarm", top_k=1)

    assert status.available is True
    assert status.chunk_count == 2
    assert results[0]["chunk_id"] == "doc_a::chunk_0000"
    assert results[0]["citation"]["title"] == "doc_a.md"
    assert results[0]["retrieval_mode"] == "persistent_faiss"


def test_indexer_writes_manifest_for_current_index_files(tmp_path: Path):
    indexer = KnowledgeFaissIndexer(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    indexer.rebuild([_chunk("doc_a::chunk_0000", "cooling alarm response")])

    manifest = tmp_path / "faiss" / "manifest.json"
    assert manifest.exists()
    content = manifest.read_text(encoding="utf-8")
    assert '"index_sha256"' in content
    assert '"chunks_sha256"' in content
    assert '"chunk_count": 1' in content


def test_indexer_keeps_existing_index_when_rebuild_fails(tmp_path: Path, monkeypatch):
    indexer = KnowledgeFaissIndexer(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    indexer.rebuild([_chunk("doc_a::chunk_0000", "cooling alarm response")])
    original_chunks = (tmp_path / "faiss" / "chunks.jsonl").read_text(encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise RuntimeError("replace failed")

    monkeypatch.setattr("src.knowledge.indexer._atomic_replace", fail_replace)

    try:
        indexer.rebuild([_chunk("doc_b::chunk_0000", "new text")])
    except RuntimeError:
        pass

    assert (tmp_path / "faiss" / "chunks.jsonl").read_text(encoding="utf-8") == original_chunks


def test_indexer_restores_existing_files_when_sidecar_replace_fails(tmp_path: Path, monkeypatch):
    indexer = KnowledgeFaissIndexer(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    indexer.rebuild([_chunk("doc_a::chunk_0000", "cooling alarm response")])
    original_index = (tmp_path / "faiss" / "index.faiss").read_bytes()
    original_chunks = (tmp_path / "faiss" / "chunks.jsonl").read_text(encoding="utf-8")

    real_replace = __import__(
        "src.knowledge.indexer",
        fromlist=["_atomic_replace"],
    )._atomic_replace
    calls = []

    def fail_on_sidecar_replace(src, dst):
        calls.append(dst.name)
        if dst.name == "chunks.jsonl":
            raise RuntimeError("sidecar replace failed")
        return real_replace(src, dst)

    monkeypatch.setattr("src.knowledge.indexer._atomic_replace", fail_on_sidecar_replace)

    try:
        indexer.rebuild([_chunk("doc_b::chunk_0000", "new text")])
    except RuntimeError:
        pass

    retriever = PersistentKnowledgeRetriever(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    assert calls == ["index.faiss", "chunks.jsonl"]
    assert (tmp_path / "faiss" / "index.faiss").read_bytes() == original_index
    assert (tmp_path / "faiss" / "chunks.jsonl").read_text(encoding="utf-8") == original_chunks
    assert retriever.search("cooling alarm", top_k=1)[0]["chunk_id"] == "doc_a::chunk_0000"


def test_retriever_rejects_sidecar_row_count_mismatch(tmp_path: Path):
    indexer = KnowledgeFaissIndexer(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    indexer.rebuild(
        [
            _chunk("doc_a::chunk_0000", "cooling alarm response"),
            _chunk("doc_b::chunk_0000", "employee access policy"),
        ]
    )
    chunks_path = tmp_path / "faiss" / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    retriever = PersistentKnowledgeRetriever(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    assert retriever.status["available"] is False
    assert "row count" in retriever.status["error"] or "sha256" in retriever.status["error"]
    assert retriever.search("cooling alarm", top_k=1) == []


def test_retriever_rejects_manifest_hash_mismatch(tmp_path: Path):
    indexer = KnowledgeFaissIndexer(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    indexer.rebuild([_chunk("doc_a::chunk_0000", "cooling alarm response")])
    manifest_path = tmp_path / "faiss" / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace('"chunk_count": 1', '"chunk_count": 2'),
        encoding="utf-8",
    )

    retriever = PersistentKnowledgeRetriever(
        index_dir=tmp_path / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    assert retriever.status["available"] is False
    assert "row count" in retriever.status["error"]


def test_service_ingests_document_and_rebuilds_index(tmp_path: Path):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    result = service.ingest_existing_file(source)
    results = service.retriever().search("cooling alarm", top_k=1)

    assert result["document"]["filename"] == "cooling.md"
    assert result["index_status"]["available"] is True
    assert results[0]["citation"]["title"] == "cooling.md"


def test_service_deduplicates_same_file_hash(tmp_path: Path):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    first = service.ingest_existing_file(source)
    second = service.ingest_existing_file(source)

    assert first["document"]["document_id"] == second["document"]["document_id"]
    assert second["deduplicated"] is True
    assert len(service.list_documents()) == 1


def test_service_cleans_chunks_when_reindex_fails(tmp_path: Path, monkeypatch):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    def fail_rebuild(*args, **kwargs):
        raise RuntimeError("reindex failed")

    monkeypatch.setattr("src.knowledge.indexer.KnowledgeFaissIndexer.rebuild", fail_rebuild)

    result = service.ingest_existing_file(source)

    assert result["document"]["status"] == "failed"
    assert result["document"]["chunk_count"] == 0
    assert service.store.load_chunks() == []


def test_service_keeps_indexed_document_when_index_status_metadata_write_fails(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    def fail_save_status(*args, **kwargs):
        raise RuntimeError("index status metadata failed")

    monkeypatch.setattr(service.store, "save_index_status", fail_save_status)

    result = service.ingest_existing_file(source)
    stored = service.store.get_document(result["document"]["document_id"])
    search_results = service.retriever().search("cooling alarm", top_k=1)

    assert result["document"]["status"] == "indexed"
    assert result["index_status"]["available"] is True
    assert result["index_status"]["metadata_error"] == "index status metadata failed"
    assert stored is not None
    assert stored.status == "indexed"
    assert len(service.store.load_chunks()) > 0
    assert search_results[0]["citation"]["title"] == "cooling.md"


def test_service_retries_failed_document_with_same_hash(tmp_path: Path, monkeypatch):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    def fail_parse(*args, **kwargs):
        raise RuntimeError("parse failed")

    monkeypatch.setattr("src.knowledge.service.parse_document", fail_parse)
    failed = service.ingest_existing_file(source)
    monkeypatch.undo()

    retried = service.ingest_existing_file(source)

    assert failed["document"]["status"] == "failed"
    assert retried["deduplicated"] is False
    assert retried["document"]["status"] == "indexed"
    assert retried["document"]["document_id"] != failed["document"]["document_id"]


def test_service_restores_document_when_delete_reindex_fails(tmp_path: Path, monkeypatch):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    indexed = service.ingest_existing_file(source)
    document_id = indexed["document"]["document_id"]
    document = service.store.get_document(document_id)
    assert document is not None
    source_path = Path(document.source_path)
    parsed_path = Path(document.parsed_path)

    def fail_rebuild(*args, **kwargs):
        raise RuntimeError("delete reindex failed")

    monkeypatch.setattr("src.knowledge.indexer.KnowledgeFaissIndexer.rebuild", fail_rebuild)

    try:
        service.delete_document(document_id)
    except RuntimeError:
        pass

    restored = service.store.get_document(document_id)
    chunks = service.store.load_chunks()
    search_results = service.retriever().search("cooling alarm", top_k=1)

    assert restored is not None
    assert restored.status == "indexed"
    assert chunks
    assert source_path.exists()
    assert parsed_path.exists()
    assert search_results[0]["citation"]["title"] == "cooling.md"


def test_service_reports_cleanup_error_after_successful_delete(tmp_path: Path, monkeypatch):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    indexed = service.ingest_existing_file(source)
    document_id = indexed["document"]["document_id"]

    def fail_cleanup(*args, **kwargs):
        raise PermissionError("file is locked")

    monkeypatch.setattr("src.knowledge.service._unlink_if_exists", fail_cleanup)

    result = service.delete_document(document_id)
    search_results = service.retriever().search("cooling alarm", top_k=1)

    assert result["deleted"] == document_id
    assert result["cleanup_errors"]
    assert "file is locked" in result["cleanup_errors"][0]
    assert service.store.get_document(document_id) is None
    assert service.store.load_chunks() == []
    assert search_results == []


def test_service_deduplicates_again_after_failed_then_successful_retry(tmp_path: Path, monkeypatch):
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nCooling alarm response procedure.", encoding="utf-8")
    service = KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )

    def fail_parse(*args, **kwargs):
        raise RuntimeError("parse failed")

    monkeypatch.setattr("src.knowledge.service.parse_document", fail_parse)
    failed = service.ingest_existing_file(source)
    monkeypatch.undo()
    retried = service.ingest_existing_file(source)
    third = service.ingest_existing_file(source)

    assert failed["document"]["status"] == "failed"
    assert retried["document"]["status"] == "indexed"
    assert third["deduplicated"] is True
    assert third["document"]["document_id"] == retried["document"]["document_id"]
    assert len(service.list_documents()) == 2
