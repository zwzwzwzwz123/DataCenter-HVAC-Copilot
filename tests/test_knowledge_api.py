from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from src.api.app import _safe_upload_filename, create_app
from src.knowledge.indexer import KnowledgeFaissIndexer
from src.knowledge.schemas import KnowledgeChunk
from src.retrieval.embeddings import DeterministicHashEmbeddingProvider


def test_upload_document_indexes_and_lists_document(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    response = client.post(
        "/knowledge/documents/upload",
        files={"file": ("cooling.md", b"# Cooling SOP\n\nCooling alarm response.", "text/markdown")},
    )
    listing = client.get("/knowledge/documents")
    status = client.get("/knowledge/status")

    assert response.status_code == 200
    assert response.json()["document"]["filename"] == "cooling.md"
    assert response.json()["index_status"]["available"] is True
    assert listing.json()["documents"][0]["filename"] == "cooling.md"
    assert status.json()["chunk_count"] >= 1


def test_upload_filename_is_sanitized_before_temp_write(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    response = client.post(
        "/knowledge/documents/upload",
        files={
            "file": (
                "C:\\temp\\escape.md",
                b"# Cooling SOP\n\nCooling alarm response.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["document"]["filename"] == "escape.md"


def test_safe_upload_filename_drops_client_path_components():
    assert _safe_upload_filename("..\\..\\escape.md") == "escape.md"
    assert _safe_upload_filename("../escape.md") == "escape.md"
    assert _safe_upload_filename("C:\\temp\\escape.md") == "escape.md"


def test_delete_document_rebuilds_index(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)
    uploaded = client.post(
        "/knowledge/documents/upload",
        files={"file": ("ops.txt", b"employee cooling procedure", "text/plain")},
    ).json()

    document_id = uploaded["document"]["document_id"]
    response = client.delete(f"/knowledge/documents/{document_id}")

    assert response.status_code == 200
    assert client.get("/knowledge/documents").json()["documents"] == []


def test_upload_returns_refresh_error_when_orchestrator_refresh_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    def fail_refresh(*args, **kwargs):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr("src.api.app.build_demo_orchestrator", fail_refresh)

    response = client.post(
        "/knowledge/documents/upload",
        files={"file": ("cooling.md", b"# Cooling SOP\n\nCooling alarm response.", "text/markdown")},
    )
    listing = client.get("/knowledge/documents")

    assert response.status_code == 200
    assert response.json()["document"]["filename"] == "cooling.md"
    assert response.json()["refresh_error"] == "refresh failed"
    assert listing.json()["documents"][0]["filename"] == "cooling.md"


def test_ask_retries_refresh_after_upload_refresh_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)
    real_build = __import__(
        "src.api.app",
        fromlist=["build_demo_orchestrator"],
    ).build_demo_orchestrator
    calls = {"count": 0}

    def fail_once_then_refresh(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("refresh failed once")
        return real_build(*args, **kwargs)

    monkeypatch.setattr("src.api.app.build_demo_orchestrator", fail_once_then_refresh)
    upload = client.post(
        "/knowledge/documents/upload",
        files={
            "file": (
                "custom_sop.md",
                b"# Custom SOP\n\nBlue coolant valve inspection is required.",
                "text/markdown",
            )
        },
    )

    response = client.post(
        "/ask",
        json={
            "question": "What does the custom SOP say about blue coolant valve inspection?",
            "task_type": "document_qa",
            "workflow_engine": "deterministic",
            "memory_enabled": False,
        },
    )

    body = response.json()
    assert upload.status_code == 200
    assert upload.json()["refresh_error"] == "refresh failed once"
    assert response.status_code == 200
    assert any("Blue coolant valve" in context["text"] for context in body["retrieved_contexts"])


def test_ask_reports_refresh_dirty_when_retry_refresh_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    def fail_refresh(*args, **kwargs):
        raise RuntimeError("refresh still failed")

    monkeypatch.setattr("src.api.app.build_demo_orchestrator", fail_refresh)
    client.post(
        "/knowledge/documents/upload",
        files={
            "file": (
                "custom_sop.md",
                b"# Custom SOP\n\nBlue coolant valve inspection is required.",
                "text/markdown",
            )
        },
    )

    response = client.post(
        "/ask",
        json={
            "question": "What does the custom SOP say about blue coolant valve inspection?",
            "task_type": "document_qa",
            "workflow_engine": "deterministic",
            "memory_enabled": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["refresh_dirty"] is True
    assert response.json()["refresh_error"] == "refresh still failed"


def test_refresh_failure_does_not_partially_update_orchestrators(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    def fail_langgraph_refresh(*args, **kwargs):
        raise RuntimeError("langgraph refresh failed")

    monkeypatch.setattr("src.api.app.LangGraphOrchestrator", fail_langgraph_refresh)
    upload = client.post(
        "/knowledge/documents/upload",
        files={
            "file": (
                "custom_sop.md",
                b"# Custom SOP\n\nBlue coolant valve inspection is required.",
                "text/markdown",
            )
        },
    )
    response = client.post(
        "/ask",
        json={
            "question": "What does the custom SOP say about blue coolant valve inspection?",
            "task_type": "document_qa",
            "workflow_engine": "deterministic",
            "memory_enabled": False,
        },
    )

    body = response.json()
    assert upload.status_code == 200
    assert upload.json()["refresh_dirty"] is True
    assert response.status_code == 200
    assert body["refresh_dirty"] is True
    assert not any("Blue coolant valve" in context["text"] for context in body["retrieved_contexts"])


def test_status_reports_refresh_dirty_after_refresh_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    def fail_refresh(*args, **kwargs):
        raise RuntimeError("refresh still failed")

    monkeypatch.setattr("src.api.app.build_demo_orchestrator", fail_refresh)
    client.post(
        "/knowledge/documents/upload",
        files={"file": ("cooling.md", b"# Cooling SOP\n\nCooling alarm response.", "text/markdown")},
    )

    status = client.get("/knowledge/status")

    assert status.status_code == 200
    assert status.json()["refresh_dirty"] is True
    assert status.json()["refresh_error"] == "refresh still failed"


def test_refresh_failure_state_survives_app_recreation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    def fail_refresh(*args, **kwargs):
        raise RuntimeError("refresh persisted failure")

    real_build = __import__(
        "src.api.app",
        fromlist=["build_demo_orchestrator"],
    ).build_demo_orchestrator
    monkeypatch.setattr("src.api.app.build_demo_orchestrator", fail_refresh)
    response = client.post(
        "/knowledge/documents/upload",
        files={"file": ("cooling.md", b"# Cooling SOP\n\nCooling alarm response.", "text/markdown")},
    )
    assert response.json()["refresh_dirty"] is True

    monkeypatch.setattr("src.api.app.build_demo_orchestrator", real_build)
    recreated = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    monkeypatch.setattr("src.api.app.build_demo_orchestrator", fail_refresh)
    status = TestClient(recreated).get("/knowledge/status")

    assert status.status_code == 200
    assert status.json()["refresh_dirty"] is True
    assert status.json()["refresh_error"] == "refresh persisted failure"


def test_delete_returns_refresh_error_when_orchestrator_refresh_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)
    uploaded = client.post(
        "/knowledge/documents/upload",
        files={"file": ("ops.txt", b"employee cooling procedure", "text/plain")},
    ).json()

    def fail_refresh(*args, **kwargs):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr("src.api.app.build_demo_orchestrator", fail_refresh)

    response = client.delete(f"/knowledge/documents/{uploaded['document']['document_id']}")

    assert response.status_code == 200
    assert response.json()["deleted"] == uploaded["document"]["document_id"]
    assert response.json()["refresh_error"] == "refresh failed"
    assert client.get("/knowledge/documents").json()["documents"] == []


def test_reindex_returns_refresh_error_when_orchestrator_refresh_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)
    client.post(
        "/knowledge/documents/upload",
        files={"file": ("ops.txt", b"employee cooling procedure", "text/plain")},
    )

    def fail_refresh(*args, **kwargs):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr("src.api.app.build_demo_orchestrator", fail_refresh)

    response = client.post("/knowledge/reindex")

    assert response.status_code == 200
    assert response.json()["index"]["available"] is True
    assert response.json()["refresh_error"] == "refresh failed"


def test_reindex_response_includes_index_metadata_error(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)
    client.post(
        "/knowledge/documents/upload",
        files={"file": ("ops.txt", b"employee cooling procedure", "text/plain")},
    )

    def fail_save_status(*args, **kwargs):
        raise RuntimeError("index metadata failed")

    monkeypatch.setattr("src.knowledge.storage.KnowledgeBaseStore.save_index_status", fail_save_status)

    response = client.post("/knowledge/reindex")

    assert response.status_code == 200
    assert response.json()["index"]["available"] is True
    assert response.json()["index"]["metadata_error"] == "index metadata failed"


def test_concurrent_knowledge_mutations_serialize_faiss_rebuilds(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    def upload(index: int) -> int:
        response = client.post(
            "/knowledge/documents/upload",
            files={
                "file": (
                    f"ops_{index}.md",
                    f"# Ops {index}\n\nCooling procedure {index}.".encode("utf-8"),
                    "text/markdown",
                )
            },
        )
        return response.status_code

    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(executor.map(upload, range(4)))

    status = client.get("/knowledge/status").json()
    listing = client.get("/knowledge/documents").json()

    assert statuses == [200, 200, 200, 200]
    assert status["index"]["available"] is True
    assert status["chunk_count"] == 4
    assert len(listing["documents"]) == 4


def test_get_document_returns_uploaded_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)
    uploaded = client.post(
        "/knowledge/documents/upload",
        files={"file": ("ops.md", b"# Ops\n\nCooling procedure.", "text/markdown")},
    ).json()

    document_id = uploaded["document"]["document_id"]
    response = client.get(f"/knowledge/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["document"]["filename"] == "ops.md"


def test_status_and_list_do_not_initialize_embedding_provider(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "sentence-transformers")

    def fail_provider(*args, **kwargs):
        raise RuntimeError("embedding provider should not load")

    monkeypatch.setattr(
        "src.knowledge.service.SentenceTransformerEmbeddingProvider",
        fail_provider,
    )
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    status = client.get("/knowledge/status")
    listing = client.get("/knowledge/documents")

    assert status.status_code == 200
    assert status.json()["document_count"] == 0
    assert listing.status_code == 200
    assert listing.json()["documents"] == []


def test_status_with_existing_index_does_not_initialize_embedding_provider(tmp_path: Path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    KnowledgeFaissIndexer(
        index_dir=knowledge_dir / "faiss",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    ).rebuild(
        [
            KnowledgeChunk(
                chunk_id="doc_a::chunk_0000",
                document_id="doc_a",
                chunk_index=0,
                text="cooling alarm response",
                page_number=1,
                section_title="Ops",
                token_count=3,
                metadata={"filename": "doc_a.md"},
            )
        ]
    )
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(knowledge_dir))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "sentence-transformers")

    def fail_provider(*args, **kwargs):
        raise RuntimeError("embedding provider should not load")

    monkeypatch.setattr(
        "src.knowledge.service.SentenceTransformerEmbeddingProvider",
        fail_provider,
    )
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)

    status = client.get("/knowledge/status")

    assert status.status_code == 200
    assert status.json()["index"]["available"] is True
    assert status.json()["index"]["chunk_count"] == 1


def test_ask_uses_uploaded_knowledge_without_restart(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    app = create_app(
        use_env_answer_generator=False,
        use_env_intent_classifier=False,
        use_dropt_policy=False,
    )
    client = TestClient(app)
    client.post(
        "/knowledge/documents/upload",
        files={
            "file": (
                "custom_sop.md",
                b"# Custom SOP\n\nBlue coolant valve inspection is required.",
                "text/markdown",
            )
        },
    )

    response = client.post(
        "/ask",
        json={
            "question": "What does the custom SOP say about blue coolant valve inspection?",
            "task_type": "document_qa",
            "workflow_engine": "deterministic",
            "memory_enabled": False,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert any("Blue coolant valve" in context["text"] for context in body["retrieved_contexts"])
    assert any(citation["title"] == "custom_sop.md" for citation in body["citations"])


def test_uploaded_knowledge_retriever_exposes_chunks_for_eval_comparison(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_PROVIDER", "deterministic")
    service = __import__(
        "src.knowledge.service",
        fromlist=["KnowledgeBaseService"],
    ).KnowledgeBaseService(
        knowledge_dir=tmp_path / "knowledge",
        embedding_provider=DeterministicHashEmbeddingProvider(),
        embedding_provider_name="deterministic",
        embedding_model="hash",
    )
    source = tmp_path / "cooling.md"
    source.write_text("# Cooling SOP\n\nBlue coolant valve inspection is required.", encoding="utf-8")
    service.ingest_existing_file(source)

    from src.api.demo_factory import build_demo_orchestrator

    orchestrator = build_demo_orchestrator(
        use_env_answer_generator=False,
        use_dropt_policy=False,
    )
    retriever = orchestrator.rag_pipeline.retriever

    assert len(retriever.chunks) >= 1
    assert retriever.chunks[0].metadata.title == "cooling.md"
