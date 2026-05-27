from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.agent.langgraph_workflow import LangGraphOrchestrator
from src.api.demo_factory import build_demo_orchestrator
from src.agent.planner import DeterministicRoutePlanner, build_route_planner_from_env
from src.api.schemas import (
    AskRequest,
    AskResponse,
    EvalRunRequest,
    EvalRunResponse,
    KnowledgeDeleteResponse,
    KnowledgeDocumentResponse,
    KnowledgeDocumentListResponse,
    KnowledgeStatusResponse,
    KnowledgeUploadResponse,
)
from src.evaluation.runner import run_baseline_eval
from src.knowledge.service import KnowledgeBaseService
from src.memory.context_manager import ContextManager, memory_enabled_from_env
from src.memory.schemas import ConversationTurn
from src.memory.storage import UnknownSessionError


def create_app(
    use_env_answer_generator: bool = True,
    use_env_intent_classifier: bool = True,
    use_dropt_policy: bool = True,
) -> FastAPI:
    app = FastAPI(title="DataCenter-HVAC Copilot", version="0.1.0")
    orchestrator = build_demo_orchestrator(
        use_env_answer_generator=use_env_answer_generator,
        use_dropt_policy=use_dropt_policy,
    )
    langgraph_orchestrator = LangGraphOrchestrator(
        orchestrator,
        route_planner=(
            build_route_planner_from_env()
            if use_env_intent_classifier
            else DeterministicRoutePlanner()
        ),
    )
    context_manager: ContextManager | None = None
    knowledge_service: KnowledgeBaseService | None = None
    persisted_refresh_state = _load_refresh_state(_refresh_state_path())
    knowledge_refresh_dirty = bool(persisted_refresh_state.get("refresh_dirty", False))
    last_refresh_error = persisted_refresh_state.get("refresh_error")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "datacenter-hvac-copilot",
            "data_source": orchestrator.data_source,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> dict:
        _try_refresh_dirty_knowledge()
        memory_enabled = bool(request.memory_enabled and memory_enabled_from_env())
        session_id = request.session_id
        conversation_context: dict[str, Any] = {}
        memory_status: dict[str, Any] = {"enabled": memory_enabled}
        memory_trace: list[dict[str, Any]] = []
        manager = None
        if memory_enabled:
            try:
                manager = _get_context_manager()
            except Exception as exc:
                memory_status.update(
                    {
                        "storage": {"available": False, "error": str(exc)},
                        "retrieval": {"available": False, "error": str(exc)},
                    }
                )

        if manager is not None:
            if session_id is None:
                try:
                    session = manager.create_session(title=_session_title(request.question))
                    session_id = session.session_id
                except Exception as exc:
                    manager = None
                    memory_status.update(
                        {
                            "storage": {"available": False, "error": str(exc)},
                            "retrieval": {
                                "available": False,
                                "error": "memory storage unavailable",
                            },
                        }
                    )
            else:
                try:
                    manager.store.require_session(session_id)
                except UnknownSessionError as exc:
                    raise HTTPException(status_code=404, detail=f"Unknown session_id: {exc.args[0]}") from exc
                except Exception as exc:
                    manager = None
                    memory_status.update(
                        {
                            "storage": {"available": False, "error": str(exc)},
                            "retrieval": {
                                "available": False,
                                "error": "memory storage unavailable",
                            },
                        }
                    )
            if manager is not None:
                try:
                    loaded = manager.load_context(session_id, request.question)
                    conversation_context = loaded.to_dict()
                    memory_status.update(loaded.memory_status)
                    memory_trace.append(
                        {
                            "node": "memory_context_loaded",
                            "session_id": session_id,
                            "retrieved_memory_count": len(loaded.relevant_memory),
                            "budget_truncated": bool(loaded.budget.get("truncated", False)),
                        }
                    )
                    memory_trace.append(
                        {
                            "node": "memory_retrieval",
                            "backend": loaded.memory_status.get("retrieval", {}).get("backend"),
                            "available": loaded.memory_status.get("retrieval", {}).get("available"),
                            "fallback_used": loaded.memory_status.get("retrieval", {}).get("fallback_used", False),
                            "retrieved_memory_count": len(loaded.relevant_memory),
                        }
                    )
                except sqlite3.Error as exc:
                    memory_status.update(
                        {
                            "storage": {"available": False, "error": str(exc)},
                            "retrieval": {
                                "available": False,
                                "error": "memory storage unavailable",
                            },
                        }
                    )
                    memory_trace.extend(
                        _memory_context_failure_trace(
                            session_id=session_id,
                            storage_available=False,
                            storage_error=str(exc),
                            retrieval_error="memory storage unavailable",
                        )
                    )
                except Exception as exc:
                    memory_status.update(
                        {
                            "storage": {"available": True, "db_path": str(manager.store.db_path)},
                            "retrieval": {"available": False, "error": str(exc)},
                        }
                    )
                    memory_trace.extend(
                        _memory_context_failure_trace(
                            session_id=session_id,
                            storage_available=True,
                            retrieval_error=str(exc),
                        )
                    )

        if request.workflow_engine == "deterministic":
            result = orchestrator.run(
                request.question,
                task_type=request.task_type,
                conversation_context=conversation_context or None,
            )
            result = {
                **result,
                "workflow_engine": "deterministic",
                "workflow_trace": [],
            }
        elif request.workflow_engine == "langgraph":
            result = langgraph_orchestrator.run(
                request.question,
                task_type=request.task_type,
                conversation_context=conversation_context or None,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="workflow_engine must be one of: deterministic, langgraph",
            )

        workflow_trace = [*memory_trace, *result.get("workflow_trace", [])]
        turn_id = None
        if manager is not None and session_id is not None:
            try:
                pending_turn_id = f"turn_{uuid.uuid4().hex}"
                turn_saved_trace = {
                    "node": "memory_turn_saved",
                    "session_id": session_id,
                    "turn_id": pending_turn_id,
                    "indexing_saved": False,
                    "trace_persisted": False,
                }
                workflow_trace.append(turn_saved_trace)
                saved = manager.store.save_turn(
                    ConversationTurn(
                        session_id=session_id,
                        turn_id=pending_turn_id,
                        question=request.question,
                        answer=str(result.get("answer", "")),
                        route=str(result.get("route", "")),
                        tools=list(result.get("tools", [])),
                        citations=list(result.get("citations", [])),
                        retrieved_contexts=list(result.get("retrieved_contexts", [])),
                        tool_results=list(result.get("tool_results", [])),
                        policy_result=dict(result.get("policy_result", {})),
                        workflow_trace=workflow_trace,
                        answer_audit=dict(result.get("answer_audit", {})),
                        data_source=dict(result.get("data_source", {})),
                        memory_context=conversation_context,
                    )
                )
                turn_id = saved.turn_id
                turn_saved_trace["turn_index"] = saved.turn_index
                memory_status["storage"] = {
                    **memory_status.get("storage", {"available": True}),
                    "saved": True,
                }
                try:
                    manager.index_turn(saved)
                    memory_status["indexing"] = {"saved": True}
                    indexing_saved = True
                    indexing_error = None
                except Exception as exc:
                    memory_status["indexing"] = {
                        "saved": False,
                        "error": str(exc),
                    }
                    indexing_saved = False
                    indexing_error = str(exc)
                turn_saved_trace["indexing_saved"] = indexing_saved
                if indexing_error:
                    turn_saved_trace["indexing_error"] = indexing_error
                try:
                    turn_saved_trace["trace_persisted"] = True
                    manager.store.update_turn_workflow_trace(turn_id, workflow_trace)
                    memory_status["trace_persistence"] = {"saved": True}
                except Exception as exc:
                    memory_status["trace_persistence"] = {
                        "saved": False,
                        "error": str(exc),
                    }
                    turn_saved_trace["trace_persisted"] = False
                    turn_saved_trace["trace_persistence_error"] = str(exc)
            except Exception as exc:
                workflow_trace = [
                    step
                    for step in workflow_trace
                    if step.get("node") != "memory_turn_saved"
                ]
                turn_id = None
                memory_status["storage"] = {
                    **memory_status.get("storage", {}),
                    "available": False,
                    "saved": False,
                    "error": str(exc),
                }

        return {
            **result,
            **_current_refresh_state(),
            "session_id": session_id if memory_enabled else None,
            "turn_id": turn_id,
            "memory_status": memory_status,
            "conversation_context": conversation_context,
            "workflow_trace": workflow_trace,
        }

    @app.post("/eval/run", response_model=EvalRunResponse)
    def eval_run(request: EvalRunRequest) -> dict:
        eval_orchestrator = build_demo_orchestrator(
            use_env_answer_generator=False,
            use_dropt_policy=False,
        )
        return run_baseline_eval(request.eval_path, eval_orchestrator)

    @app.post("/knowledge/documents/upload", response_model=KnowledgeUploadResponse)
    def upload_knowledge_document(file: Annotated[UploadFile, File()]) -> dict:
        safe_filename = _safe_upload_filename(file.filename or "")
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in {".md", ".txt", ".pdf", ".docx"}:
            raise HTTPException(
                status_code=400,
                detail="Supported file types: .md, .txt, .pdf, .docx",
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / (safe_filename or f"upload{suffix}")
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            result = _get_knowledge_service().ingest_existing_file(tmp_path)
            _mark_knowledge_refresh_dirty()
            _attach_refresh_status(result)
            return result

    @app.get("/knowledge/documents", response_model=KnowledgeDocumentListResponse)
    def list_knowledge_documents() -> dict:
        return {"documents": _get_knowledge_service().list_documents()}

    @app.get("/knowledge/documents/{document_id}", response_model=KnowledgeDocumentResponse)
    def get_knowledge_document(document_id: str) -> dict:
        document = _get_knowledge_service().get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail=f"Unknown document_id: {document_id}")
        return {"document": document}

    @app.get("/knowledge/status", response_model=KnowledgeStatusResponse)
    def knowledge_status() -> dict:
        _try_refresh_dirty_knowledge()
        result = _get_knowledge_service().status()
        _attach_refresh_state(result)
        return result

    @app.post("/knowledge/reindex", response_model=KnowledgeStatusResponse)
    def reindex_knowledge() -> dict:
        reindex_result = _get_knowledge_service().reindex()
        result = _get_knowledge_service().status()
        _merge_index_rebuild_metadata(result, reindex_result)
        _mark_knowledge_refresh_dirty()
        _attach_refresh_status(result)
        return result

    @app.delete("/knowledge/documents/{document_id}", response_model=KnowledgeDeleteResponse)
    def delete_knowledge_document(document_id: str) -> dict:
        result = _get_knowledge_service().delete_document(document_id)
        _mark_knowledge_refresh_dirty()
        _attach_refresh_status(result)
        return result

    def _get_context_manager() -> ContextManager:
        nonlocal context_manager
        if context_manager is None:
            context_manager = ContextManager()
        return context_manager

    def _get_knowledge_service() -> KnowledgeBaseService:
        nonlocal knowledge_service
        if knowledge_service is None:
            knowledge_service = KnowledgeBaseService(
                knowledge_dir=_knowledge_dir(),
                embedding_provider_name=os.getenv(
                    "KNOWLEDGE_EMBEDDING_PROVIDER",
                    "sentence-transformers",
                ),
                embedding_model=os.getenv(
                    "KNOWLEDGE_EMBEDDING_MODEL",
                    "BAAI/bge-small-zh-v1.5",
                ),
            )
        return knowledge_service

    def _refresh_orchestrators() -> None:
        nonlocal orchestrator, langgraph_orchestrator
        new_orchestrator = build_demo_orchestrator(
            use_env_answer_generator=use_env_answer_generator,
            use_dropt_policy=use_dropt_policy,
        )
        new_langgraph_orchestrator = LangGraphOrchestrator(
            new_orchestrator,
            route_planner=(
                build_route_planner_from_env()
                if use_env_intent_classifier
                else DeterministicRoutePlanner()
            ),
        )
        orchestrator = new_orchestrator
        langgraph_orchestrator = new_langgraph_orchestrator

    def _mark_knowledge_refresh_dirty() -> None:
        nonlocal knowledge_refresh_dirty, last_refresh_error
        knowledge_refresh_dirty = True
        _persist_refresh_state(
            _refresh_state_path(),
            refresh_dirty=knowledge_refresh_dirty,
            refresh_error=last_refresh_error,
        )

    def _attach_refresh_status(result: dict) -> None:
        nonlocal knowledge_refresh_dirty, last_refresh_error
        try:
            _refresh_orchestrators()
            knowledge_refresh_dirty = False
            last_refresh_error = None
        except Exception as exc:
            knowledge_refresh_dirty = True
            last_refresh_error = str(exc)
        _persist_refresh_state(
            _refresh_state_path(),
            refresh_dirty=knowledge_refresh_dirty,
            refresh_error=last_refresh_error,
        )
        _attach_refresh_state(result)

    def _try_refresh_dirty_knowledge() -> None:
        if knowledge_refresh_dirty:
            _attach_refresh_status({})

    def _attach_refresh_state(result: dict) -> None:
        result.update(_current_refresh_state())

    def _current_refresh_state() -> dict[str, Any]:
        state: dict[str, Any] = {"refresh_dirty": knowledge_refresh_dirty}
        if last_refresh_error is not None:
            state["refresh_error"] = last_refresh_error
        return state

    def _merge_index_rebuild_metadata(result: dict, reindex_result: dict) -> None:
        index_status = result.setdefault("index", {})
        if "metadata_error" in reindex_result:
            index_status["metadata_error"] = reindex_result["metadata_error"]

    return app


def _session_title(question: str) -> str:
    text = " ".join(question.split())
    return text[:80] if text else "New HVAC analysis session"


def _memory_context_failure_trace(
    *,
    session_id: str | None,
    storage_available: bool,
    retrieval_error: str,
    storage_error: str | None = None,
) -> list[dict[str, Any]]:
    context_node = {
        "node": "memory_context_loaded",
        "session_id": session_id,
        "available": False,
        "storage_available": storage_available,
        "error": storage_error or retrieval_error,
    }
    retrieval_node = {
        "node": "memory_retrieval",
        "available": False,
        "error": retrieval_error,
        "retrieved_memory_count": 0,
    }
    if storage_error is not None:
        retrieval_node["storage_error"] = storage_error
    return [context_node, retrieval_node]


def _safe_upload_filename(filename: str) -> str:
    basename = filename.replace("\\", "/").split("/")[-1]
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in basename)


def _knowledge_dir() -> Path:
    return Path(os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge"))


def _refresh_state_path() -> Path:
    return _knowledge_dir() / "refresh_state.json"


def _load_refresh_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _persist_refresh_state(
    path: Path,
    *,
    refresh_dirty: bool,
    refresh_error: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"refresh_dirty": refresh_dirty}
    if refresh_error is not None:
        payload["refresh_error"] = refresh_error
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


app = create_app()
