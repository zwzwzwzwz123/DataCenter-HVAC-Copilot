from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.agent.langgraph_workflow import LangGraphOrchestrator
from src.api.demo_factory import build_demo_orchestrator
from src.agent.planner import DeterministicRoutePlanner, build_route_planner_from_env
from src.api.schemas import AskRequest, AskResponse, EvalRunRequest, EvalRunResponse
from src.evaluation.runner import run_baseline_eval
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

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "datacenter-hvac-copilot",
            "data_source": orchestrator.data_source,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> dict:
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
            try:
                if session_id is None:
                    session = manager.create_session(title=_session_title(request.question))
                    session_id = session.session_id
                else:
                    manager.store.require_session(session_id)
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
            except UnknownSessionError as exc:
                raise HTTPException(status_code=404, detail=f"Unknown session_id: {exc.args[0]}") from exc
            except Exception as exc:
                memory_status.update(
                    {
                        "storage": {"available": True, "db_path": str(manager.store.db_path)},
                        "retrieval": {"available": False, "error": str(exc)},
                    }
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
                saved = manager.store.save_turn(
                    ConversationTurn(
                        session_id=session_id,
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
                memory_status["storage"] = {
                    **memory_status.get("storage", {"available": True}),
                    "saved": True,
                }
                try:
                    manager.index_turn(saved)
                    memory_status["indexing"] = {"saved": True}
                except Exception as exc:
                    memory_status["indexing"] = {
                        "saved": False,
                        "error": str(exc),
                    }
                workflow_trace.append(
                    {
                        "node": "memory_turn_saved",
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "turn_index": saved.turn_index,
                    }
                )
            except Exception as exc:
                memory_status["storage"] = {
                    **memory_status.get("storage", {}),
                    "available": False,
                    "saved": False,
                    "error": str(exc),
                }

        return {
            **result,
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

    def _get_context_manager() -> ContextManager:
        nonlocal context_manager
        if context_manager is None:
            context_manager = ContextManager()
        return context_manager

    return app


app = create_app()


def _session_title(question: str) -> str:
    text = " ".join(question.split())
    return text[:80] if text else "New HVAC analysis session"
