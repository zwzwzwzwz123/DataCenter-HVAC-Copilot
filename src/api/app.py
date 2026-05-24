from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.agent.langgraph_workflow import LangGraphOrchestrator
from src.api.demo_factory import build_demo_orchestrator
from src.agent.planner import DeterministicRoutePlanner, build_route_planner_from_env
from src.api.schemas import AskRequest, AskResponse, EvalRunRequest, EvalRunResponse
from src.evaluation.runner import run_baseline_eval


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

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "datacenter-hvac-copilot",
            "data_source": orchestrator.data_source,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> dict:
        if request.workflow_engine == "deterministic":
            result = orchestrator.run(request.question, task_type=request.task_type)
            return {
                **result,
                "workflow_engine": "deterministic",
                "workflow_trace": [],
            }
        if request.workflow_engine == "langgraph":
            return langgraph_orchestrator.run(request.question, task_type=request.task_type)
        raise HTTPException(
            status_code=400,
            detail="workflow_engine must be one of: deterministic, langgraph",
        )

    @app.post("/eval/run", response_model=EvalRunResponse)
    def eval_run(request: EvalRunRequest) -> dict:
        eval_orchestrator = build_demo_orchestrator(
            use_env_answer_generator=False,
            use_dropt_policy=False,
        )
        return run_baseline_eval(request.eval_path, eval_orchestrator)

    return app


app = create_app()
