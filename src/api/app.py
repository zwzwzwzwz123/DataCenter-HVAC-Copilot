from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from src.api.demo_factory import build_demo_orchestrator
from src.api.schemas import AskRequest, AskResponse, EvalRunRequest, EvalRunResponse
from src.evaluation.runner import run_baseline_eval


def create_app() -> FastAPI:
    app = FastAPI(title="DataCenter-HVAC Copilot", version="0.1.0")
    orchestrator = build_demo_orchestrator()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "datacenter-hvac-copilot",
            "data_source": orchestrator.data_source,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> dict:
        return orchestrator.run(request.question, task_type=request.task_type)

    @app.post("/eval/run", response_model=EvalRunResponse)
    def eval_run(request: EvalRunRequest) -> dict:
        return run_baseline_eval(request.eval_path, orchestrator)

    return app


app = create_app()
