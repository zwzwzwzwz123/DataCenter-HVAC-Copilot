from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    task_type: str | None = None
    workflow_engine: str = "langgraph"
    session_id: str | None = None
    memory_enabled: bool = True


class AskResponse(BaseModel):
    question: str
    route: str
    answer: str
    answer_generator: str | None = None
    answer_audit: dict = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    retrieved_contexts: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    policy_result: dict = Field(default_factory=dict)
    route_reason: str | None = None
    data_source: dict[str, str] = Field(default_factory=dict)
    workflow_engine: str = "langgraph"
    workflow_trace: list[dict] = Field(default_factory=list)
    session_id: str | None = None
    turn_id: str | None = None
    memory_status: dict = Field(default_factory=dict)
    conversation_context: dict = Field(default_factory=dict)


class EvalRunRequest(BaseModel):
    eval_path: str = "data/eval/hvac_eval.jsonl"


class EvalRunResponse(BaseModel):
    metrics: dict[str, float]
    predictions: list[dict]
