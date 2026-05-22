from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agent.answer_generator import AnswerGeneratorInput, GeneratedAnswer
from src.agent.orchestrator import BaselineOrchestrator
from src.agent.react_agent import ReActOrchestrator
from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.retriever import KeywordRetriever


class SpyAnswerGenerator:
    def __init__(self) -> None:
        self.payloads: list[AnswerGeneratorInput] = []

    def generate(self, payload: AnswerGeneratorInput) -> GeneratedAnswer:
        self.payloads.append(payload)
        return GeneratedAnswer(answer=f"generated:{payload.route}", generator="spy")


def _baseline() -> BaselineOrchestrator:
    document = load_markdown_document(
        Path("data/documents/sample_hvac_guidance.md"),
        source_id="sample_hvac_guidance",
        title="Sample HVAC Guidance",
        published_at="2026",
        category="internal_note",
    )
    chunks = chunk_document(document, chunk_size=45, overlap=5)
    rag = ExtractiveRAGPipeline(KeywordRetriever(chunks))
    trajectory = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "scenario_id": ["episode_001"] * 4,
            "zone_id": ["zone_a"] * 4,
            "zone_temperature": [23.0, 24.0, 30.0, 25.0],
            "cooling_power": [100.0, 110.0, 180.0, 120.0],
            "fan_power": [20.0, 21.0, 30.0, 24.0],
            "hvac_power": [120.0, 131.0, 210.0, 144.0],
            "comfort_violation": [False, False, True, False],
        }
    )
    return BaselineOrchestrator(
        rag_pipeline=rag,
        trajectory=trajectory,
        answer_generator=SpyAnswerGenerator(),
    )


def test_react_orchestrator_emits_two_step_policy_trace() -> None:
    orchestrator = ReActOrchestrator(_baseline())

    result = orchestrator.run(
        "Before policy recommendation, check the latest 3 hour zone_temperature max.",
        task_type="policy_recommendation",
    )

    assert result["workflow_engine"] == "react"
    assert result["route"] == "policy_recommendation"
    assert len(result["react_trace"]) >= 2
    assert result["react_trace"][0]["action"] == "timeseries_query"
    assert result["react_trace"][-1]["action"] == "policy_recommendation"
    assert result["react_trace"][-1]["observation"]["policy_name"] == "rule_based"
    assert result["tools"] == ["query_metric", "rule_based_policy"]
    assert len(result["tool_results"]) == 2
    assert "query_metric" in result["answer"]
    assert "rule_based" in result["answer"]


def test_react_orchestrator_returns_trace_for_document_qa() -> None:
    orchestrator = ReActOrchestrator(_baseline())

    result = orchestrator.run("为什么 PUE-like values 不能编造？", task_type="document_qa")

    assert result["workflow_engine"] == "react"
    assert len(result["react_trace"]) == 1
    assert result["react_trace"][0]["action"] == "document_qa"
    assert result["answer_generator"] == "spy"
