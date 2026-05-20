from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.agent.orchestrator import BaselineOrchestrator
from src.api.demo_factory import build_demo_orchestrator
from src.evaluation.runner import (
    run_baseline_comparison,
    run_baseline_eval,
    save_predictions_jsonl,
)
from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.retriever import KeywordRetriever


def mock_orchestrator():
    document = load_markdown_document(
        Path("data/documents/sample_hvac_guidance.md"),
        source_id="hvac_energy_reference",
        title="HVAC Energy Reference",
        published_at="2026",
        category="internal_note",
    )
    rag = ExtractiveRAGPipeline(KeywordRetriever(chunk_document(document, chunk_size=45, overlap=5)))
    trajectory = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "scenario_id": ["episode_001"] * 4,
            "zone_id": ["zone_a"] * 4,
            "zone_temperature": [23.0, 24.0, 30.0, 25.0],
            "cooling_power": [100.0, 110.0, 180.0, 120.0],
            "fan_power": [20.0, 21.0, 30.0, 24.0],
        }
    )
    return BaselineOrchestrator(rag_pipeline=rag, trajectory=trajectory)


def test_run_baseline_eval_returns_predictions_and_metrics():
    result = run_baseline_eval(
        eval_path=Path("data/eval/hvac_eval.jsonl"),
        orchestrator=mock_orchestrator(),
    )

    assert len(result["predictions"]) >= 30
    assert "citation_hit_rate" in result["metrics"]
    assert "context_recall" in result["metrics"]
    assert "expected_keyword_coverage" in result["metrics"]
    assert "lexical_answer_coverage" in result["metrics"]
    assert "tool_selection_accuracy" in result["metrics"]
    assert result["metrics"]["tool_selection_accuracy"] == 1.0


def test_run_baseline_comparison_returns_three_named_modes():
    result = run_baseline_comparison(
        eval_path=Path("data/eval/hvac_eval.jsonl"),
        orchestrator=mock_orchestrator(),
    )

    assert [run["mode"] for run in result["runs"]] == [
        "llm_only",
        "rag_keyword",
        "rag_hybrid",
        "rag_hybrid_rerank",
        "rag",
        "rag_tool_agent",
    ]
    assert set(result["summary"]) == {
        "llm_only",
        "rag_keyword",
        "rag_hybrid",
        "rag_hybrid_rerank",
        "rag",
        "rag_tool_agent",
    }
    assert result["summary"]["rag_tool_agent"]["tool_selection_accuracy"] == 1.0
    assert result["summary"]["llm_only"]["tool_selection_accuracy"] == 0.0
    assert "by_task_type" in result
    assert "rag_tool_agent" in result["by_task_type"]
    assert "document_qa" in result["by_task_type"]["rag_tool_agent"]
    assert "timeseries_query" in result["by_task_type"]["rag_tool_agent"]


def test_demo_baseline_comparison_shows_hybrid_retrieval_context_gain():
    result = run_baseline_comparison(
        eval_path=Path("data/eval/hvac_eval.jsonl"),
        orchestrator=build_demo_orchestrator(),
    )

    assert (
        result["summary"]["rag_hybrid"]["context_recall"]
        > result["summary"]["rag_keyword"]["context_recall"]
    )


def test_save_predictions_jsonl_writes_records(tmp_path: Path):
    result = run_baseline_eval(
        eval_path=Path("data/eval/hvac_eval.jsonl"),
        orchestrator=mock_orchestrator(),
    )
    output_path = tmp_path / "predictions.jsonl"

    save_predictions_jsonl(result["predictions"], output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 30
    assert '"id": "doc_qa_001"' in lines[0]


def test_run_eval_script_can_be_executed_directly(tmp_path: Path):
    output_path = tmp_path / "script_predictions.jsonl"
    completed = subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--output", str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert output_path.exists()
    assert "Saved predictions to" in completed.stdout


def test_run_eval_script_can_write_comparison_summary(tmp_path: Path):
    output_path = tmp_path / "script_predictions.jsonl"
    comparison_path = tmp_path / "comparison_summary.json"
    report_path = tmp_path / "experiment_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--output",
            str(output_path),
            "--comparison-output",
            str(comparison_path),
            "--report-output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert comparison_path.exists()
    content = comparison_path.read_text(encoding="utf-8")
    assert '"summary"' in content
    assert '"by_task_type"' in content
    assert '"llm_only"' in content
    assert '"rag"' in content
    assert '"rag_tool_agent"' in content
    assert report_path.exists()
    assert "# 实验报告" in report_path.read_text(encoding="utf-8")
