from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.agent.orchestrator import BaselineOrchestrator
from src.api.demo_factory import build_demo_orchestrator
from src.evaluation.runner import (
    build_dense_rag_pipeline,
    run_baseline_comparison,
    run_baseline_eval,
    save_predictions_jsonl,
)
from src.evaluation.llm_judge import DeterministicKeywordJudge
from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.retriever import KeywordRetriever


def write_small_eval_dataset(path: Path) -> Path:
    records = [
        {
            "id": "doc_qa_015",
            "question": "rack delta-t return differential alarm evidence 应该看哪些证据？",
            "task_type": "document_qa",
            "gold_answer": "应引用 rack_delta_t_short_note，说明需要比较 supply return temperature delta、zone_temperature 和 recent control_action。",
            "required_tools": [],
            "required_documents": ["rack_delta_t_short_note"],
            "expected_keywords": ["rack delta-t", "return differential", "control_action"],
            "expected_output_format": "answer_with_citations",
        },
        {
            "id": "ts_query_001",
            "question": "episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？",
            "task_type": "timeseries_query",
            "gold_answer": "应调用 query_metric 并返回 zone_temperature 的最大值。",
            "required_tools": ["query_metric"],
            "required_documents": [],
            "expected_keywords": ["zone_temperature", "最大值"],
            "expected_output_format": "structured_tool_result",
        },
        {
            "id": "policy_001",
            "question": "如果当前温度超过舒适上限，是否应该调整控制策略？",
            "task_type": "policy_recommendation",
            "gold_answer": "应调用 rule_based_policy，不能伪造 DiffFNO 效果。",
            "required_tools": ["rule_based_policy"],
            "required_documents": [],
            "expected_keywords": ["rule_based_policy", "不能伪造"],
            "expected_output_format": "recommendation_with_policy_result",
        },
    ]
    path.write_text(
        "\n".join(__import__("json").dumps(record, ensure_ascii=False) for record in records)
        + "\n",
        encoding="utf-8",
    )
    return path


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


def test_run_baseline_eval_returns_predictions_and_metrics(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    result = run_baseline_eval(
        eval_path=eval_path,
        orchestrator=mock_orchestrator(),
    )

    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["answer_audit"]["passed"] is True
    assert "citation_hit_rate" in result["metrics"]
    assert "context_recall" in result["metrics"]
    assert "expected_keyword_coverage" in result["metrics"]
    assert "lexical_answer_coverage" in result["metrics"]
    assert "tool_selection_accuracy" in result["metrics"]
    assert result["metrics"]["tool_selection_accuracy"] == 1.0
    assert "llm_judge_correctness" not in result["metrics"]


def test_run_baseline_eval_can_add_optional_llm_judge_metrics(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    result = run_baseline_eval(
        eval_path=eval_path,
        orchestrator=mock_orchestrator(),
        llm_judge=DeterministicKeywordJudge(),
    )

    assert "llm_judge_correctness" in result["metrics"]
    assert "llm_judge_faithfulness" in result["metrics"]
    assert result["predictions"][0]["llm_judge"]["judge_name"] == "deterministic_keyword_judge"


def test_run_baseline_comparison_returns_three_named_modes(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    result = run_baseline_comparison(
        eval_path=eval_path,
        orchestrator=mock_orchestrator(),
    )

    assert [run["mode"] for run in result["runs"]] == [
        "llm_only",
        "rag_keyword",
        "rag_dense",
        "rag_hybrid",
        "rag_hybrid_rerank",
        "rag",
        "rag_tool_agent",
    ]
    assert set(result["summary"]) == {
        "llm_only",
        "rag_keyword",
        "rag_dense",
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


def test_demo_baseline_comparison_shows_hybrid_retrieval_context_gain(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    result = run_baseline_comparison(
        eval_path=eval_path,
        orchestrator=build_demo_orchestrator(use_env_answer_generator=False),
    )

    assert (
        result["summary"]["rag_hybrid"]["context_recall"]
        > result["summary"]["rag_keyword"]["context_recall"]
    )


def test_save_predictions_jsonl_writes_records(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    result = run_baseline_eval(
        eval_path=eval_path,
        orchestrator=mock_orchestrator(),
    )
    output_path = tmp_path / "predictions.jsonl"

    save_predictions_jsonl(result["predictions"], output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert '"id": "doc_qa_015"' in lines[0]


def test_run_eval_script_can_be_executed_directly(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    output_path = tmp_path / "script_predictions.jsonl"
    comparison_path = tmp_path / "baseline_comparison.json"
    report_path = tmp_path / "experiment_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--eval-path",
            str(eval_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert output_path.exists()
    assert comparison_path.exists()
    assert report_path.exists()
    review_sample_path = tmp_path / "human_review_sample.jsonl"
    review_annotations_path = tmp_path / "human_review_annotations.jsonl"
    assert review_sample_path.exists()
    assert review_annotations_path.exists()
    annotation_content = review_annotations_path.read_text(encoding="utf-8")
    assert '"correctness_score": null' in annotation_content
    assert '"faithfulness_score": null' in annotation_content
    assert "Saved predictions to" in completed.stdout


def test_run_eval_script_can_write_comparison_summary(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    output_path = tmp_path / "script_predictions.jsonl"
    comparison_path = tmp_path / "comparison_summary.json"
    report_path = tmp_path / "experiment_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--eval-path",
            str(eval_path),
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


def test_run_eval_script_can_enable_optional_llm_judge(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    output_path = tmp_path / "script_predictions.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--eval-path",
            str(eval_path),
            "--output",
            str(output_path),
            "--enable-llm-judge",
            "--llm-judge-provider",
            "deterministic",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert "llm_judge_correctness" in completed.stdout
    assert '"llm_judge"' in output_path.read_text(encoding="utf-8")


def test_run_eval_script_reports_missing_real_dense_dependencies(tmp_path: Path):
    eval_path = write_small_eval_dataset(tmp_path / "small_eval.jsonl")
    output_path = tmp_path / "script_predictions.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--eval-path",
            str(eval_path),
            "--output",
            str(output_path),
            "--dense-provider",
            "sentence-transformers",
            "--dense-backend",
            "faiss",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        assert completed.returncode != 0
        assert "pip install -e" in completed.stderr
    else:
        assert completed.returncode == 0
        assert output_path.exists()


def test_build_dense_rag_pipeline_can_request_faiss_sentence_transformer_backend():
    chunks = getattr(mock_orchestrator().rag_pipeline.retriever, "chunks", [])

    try:
        rag = build_dense_rag_pipeline(
            chunks,
            provider="sentence-transformers",
            backend="faiss",
        )
    except ImportError as exc:
        assert "pip install -e" in str(exc)
    else:
        assert rag.retriever.search("cooling", top_k=1)
