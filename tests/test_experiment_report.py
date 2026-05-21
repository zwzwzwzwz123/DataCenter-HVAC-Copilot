from pathlib import Path

from src.evaluation.report import render_experiment_report, save_experiment_report


def test_render_experiment_report_creates_markdown_table():
    comparison = {
        "llm_only": {
            "citation_hit_rate": 0.0,
            "context_recall": 0.0,
            "expected_keyword_coverage": 0.0,
            "lexical_answer_coverage": 0.0,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.0,
            "answer_correctness_proxy": 0.0,
            "faithfulness_proxy": 0.0,
        },
        "rag_keyword": {
            "citation_hit_rate": 0.4,
            "context_recall": 0.5,
            "expected_keyword_coverage": 0.4,
            "lexical_answer_coverage": 0.2,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.16666666666666666,
            "answer_correctness_proxy": 0.2,
            "faithfulness_proxy": 0.3,
        },
        "rag_hybrid": {
            "citation_hit_rate": 0.5,
            "context_recall": 0.75,
            "expected_keyword_coverage": 0.5,
            "lexical_answer_coverage": 0.25,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.16666666666666666,
            "answer_correctness_proxy": 0.3,
            "faithfulness_proxy": 0.4,
        },
        "rag_hybrid_rerank": {
            "citation_hit_rate": 0.5,
            "context_recall": 0.75,
            "expected_keyword_coverage": 0.5,
            "lexical_answer_coverage": 0.25,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.16666666666666666,
            "answer_correctness_proxy": 0.35,
            "faithfulness_proxy": 0.45,
        },
        "rag_tool_agent": {
            "citation_hit_rate": 0.5,
            "context_recall": 0.75,
            "expected_keyword_coverage": 0.8,
            "lexical_answer_coverage": 0.45,
            "tool_selection_accuracy": 1.0,
            "tool_execution_success_rate": 1.0,
            "evidence_coverage": 0.8666666666666667,
            "answer_correctness_proxy": 0.6,
            "faithfulness_proxy": 0.7,
        },
    }

    markdown = render_experiment_report(
        comparison,
        eval_record_count=30,
        expected_keyword_record_count=12,
        by_task_type={
            "rag_tool_agent": {
                "document_qa": {
                    "citation_hit_rate": 0.5,
                    "context_recall": 0.75,
                    "expected_keyword_coverage": 0.8,
                    "lexical_answer_coverage": 0.45,
                    "tool_selection_accuracy": 0.0,
                    "tool_execution_success_rate": 0.0,
                    "evidence_coverage": 0.5,
                    "answer_correctness_proxy": 0.6,
                    "faithfulness_proxy": 0.7,
                },
                "timeseries_query": {
                    "citation_hit_rate": 0.0,
                    "context_recall": 0.0,
                    "expected_keyword_coverage": 0.7,
                    "lexical_answer_coverage": 0.2,
                    "tool_selection_accuracy": 1.0,
                    "tool_execution_success_rate": 1.0,
                    "evidence_coverage": 1.0,
                    "answer_correctness_proxy": 0.4,
                    "faithfulness_proxy": 0.5,
                },
            }
        },
    )

    assert markdown.startswith("# 实验报告")
    assert "BEAR 仿真轨迹" in markdown
    assert "12 条样例包含人工维护的 expected_keywords" in markdown
    assert "answer_correctness_proxy" in markdown
    assert "faithfulness_proxy" in markdown
    assert "| rag_tool_agent | 0.500 | 0.750 | 0.800 | 0.450 | 1.000 | 1.000 | 0.867 | 0.600 | 0.700 |" in markdown
    assert "当前结论" in markdown
    assert "rag_hybrid` 在 citation/context 指标上优于 `rag_keyword" in markdown
    assert "rag_dense" in markdown
    assert "rag_hybrid_rerank" in markdown
    assert "## 按任务类型指标" in markdown
    assert "| rag_tool_agent | document_qa | 0.500 | 0.750" in markdown
    assert "| rag_tool_agent | timeseries_query | 0.000 | 0.000" in markdown


def test_save_experiment_report_writes_utf8_markdown(tmp_path: Path):
    output_path = tmp_path / "experiment_report.md"

    save_experiment_report(
        {
            "llm_only": {
                "citation_hit_rate": 0.0,
                "context_recall": 0.0,
                "expected_keyword_coverage": 0.0,
                "lexical_answer_coverage": 0.0,
                "tool_selection_accuracy": 0.0,
                "tool_execution_success_rate": 0.0,
                "evidence_coverage": 0.0,
                "answer_correctness_proxy": 0.0,
                "faithfulness_proxy": 0.0,
            }
        },
        output_path=output_path,
        eval_record_count=30,
        expected_keyword_record_count=1,
    )

    content = output_path.read_text(encoding="utf-8")
    assert "# 实验报告" in content
    assert "llm_only" in content


def test_render_experiment_report_includes_optional_llm_judge_columns() -> None:
    markdown = render_experiment_report(
        {
            "rag_tool_agent": {
                "citation_hit_rate": 0.5,
                "context_recall": 0.5,
                "expected_keyword_coverage": 0.5,
                "lexical_answer_coverage": 0.5,
                "tool_selection_accuracy": 1.0,
                "tool_execution_success_rate": 1.0,
                "evidence_coverage": 1.0,
                "answer_correctness_proxy": 0.5,
                "faithfulness_proxy": 0.5,
                "llm_judge_correctness": 0.7,
                "llm_judge_faithfulness": 0.8,
            }
        },
        eval_record_count=1,
        expected_keyword_record_count=1,
    )

    assert "llm_judge_correctness" in markdown
    assert "| rag_tool_agent | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.700 | 0.800 |" in markdown


def test_render_experiment_report_includes_pending_human_calibration() -> None:
    markdown = render_experiment_report(
        {
            "rag_tool_agent": {
                "citation_hit_rate": 0.5,
                "context_recall": 0.5,
                "expected_keyword_coverage": 0.5,
                "lexical_answer_coverage": 0.5,
                "tool_selection_accuracy": 1.0,
                "tool_execution_success_rate": 1.0,
                "evidence_coverage": 1.0,
                "answer_correctness_proxy": 0.5,
                "faithfulness_proxy": 0.5,
            }
        },
        eval_record_count=100,
        expected_keyword_record_count=100,
        human_calibration={
            "sample_count": 24,
            "labeled_count": 0,
            "pending_count": 24,
            "mean_correctness": None,
            "mean_faithfulness": None,
            "safety_pass_rate": None,
            "status": "pending_human_review",
        },
    )

    assert "## Human Calibration" in markdown
    assert "pending_human_review" in markdown
    assert "24" in markdown
    assert "不会把 deterministic proxy 或 LLM judge 当作人工评审" in markdown


def test_render_experiment_report_includes_labeled_human_calibration() -> None:
    markdown = render_experiment_report(
        {
            "rag_tool_agent": {
                "citation_hit_rate": 0.5,
                "context_recall": 0.5,
                "expected_keyword_coverage": 0.5,
                "lexical_answer_coverage": 0.5,
                "tool_selection_accuracy": 1.0,
                "tool_execution_success_rate": 1.0,
                "evidence_coverage": 1.0,
                "answer_correctness_proxy": 0.5,
                "faithfulness_proxy": 0.5,
            }
        },
        eval_record_count=100,
        expected_keyword_record_count=100,
        human_calibration={
            "sample_count": 2,
            "labeled_count": 2,
            "pending_count": 0,
            "mean_correctness": 0.75,
            "mean_faithfulness": 0.5,
            "safety_pass_rate": 1.0,
            "status": "complete",
        },
    )

    assert "| sample_count | labeled_count | pending_count | mean_correctness | mean_faithfulness | safety_pass_rate | status |" in markdown
    assert "| 2 | 2 | 0 | 0.750 | 0.500 | 1.000 | complete |" in markdown
