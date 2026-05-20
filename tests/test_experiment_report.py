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
        },
        "rag_keyword": {
            "citation_hit_rate": 0.4,
            "context_recall": 0.5,
            "expected_keyword_coverage": 0.4,
            "lexical_answer_coverage": 0.2,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.16666666666666666,
        },
        "rag_hybrid": {
            "citation_hit_rate": 0.5,
            "context_recall": 0.75,
            "expected_keyword_coverage": 0.5,
            "lexical_answer_coverage": 0.25,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.16666666666666666,
        },
        "rag_hybrid_rerank": {
            "citation_hit_rate": 0.5,
            "context_recall": 0.75,
            "expected_keyword_coverage": 0.5,
            "lexical_answer_coverage": 0.25,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.16666666666666666,
        },
        "rag_tool_agent": {
            "citation_hit_rate": 0.5,
            "context_recall": 0.75,
            "expected_keyword_coverage": 0.8,
            "lexical_answer_coverage": 0.45,
            "tool_selection_accuracy": 1.0,
            "tool_execution_success_rate": 1.0,
            "evidence_coverage": 0.8666666666666667,
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
                },
                "timeseries_query": {
                    "citation_hit_rate": 0.0,
                    "context_recall": 0.0,
                    "expected_keyword_coverage": 0.7,
                    "lexical_answer_coverage": 0.2,
                    "tool_selection_accuracy": 1.0,
                    "tool_execution_success_rate": 1.0,
                    "evidence_coverage": 1.0,
                },
            }
        },
    )

    assert markdown.startswith("# 实验报告")
    assert "BEAR 仿真轨迹" in markdown
    assert "12 条样例包含人工维护的 expected_keywords" in markdown
    assert "| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage |" in markdown
    assert "| rag_tool_agent | 0.500 | 0.750 | 0.800 | 0.450 | 1.000 | 1.000 | 0.867 |" in markdown
    assert "当前结论" in markdown
    assert "rag_hybrid` 在 citation/context 指标上优于 `rag_keyword" in markdown
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
            }
        },
        output_path=output_path,
        eval_record_count=30,
        expected_keyword_record_count=1,
    )

    content = output_path.read_text(encoding="utf-8")
    assert "# 实验报告" in content
    assert "llm_only" in content
