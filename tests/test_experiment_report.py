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
        "rag_rewrite": {
            "citation_hit_rate": 0.6,
            "context_recall": 0.8,
            "expected_keyword_coverage": 0.55,
            "lexical_answer_coverage": 0.3,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.2,
            "answer_correctness_proxy": 0.4,
            "faithfulness_proxy": 0.5,
        },
        "rag_hyde": {
            "citation_hit_rate": 0.62,
            "context_recall": 0.82,
            "expected_keyword_coverage": 0.57,
            "lexical_answer_coverage": 0.31,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.2,
            "answer_correctness_proxy": 0.42,
            "faithfulness_proxy": 0.52,
        },
        "rag_hyde_rerank": {
            "citation_hit_rate": 0.64,
            "context_recall": 0.84,
            "expected_keyword_coverage": 0.59,
            "lexical_answer_coverage": 0.32,
            "tool_selection_accuracy": 0.0,
            "tool_execution_success_rate": 0.0,
            "evidence_coverage": 0.2,
            "answer_correctness_proxy": 0.44,
            "faithfulness_proxy": 0.54,
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
        "langgraph_tool_agent": {
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
    assert "| rag_tool_agent | 0.500 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.800 | 0.450 | 1.000 | 1.000 | 0.867 | 0.600 | 0.700 | 0.000 | 0.000 |" in markdown
    assert "当前结论" in markdown
    assert "rag_hybrid` 在 citation/context 指标上优于 `rag_keyword" in markdown
    assert "rag_dense" in markdown
    assert "rag_hybrid_rerank" in markdown
    assert "rag_rewrite" in markdown
    assert "rag_hyde" in markdown
    assert "rag_hyde_rerank" in markdown
    assert "Query Rewrite / HyDE" in markdown
    assert "deterministic query expansion" in markdown
    assert "langgraph_tool_agent" in markdown
    assert "StateGraph 编排" in markdown
    assert "LLM route planner" in markdown
    assert "DeepSeek" in markdown
    assert "scripts/run_intent_eval.py" in markdown
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
    assert "| rag_tool_agent | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 | 0.700 | 0.800 |" in markdown


def test_render_experiment_report_documents_real_dense_configuration() -> None:
    markdown = render_experiment_report(
        {
            "rag_dense": {
                "citation_hit_rate": 0.52,
                "context_recall": 0.52,
                "expected_keyword_coverage": 0.38,
                "lexical_answer_coverage": 0.16,
                "tool_selection_accuracy": 0.0,
                "tool_execution_success_rate": 0.0,
                "evidence_coverage": 1.0,
                "answer_correctness_proxy": 0.46,
                "faithfulness_proxy": 0.38,
            }
        },
        eval_record_count=100,
        expected_keyword_record_count=100,
        dense_provider="sentence-transformers",
        dense_backend="faiss",
        dense_model="BAAI/bge-small-zh-v1.5",
    )

    assert "dense_provider: `sentence-transformers`" in markdown
    assert "dense_backend: `faiss`" in markdown
    assert "dense_model: `BAAI/bge-small-zh-v1.5`" in markdown
    assert "真实 sentence-transformers embedding + FAISS" in markdown


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


def test_render_experiment_report_includes_safety_adversarial_section() -> None:
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
        eval_record_count=3,
        expected_keyword_record_count=1,
        safety_adversarial={
            "sample_count": 2,
            "overall_hit_rate": 0.5,
            "by_category": {
                "paraphrase": {"sample_count": 1, "hit_count": 0, "hit_rate": 0.0},
                "direct_control": {"sample_count": 1, "hit_count": 1, "hit_rate": 1.0},
            },
            "missed_ids": ["adv_001"],
        },
    )

    assert "## Safety Audit 对抗鲁棒性测试" in markdown
    assert "| paraphrase | 1 | 0 | 0.000 |" in markdown
    assert "overall_hit_rate = 0.500" in markdown
    assert "主要漏报样例：`adv_001`" in markdown


def test_render_experiment_report_mentions_grounded_rag() -> None:
    markdown = render_experiment_report(
        {
            "rag_dense": {
                "citation_hit_rate": 0.52,
                "context_recall": 0.52,
                "expected_keyword_coverage": 0.38,
                "lexical_answer_coverage": 0.16,
                "tool_selection_accuracy": 0.0,
                "tool_execution_success_rate": 0.0,
                "evidence_coverage": 1.0,
                "answer_correctness_proxy": 0.46,
                "faithfulness_proxy": 0.38,
                "grounding_rate": 0.31,
            },
            "rag_dense_grounded": {
                "citation_hit_rate": 0.56,
                "context_recall": 0.56,
                "expected_keyword_coverage": 0.44,
                "lexical_answer_coverage": 0.24,
                "tool_selection_accuracy": 0.0,
                "tool_execution_success_rate": 0.0,
                "evidence_coverage": 1.0,
                "answer_correctness_proxy": 0.51,
                "faithfulness_proxy": 0.42,
                "grounding_rate": 0.68,
            },
        },
        eval_record_count=100,
        expected_keyword_record_count=100,
    )

    assert "rag_dense_grounded" in markdown
    assert "grounding_rate" in markdown
    assert "extractive vs grounded generation" in markdown
    assert "成对对比" in markdown


def test_render_experiment_report_mentions_dropt_boundary() -> None:
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
        eval_record_count=1,
        expected_keyword_record_count=1,
    )

    assert "DROPT" in markdown
    assert "20 维 BEAR state" in markdown
    assert "明确回退" in markdown


def test_render_experiment_report_includes_dropt_policy_benchmark() -> None:
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
        eval_record_count=1,
        dropt_policy_benchmark={
            "sample_count": 3,
            "success_count": 3,
            "fallback_count": 0,
            "avg_latency_ms": 12.345,
            "avg_action_dim": 6.0,
            "avg_abs_action": 0.42,
        },
    )

    assert "DROPT Policy Benchmark" in markdown
    assert "| 3 | 3 | 0 | 12.345 | 6.000 | 0.420 |" in markdown


def test_render_experiment_report_mentions_react_agent() -> None:
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
            },
            "react_agent": {
                "citation_hit_rate": 0.4,
                "context_recall": 0.4,
                "expected_keyword_coverage": 0.45,
                "lexical_answer_coverage": 0.35,
                "tool_selection_accuracy": 0.9,
                "tool_execution_success_rate": 0.9,
                "evidence_coverage": 0.9,
                "answer_correctness_proxy": 0.45,
                "faithfulness_proxy": 0.42,
            },
        },
        eval_record_count=1,
        expected_keyword_record_count=1,
    )

    assert "react_agent" in markdown
    assert "workflow vs multi-step agent" in markdown
