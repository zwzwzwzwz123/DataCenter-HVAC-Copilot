from pathlib import Path

from src.evaluation.dataset import load_eval_dataset
from src.evaluation.metrics import (
    answer_correctness_proxy,
    citation_hit_rate,
    context_recall,
    evidence_coverage,
    expected_keyword_coverage,
    faithfulness_proxy,
    grounding_rate,
    hallucination_proxy_rate,
    lexical_answer_coverage,
    planned_step_accuracy,
    planned_step_order_accuracy,
    policy_final_step_rate,
    retrieval_mrr_at_k,
    retrieval_ndcg_at_k,
    retrieval_recall_at_k,
    required_step_recall,
    tool_execution_success_rate,
    tool_selection_accuracy,
)


def test_load_eval_dataset_reads_jsonl_records():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))

    assert len(records) >= 34
    assert records[0].id == "doc_qa_001"
    assert records[0].task_type == "document_qa"
    assert records[0].expected_output_format == "answer_with_citations"
    assert isinstance(records[0].expected_keywords, list)


def test_load_eval_dataset_accepts_utf8_bom_jsonl(tmp_path: Path):
    dataset_path = tmp_path / "eval_bom.jsonl"
    dataset_path.write_text(
        '\ufeff{"id":"doc_001","question":"q","task_type":"document_qa",'
        '"gold_answer":"a","required_tools":[],"required_documents":[],'
        '"expected_keywords":["q"],"expected_output_format":"answer"}\n',
        encoding="utf-8",
    )

    records = load_eval_dataset(dataset_path)

    assert records[0].id == "doc_001"


def test_load_eval_dataset_accepts_compound_task_expected_steps(tmp_path: Path):
    dataset_path = tmp_path / "compound_eval.jsonl"
    dataset_path.write_text(
        (
            '{"id":"compound_001","question":"最近温度异常升高，并给出控制建议",'
            '"task_type":"policy_recommendation","gold_answer":"先查询温度，再诊断异常，最后给策略建议。",'
            '"required_tools":["query_metric","detect_anomaly","rule_based_policy"],'
            '"required_documents":[],"expected_keywords":["温度","异常","策略"],'
            '"expected_steps":["timeseries_query","anomaly_diagnosis","policy_recommendation"],'
            '"expected_output_format":"multi_step_policy_with_tool_evidence"}\n'
        ),
        encoding="utf-8",
    )

    records = load_eval_dataset(dataset_path)

    assert records[0].expected_steps == [
        "timeseries_query",
        "anomaly_diagnosis",
        "policy_recommendation",
    ]


def test_eval_dataset_has_curated_keywords_for_representative_records():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    keyword_records = [record for record in records if record.expected_keywords]

    assert len(records) == 108
    assert len(keyword_records) == 108
    assert {
        "document_qa",
        "timeseries_query",
        "anomaly_diagnosis",
        "policy_recommendation",
    }.issubset({record.task_type for record in keyword_records})


def test_eval_dataset_loads_quality_proxy_annotations():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    record = {record.id: record for record in records}["doc_qa_006"]

    assert "BEAR" in record.must_include
    assert "真实数据中心生产遥测" in record.must_not_include


def test_eval_dataset_has_quality_proxy_annotations_for_representative_records():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    annotated = [record for record in records if record.must_include or record.must_not_include]

    assert len(annotated) >= 40


def test_eval_dataset_task_type_distribution_matches_stage_target():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    counts = {}
    for record in records:
        counts[record.task_type] = counts.get(record.task_type, 0) + 1

    assert counts == {
        "document_qa": 40,
        "timeseries_query": 20,
        "anomaly_diagnosis": 20,
        "policy_recommendation": 28,
    }


def test_citation_hit_rate_counts_required_documents():
    records = _records_by_id(["doc_qa_001", "ts_query_001"])
    predictions = {
        "doc_qa_001": {
            "citations": [
                {"source_id": "hvac_energy_reference"},
                {"source_id": "other_doc"},
            ]
        },
        "ts_query_001": {"citations": []},
    }

    assert citation_hit_rate(records, predictions) == 1.0


def test_context_recall_counts_required_documents_in_retrieved_contexts():
    records = _records_by_id(["doc_qa_001", "doc_qa_009"])
    predictions = {
        "doc_qa_001": {
            "retrieved_contexts": [
                {"citation": {"source_id": "hvac_energy_reference"}},
                {"citation": {"source_id": "other_doc"}},
            ]
        },
        "doc_qa_009": {
            "retrieved_contexts": [
                {"citation": {"source_id": "hvac_energy_reference"}},
            ]
        },
    }

    assert context_recall(records, predictions) == 0.5


def test_document_metrics_match_required_document_aliases_from_citation_metadata(tmp_path: Path):
    dataset_path = tmp_path / "alias_eval.jsonl"
    dataset_path.write_text(
        (
            '{"id":"alias_001","question":"q","task_type":"document_qa",'
            '"gold_answer":"a","required_tools":[],'
            '"required_documents":["lbnl_air_management_tool_user_manual_2023.pdf"],'
            '"expected_keywords":[],"expected_output_format":"answer"}\n'
        ),
        encoding="utf-8",
    )
    records = load_eval_dataset(dataset_path)
    predictions = {
        "alias_001": {
            "citations": [
                {
                    "source_id": "doc_b0eb00b5ec964fcc9a54cf4190296f5c",
                    "title": "lbnl_air_management_tool_user_manual_2023.pdf",
                    "source_path": (
                        "data/knowledge/uploads/"
                        "doc_b0eb00b5ec964fcc9a54cf4190296f5c_"
                        "lbnl_air_management_tool_user_manual_2023.pdf"
                    ),
                }
            ],
            "retrieved_contexts": [
                {
                    "citation": {
                        "source_id": "doc_b0eb00b5ec964fcc9a54cf4190296f5c",
                        "title": "lbnl_air_management_tool_user_manual_2023.pdf",
                        "source_path": (
                            "data/knowledge/uploads/"
                            "doc_b0eb00b5ec964fcc9a54cf4190296f5c_"
                            "lbnl_air_management_tool_user_manual_2023.pdf"
                        ),
                    },
                    "metadata": {
                        "filename": "lbnl_air_management_tool_user_manual_2023.pdf",
                        "file_hash": "hash-air-management",
                        "source_url": "https://datacenters.lbl.gov/air-management.pdf",
                    },
                }
            ],
        }
    }

    assert citation_hit_rate(records, predictions) == 1.0
    assert context_recall(records, predictions) == 1.0
    assert retrieval_recall_at_k(records, predictions, k=1) == 1.0
    assert retrieval_mrr_at_k(records, predictions, k=1) == 1.0
    assert retrieval_ndcg_at_k(records, predictions, k=1) == 1.0


def test_retrieval_metrics_match_file_hash_and_url_aliases(tmp_path: Path):
    dataset_path = tmp_path / "alias_eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"alias_hash","question":"q1","task_type":"document_qa",'
                    '"gold_answer":"a","required_tools":[],'
                    '"required_documents":["hash-thermal-guidelines"],'
                    '"expected_keywords":[],"expected_output_format":"answer"}'
                ),
                (
                    '{"id":"alias_url","question":"q2","task_type":"document_qa",'
                    '"gold_answer":"a","required_tools":[],'
                    '"required_documents":["https://www.ashrae.org/thermal-refcard.pdf"],'
                    '"expected_keywords":[],"expected_output_format":"answer"}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_eval_dataset(dataset_path)
    predictions = {
        "alias_hash": {
            "retrieved_contexts": [
                {
                    "citation": {"source_id": "doc_fc01407c4a9c49d18694b13e669e371d"},
                    "metadata": {
                        "filename": "ashrae_tc99_thermal_guidelines_refcard_2021.pdf",
                        "file_hash": "hash-thermal-guidelines",
                    },
                }
            ]
        },
        "alias_url": {
            "retrieved_contexts": [
                {
                    "citation": {"source_id": "doc_fc01407c4a9c49d18694b13e669e371d"},
                    "metadata": {
                        "filename": "ashrae_tc99_thermal_guidelines_refcard_2021.pdf",
                        "source_url": "https://www.ashrae.org/thermal-refcard.pdf",
                    },
                }
            ]
        },
    }

    assert retrieval_recall_at_k(records, predictions, k=1) == 1.0
    assert retrieval_mrr_at_k(records, predictions, k=1) == 1.0


def test_retrieval_ranking_metrics_use_required_documents_as_binary_relevance(tmp_path: Path):
    dataset_path = tmp_path / "ranking_eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"rank_001","question":"q1","task_type":"document_qa",'
                    '"gold_answer":"a1","required_tools":[],"required_documents":["doc_a","doc_b"],'
                    '"expected_keywords":[],"expected_output_format":"answer"}'
                ),
                (
                    '{"id":"rank_002","question":"q2","task_type":"document_qa",'
                    '"gold_answer":"a2","required_tools":[],"required_documents":["doc_c"],'
                    '"expected_keywords":[],"expected_output_format":"answer"}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_eval_dataset(dataset_path)
    predictions = {
        "rank_001": {
            "retrieved_contexts": [
                {"citation": {"source_id": "noise"}},
                {"citation": {"source_id": "doc_a"}},
                {"citation": {"source_id": "doc_b"}},
            ]
        },
        "rank_002": {
            "retrieved_contexts": [
                {"citation": {"source_id": "doc_x"}},
                {"citation": {"source_id": "doc_y"}},
                {"citation": {"source_id": "doc_c"}},
            ]
        },
    }

    assert retrieval_recall_at_k(records, predictions, k=1) == 0.0
    assert retrieval_recall_at_k(records, predictions, k=3) == 1.0
    assert round(retrieval_mrr_at_k(records, predictions, k=3), 3) == 0.417
    assert round(retrieval_ndcg_at_k(records, predictions, k=3), 3) == 0.597


def test_tool_selection_accuracy_counts_required_tools():
    records = _records_by_id(["ts_query_001", "anomaly_001", "policy_001"])
    predictions = {
        "ts_query_001": {"tools": ["query_metric"]},
        "anomaly_001": {"tools": ["query_metric"]},
        "policy_001": {"tools": ["rule_based_policy", "mpc_like_policy"]},
    }

    assert round(tool_selection_accuracy(records, predictions), 2) == 0.67


def test_lexical_answer_coverage_counts_gold_answer_keywords():
    records = _records_by_id(["doc_qa_010", "policy_001"])
    predictions = {
        "doc_qa_010": {
            "answer": "Agent 负责任务路由、证据收集和解释，控制建议来自 policy 工具。"
        },
        "policy_001": {
            "answer": "应调用 rule_based_policy，并说明 diffusion 模型不可用时不能伪造。"
        },
    }

    assert lexical_answer_coverage(records, predictions) > 0.2


def test_expected_keyword_coverage_uses_curated_keywords_only():
    records = _records_by_id(["doc_qa_010", "ts_query_001"])
    predictions = {
        "doc_qa_010": {
            "answer": "Agent 负责任务路由和证据收集，控制建议来自 policy 工具。"
        },
        "ts_query_001": {
            "answer": "zone_temperature 最大值为 30.0，并返回时间范围。"
        },
    }

    assert expected_keyword_coverage(records, predictions) == 1.0


def test_answer_correctness_proxy_scores_must_include_matches():
    records = _records_by_id(["doc_qa_006"])
    predictions = {
        "doc_qa_006": {
            "answer": "BEAR 是 HVAC 仿真轨迹，可作为可控代理场景。",
            "citations": [{"source_id": "bear_data_boundary_note"}],
            "tool_results": [],
        }
    }

    assert answer_correctness_proxy(records, predictions) == 1.0


def test_faithfulness_proxy_penalizes_missing_evidence_and_forbidden_terms():
    records = _records_by_id(["doc_qa_006", "policy_002"])
    predictions = {
        "doc_qa_006": {
            "answer": "BEAR 是真实数据中心生产遥测。",
            "citations": [{"source_id": "bear_data_boundary_note"}],
            "tool_results": [],
        },
        "policy_002": {
            "answer": "LLM 不应直接编造控制动作。",
            "citations": [],
            "tool_results": [],
        },
    }

    assert faithfulness_proxy(records, predictions) == 0.25


def test_hallucination_proxy_rate_counts_forbidden_boundary_violations():
    records = _records_by_id(["doc_qa_006", "policy_002"])
    predictions = {
        "doc_qa_006": {
            "answer": "BEAR 是真实数据中心生产遥测。",
            "citations": [{"source_id": "bear_data_boundary_note"}],
        },
        "policy_002": {
            "answer": "LLM 只解释 policy 工具结果。",
            "tool_results": [{"policy_name": "rule_based"}],
        },
    }

    assert hallucination_proxy_rate(records, predictions) == 0.5


def test_tool_execution_success_rate_counts_non_empty_tool_results():
    records = _records_by_id(["ts_query_001", "anomaly_001", "policy_001"])
    predictions = {
        "ts_query_001": {"tools": ["query_metric"], "tool_results": [{"summary": {"max": 30.0}}]},
        "anomaly_001": {"tools": ["detect_anomaly"], "tool_results": []},
        "policy_001": {
            "tools": ["rule_based_policy"],
            "tool_results": [{"policy_name": "rule_based"}],
        },
    }

    assert round(tool_execution_success_rate(records, predictions), 2) == 0.67


def test_evidence_coverage_counts_required_evidence_from_citations_or_tools():
    records = _records_by_id(["doc_qa_001", "ts_query_001", "anomaly_001", "policy_001"])
    predictions = {
        "doc_qa_001": {"citations": [{"source_id": "hvac_energy_reference"}], "tool_results": []},
        "ts_query_001": {"citations": [], "tool_results": [{"summary": {"max": 30.0}}]},
        "anomaly_001": {"citations": [], "tool_results": []},
        "policy_001": {"citations": [], "tool_results": [{"policy_name": "rule_based"}]},
    }

    assert evidence_coverage(records, predictions) == 0.75


def test_grounding_rate_counts_answer_citations_present_in_retrieved_contexts(tmp_path: Path):
    dataset_path = tmp_path / "grounding_eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"doc_001","question":"q1","task_type":"document_qa",'
                    '"gold_answer":"a","required_tools":[],"required_documents":["doc_a"],'
                    '"expected_keywords":[],"expected_output_format":"answer"}'
                ),
                (
                    '{"id":"doc_002","question":"q2","task_type":"document_qa",'
                    '"gold_answer":"a","required_tools":[],"required_documents":["doc_b"],'
                    '"expected_keywords":[],"expected_output_format":"answer"}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_eval_dataset(dataset_path)
    predictions = {
        "doc_001": {
            "answer": "answer\n\nCitations:\n- doc_a: source",
            "retrieved_contexts": [{"citation": {"source_id": "doc_a"}}],
        },
        "doc_002": {
            "answer": "answer\n\nCitations:\n- doc_x: source",
            "retrieved_contexts": [{"citation": {"source_id": "doc_b"}}],
        },
    }

    assert grounding_rate(records, predictions) == 0.5


def test_planner_metrics_return_none_without_expected_steps():
    records = _records_by_id(["ts_query_001", "policy_001"])
    predictions = {
        "ts_query_001": {"planned_steps": [{"route": "timeseries_query"}]},
        "policy_001": {"planned_steps": [{"route": "policy_recommendation"}]},
    }

    assert planned_step_accuracy(records, predictions) is None
    assert planned_step_order_accuracy(records, predictions) is None
    assert required_step_recall(records, predictions) is None
    assert policy_final_step_rate(records, predictions) is None


def test_planner_metrics_score_expected_step_sets_and_order(tmp_path: Path):
    dataset_path = tmp_path / "compound_eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"compound_001","question":"最近温度异常升高，并给出控制建议",'
                    '"task_type":"policy_recommendation","gold_answer":"先查时序，再诊断异常，最后给策略。",'
                    '"required_tools":[],"required_documents":[],"expected_keywords":[],'
                    '"expected_steps":["timeseries_query","anomaly_diagnosis","policy_recommendation"],'
                    '"expected_output_format":"multi_step_policy_with_tool_evidence"}'
                ),
                (
                    '{"id":"compound_002","question":"查询温度趋势，判断是否异常",'
                    '"task_type":"anomaly_diagnosis","gold_answer":"先查时序，再诊断异常。",'
                    '"required_tools":[],"required_documents":[],"expected_keywords":[],'
                    '"expected_steps":["timeseries_query","anomaly_diagnosis"],'
                    '"expected_output_format":"multi_step_anomaly_with_tool_evidence"}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_eval_dataset(dataset_path)
    predictions = {
        "compound_001": {
            "planned_steps": [
                {"route": "timeseries_query"},
                {"route": "anomaly_diagnosis"},
                {"route": "policy_recommendation"},
            ]
        },
        "compound_002": {
            "planned_steps": [
                {"route": "anomaly_diagnosis"},
                {"route": "timeseries_query"},
            ]
        },
    }

    assert planned_step_accuracy(records, predictions) == 1.0
    assert planned_step_order_accuracy(records, predictions) == 0.5
    assert required_step_recall(records, predictions) == 1.0
    assert policy_final_step_rate(records, predictions) == 1.0


def _records_by_id(ids: list[str]):
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    by_id = {record.id: record for record in records}
    return [by_id[id_] for id_ in ids]
