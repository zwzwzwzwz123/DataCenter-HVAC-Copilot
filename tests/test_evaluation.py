from pathlib import Path

from src.evaluation.dataset import load_eval_dataset
from src.evaluation.metrics import (
    answer_correctness_proxy,
    citation_hit_rate,
    context_recall,
    evidence_coverage,
    expected_keyword_coverage,
    faithfulness_proxy,
    lexical_answer_coverage,
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


def test_eval_dataset_has_curated_keywords_for_representative_records():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    keyword_records = [record for record in records if record.expected_keywords]

    assert len(records) == 108
    assert len(keyword_records) == 108
    assert {"document_qa", "timeseries_query", "anomaly_diagnosis", "policy_recommendation"}.issubset(
        {record.task_type for record in keyword_records}
    )


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
        # 20 original policy records plus 8 multi-hop policy records used to
        # distinguish single-step workflow from deterministic ReAct planning.
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


def test_tool_execution_success_rate_counts_non_empty_tool_results():
    records = _records_by_id(["ts_query_001", "anomaly_001", "policy_001"])
    predictions = {
        "ts_query_001": {"tools": ["query_metric"], "tool_results": [{"summary": {"max": 30.0}}]},
        "anomaly_001": {"tools": ["detect_anomaly"], "tool_results": []},
        "policy_001": {"tools": ["rule_based_policy"], "tool_results": [{"policy_name": "rule_based"}]},
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


def _records_by_id(ids: list[str]):
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    by_id = {record.id: record for record in records}
    return [by_id[id_] for id_ in ids]
