from collections import Counter

from src.evaluation.dataset import load_eval_dataset


def test_real_eval_dataset_has_challenging_50_record_mix() -> None:
    records = load_eval_dataset("data/eval/real_eval.jsonl")

    assert len(records) == 50
    assert len({record.id for record in records}) == 50

    task_counts = Counter(record.task_type for record in records)
    assert task_counts == {
        "document_qa": 30,
        "timeseries_query": 7,
        "anomaly_diagnosis": 6,
        "policy_recommendation": 7,
    }

    ids = {record.id for record in records}
    assert sum(record.id.startswith("real_doc_interfere_") for record in records) >= 6
    assert sum(record.id.startswith("real_doc_boundary_") for record in records) >= 3
    assert sum(record.id.startswith("real_doc_multi_") for record in records) >= 5
    assert sum(bool(record.expected_steps) for record in records) >= 4

    assert "real_doc_interfere_001" in ids
    assert "real_doc_boundary_001" in ids
    assert "real_policy_multi_002" in ids

    document_records = [record for record in records if record.required_documents]
    multi_doc_records = [record for record in document_records if len(record.required_documents) >= 2]
    assert len(document_records) >= 30
    assert len(multi_doc_records) >= 8

    for record in records:
        assert record.question
        assert record.gold_answer
        assert record.expected_output_format
