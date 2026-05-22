from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.agent.intent_classifier import IntentDecision
from src.evaluation.dataset import EvalRecord
from src.evaluation.intent_routing import compare_intent_classifiers, evaluate_intent_classifier


class StaticIntentClassifier:
    def __init__(self, route: str) -> None:
        self.route = route

    def classify(self, question: str, task_type: str | None = None) -> IntentDecision:
        return IntentDecision(
            route=self.route,
            required_tools=[],
            reason="static test classifier",
            classifier="static",
            confidence=0.8,
            fallback_used=False,
        )


def test_evaluate_intent_classifier_reports_accuracy_and_confusion_matrix() -> None:
    records = [
        _record("r1", "document_qa", "什么是 BEAR 数据？"),
        _record("r2", "document_qa", "请解释 PUE"),
    ]

    result = evaluate_intent_classifier(records, StaticIntentClassifier("document_qa"))

    assert result["total"] == 2
    assert result["correct"] == 2
    assert result["accuracy"] == 1.0
    assert result["fallback_rate"] == 0.0
    assert result["confusion_matrix"]["document_qa"]["document_qa"] == 2
    assert result["predictions"][0]["expected_route"] == "document_qa"


def test_compare_intent_classifiers_keeps_provider_names() -> None:
    records = [_record("r1", "timeseries_query", "画一下温度趋势")]

    result = compare_intent_classifiers(
        records,
        {
            "keyword": StaticIntentClassifier("timeseries_query"),
            "llm": StaticIntentClassifier("document_qa"),
        },
    )

    assert result["keyword"]["accuracy"] == 1.0
    assert result["llm"]["accuracy"] == 0.0


def test_run_intent_eval_writes_comparison_artifact(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "r1",
                "question": "画一下 zone_temperature 的趋势",
                "task_type": "timeseries_query",
                "gold_answer": "trend",
                "required_tools": ["plot_trend"],
                "required_documents": [],
                "expected_keywords": ["zone_temperature"],
                "expected_output_format": "tool_summary",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "intent_routing_comparison.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_intent_eval.py",
            "--eval-path",
            str(dataset_path),
            "--output",
            str(output_path),
            "--providers",
            "rule_based",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert "Saved intent routing comparison" in completed.stdout
    assert artifact["runs"]["rule_based"]["status"] == "complete"
    assert artifact["runs"]["rule_based"]["metrics"]["total"] == 1


def _record(record_id: str, task_type: str, question: str) -> EvalRecord:
    return EvalRecord(
        id=record_id,
        question=question,
        task_type=task_type,
        gold_answer="gold",
        required_tools=[],
        required_documents=[],
        expected_keywords=[],
        expected_output_format="answer",
    )
