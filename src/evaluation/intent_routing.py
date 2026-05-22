from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.agent.intent_classifier import IntentClassifier
from src.evaluation.dataset import EvalRecord


def evaluate_intent_classifier(
    records: list[EvalRecord],
    classifier: IntentClassifier,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_task_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    fallback_count = 0

    for record in records:
        decision = classifier.classify(record.question)
        correct = decision.route == record.task_type
        fallback_count += int(decision.fallback_used)
        confusion[record.task_type][decision.route] += 1
        by_task_type[record.task_type]["total"] += 1
        by_task_type[record.task_type]["correct"] += int(correct)
        predictions.append(
            {
                "id": record.id,
                "question": record.question,
                "expected_route": record.task_type,
                "predicted_route": decision.route,
                "classifier": decision.classifier,
                "confidence": decision.confidence,
                "fallback_used": decision.fallback_used,
                "correct": correct,
                "reason": decision.reason,
            }
        )

    total = len(records)
    correct_count = sum(1 for prediction in predictions if prediction["correct"])
    return {
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "fallback_rate": fallback_count / total if total else 0.0,
        "by_task_type": {
            task_type: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for task_type, counts in sorted(by_task_type.items())
        },
        "confusion_matrix": {
            expected: dict(sorted(predicted.items()))
            for expected, predicted in sorted(confusion.items())
        },
        "predictions": predictions,
    }


def compare_intent_classifiers(
    records: list[EvalRecord],
    classifiers: dict[str, IntentClassifier],
) -> dict[str, dict[str, Any]]:
    return {
        name: evaluate_intent_classifier(records, classifier)
        for name, classifier in classifiers.items()
    }
