from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.agent.answer_audit import audit_answer


class SafetyAdversarialRecord(BaseModel):
    id: str
    category: str
    question: str
    route: str = "document_qa"
    answer: str
    expected_violation: str
    policy_result: dict[str, Any] | None = None


def load_safety_adversarial_dataset(path: str | Path) -> list[SafetyAdversarialRecord]:
    dataset_path = Path(path)
    records: list[SafetyAdversarialRecord] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(SafetyAdversarialRecord.model_validate(json.loads(stripped)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def evaluate_safety_adversarial_dataset(
    records: list[SafetyAdversarialRecord],
) -> dict[str, Any]:
    rows = []
    category_counts: dict[str, dict[str, int]] = {}
    for record in records:
        audit = audit_answer(
            record.answer,
            route=record.route,
            policy_result=record.policy_result,
        )
        hit = record.expected_violation in audit.get("violations", [])
        category = category_counts.setdefault(record.category, {"sample_count": 0, "hit_count": 0})
        category["sample_count"] += 1
        if hit:
            category["hit_count"] += 1
        rows.append(
            {
                "id": record.id,
                "category": record.category,
                "question": record.question,
                "expected_violation": record.expected_violation,
                "violations": audit.get("violations", []),
                "hit": hit,
            }
        )

    sample_count = len(records)
    hit_count = sum(1 for row in rows if row["hit"])
    by_category = {
        category: {
            "sample_count": counts["sample_count"],
            "hit_count": counts["hit_count"],
            "hit_rate": _safe_rate(counts["hit_count"], counts["sample_count"]),
        }
        for category, counts in sorted(category_counts.items())
    }
    return {
        "sample_count": sample_count,
        "hit_count": hit_count,
        "overall_hit_rate": _safe_rate(hit_count, sample_count),
        "by_category": by_category,
        "missed_ids": [str(row["id"]) for row in rows if not row["hit"]],
        "predictions": rows,
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
