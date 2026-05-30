from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalRecord(BaseModel):
    id: str
    question: str
    task_type: str
    gold_answer: str
    required_tools: list[str] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    expected_steps: list[str] = Field(default_factory=list)
    expected_tool_sequence: list[str] = Field(default_factory=list)
    expected_recoveries: list[str] = Field(default_factory=list)
    expected_runtime_events: list[str] = Field(default_factory=list)
    runtime_scenario: str | None = None
    difficulty: str | None = None
    capability_tags: list[str] = Field(default_factory=list)
    distractor_type: str | None = None
    expected_failure_mode: str | None = None
    grading_rubric: dict[str, float] = Field(default_factory=dict)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    expected_output_format: str


def load_eval_dataset(path: str | Path) -> list[EvalRecord]:
    dataset_path = Path(path)
    records: list[EvalRecord] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(EvalRecord.model_validate(json.loads(stripped)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records
