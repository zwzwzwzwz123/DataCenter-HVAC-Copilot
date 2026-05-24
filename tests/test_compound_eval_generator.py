from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.compound_task_generator import (
    CompoundTaskGenerator,
    validate_compound_task_candidates,
    write_compound_task_dataset,
)


class FakeCompoundTransport:
    def __init__(self, content: object | None = None) -> None:
        self.calls: list[dict] = []
        self.content = content

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        content = self.content or {
            "records": [
                {
                    "id": "compound_generated_001",
                    "question": "最近温度异常升高，并给出控制建议",
                    "task_type": "policy_recommendation",
                    "gold_answer": "先查询温度，再诊断异常，最后给策略建议。",
                    "required_tools": [
                        "query_metric",
                        "detect_anomaly",
                        "rule_based_policy",
                    ],
                    "required_documents": [],
                    "expected_keywords": ["温度", "异常", "策略"],
                    "expected_steps": [
                        "timeseries_query",
                        "anomaly_diagnosis",
                        "policy_recommendation",
                    ],
                    "expected_output_format": "multi_step_policy_with_tool_evidence",
                },
                {
                    "id": "bad_policy_order",
                    "question": "给建议再查温度",
                    "task_type": "policy_recommendation",
                    "gold_answer": "bad",
                    "required_tools": [],
                    "required_documents": [],
                    "expected_keywords": [],
                    "expected_steps": [
                        "policy_recommendation",
                        "timeseries_query",
                    ],
                    "expected_output_format": "bad",
                },
            ]
        }
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(content, ensure_ascii=False)
                    }
                }
            ]
        }


def test_validate_compound_task_candidates_filters_invalid_policy_order() -> None:
    records = validate_compound_task_candidates(
        [
            {
                "id": "compound_001",
                "question": "查询温度趋势，判断是否异常",
                "task_type": "anomaly_diagnosis",
                "gold_answer": "先查询时序，再诊断异常。",
                "required_tools": ["query_metric", "detect_anomaly"],
                "required_documents": [],
                "expected_keywords": ["温度", "异常"],
                "expected_steps": ["timeseries_query", "anomaly_diagnosis"],
                "expected_output_format": "multi_step_anomaly_with_tool_evidence",
            },
            {
                "id": "compound_bad",
                "question": "bad",
                "task_type": "policy_recommendation",
                "gold_answer": "bad",
                "required_tools": [],
                "required_documents": [],
                "expected_keywords": [],
                "expected_steps": ["policy_recommendation", "timeseries_query"],
                "expected_output_format": "bad",
            },
        ]
    )

    assert [record.id for record in records] == ["compound_001"]
    assert records[0].expected_steps == ["timeseries_query", "anomaly_diagnosis"]


def test_validate_compound_task_candidates_rejects_unsupported_domain_terms() -> None:
    records = validate_compound_task_candidates(
        [
            {
                "id": "compound_unsupported",
                "question": "Check AHU-1 pressure and humidity anomalies.",
                "task_type": "anomaly_diagnosis",
                "gold_answer": "Pressure and humidity are anomalous.",
                "required_tools": ["query_metric", "detect_anomaly"],
                "required_documents": [],
                "expected_keywords": ["pressure", "humidity"],
                "expected_steps": ["timeseries_query", "anomaly_diagnosis"],
                "expected_output_format": "multi_step_anomaly_with_tool_evidence",
            },
            {
                "id": "compound_supported",
                "question": "Check zone_temperature trend and diagnose anomalies.",
                "task_type": "anomaly_diagnosis",
                "gold_answer": "Use zone_temperature evidence and detect_anomaly.",
                "required_tools": ["query_metric", "detect_anomaly"],
                "required_documents": [],
                "expected_keywords": ["zone_temperature", "anomaly"],
                "expected_steps": ["timeseries_query", "anomaly_diagnosis"],
                "expected_output_format": "multi_step_anomaly_with_tool_evidence",
            },
        ]
    )

    assert [record.id for record in records] == ["compound_supported"]


def test_compound_task_generator_uses_llm_and_local_validation() -> None:
    transport = FakeCompoundTransport()
    generator = CompoundTaskGenerator(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="compound-test",
        transport=transport,
    )

    records = generator.generate(count=2)

    assert [record.id for record in records] == ["compound_generated_001"]
    assert records[0].expected_steps == [
        "timeseries_query",
        "anomaly_diagnosis",
        "policy_recommendation",
    ]
    assert transport.calls[0]["url"] == "https://example.deepseek.test/chat/completions"
    assert "compound_task" in transport.calls[0]["payload"]["messages"][0]["content"]


def test_compound_task_generator_accepts_top_level_record_list() -> None:
    transport = FakeCompoundTransport(
        content=[
            {
                "id": "compound_generated_list_001",
                "question": "查询温度趋势，判断是否异常",
                "task_type": "anomaly_diagnosis",
                "gold_answer": "先查询时序，再诊断异常。",
                "required_tools": ["query_metric", "detect_anomaly"],
                "required_documents": [],
                "expected_keywords": ["温度", "异常"],
                "expected_steps": ["timeseries_query", "anomaly_diagnosis"],
                "expected_output_format": "multi_step_anomaly_with_tool_evidence",
            }
        ]
    )
    generator = CompoundTaskGenerator(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="compound-test",
        transport=transport,
    )

    records = generator.generate(count=1)

    assert [record.id for record in records] == ["compound_generated_list_001"]


def test_write_compound_task_dataset_writes_valid_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "compound_eval.jsonl"
    records = validate_compound_task_candidates(
        [
            {
                "id": "compound_001",
                "question": "查询 zone_temperature 趋势，判断是否异常，再说明策略边界",
                "task_type": "policy_recommendation",
                "gold_answer": "先查询时序，再诊断异常，最后说明策略边界。",
                "required_tools": ["query_metric", "detect_anomaly", "rule_based_policy"],
                "required_documents": [],
                "expected_keywords": ["zone_temperature", "异常", "策略边界"],
                "expected_steps": [
                    "timeseries_query",
                    "anomaly_diagnosis",
                    "policy_recommendation",
                ],
                "expected_output_format": "multi_step_policy_with_tool_evidence",
            }
        ]
    )

    write_compound_task_dataset(records, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["expected_steps"][-1] == "policy_recommendation"
