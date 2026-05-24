from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.deepseek_generator import Transport
from src.agent.planner import MAX_PLAN_STEPS
from src.agent.router import SUPPORTED_ROUTES
from src.evaluation.dataset import EvalRecord

ALLOWED_REQUIRED_TOOLS = {
    "query_metric",
    "compare_period",
    "plot_metric_trend",
    "compute_energy_breakdown",
    "detect_anomaly",
    "rule_based_policy",
}
NON_TOOL_MARKERS = {
    "document_qa",
    "retrieve_policy_doc",
    "retrieval",
    "rag_retrieval",
    "timeseries_query",
    "anomaly_diagnosis",
    "policy_recommendation",
}
SUPPORTED_DOMAIN_TERMS = {
    "zone_temperature",
    "温度",
    "temperature",
    "cooling_power",
    "hvac_power",
    "fan_power",
    "control_action",
    "outdoor_temp",
    "internal_load",
    "energy",
    "energy breakdown",
    "comfort_violation",
    "setpoint",
    "policy",
    "策略",
    "控制",
    "anomaly",
    "异常",
    "trend",
    "趋势",
    "episode_001",
    "zone_a",
}
UNSUPPORTED_DOMAIN_TERMS = {
    "humidity",
    "pressure",
    "ahu",
    "chiller",
    "server room",
    "return air",
    "supply air",
    "damper",
    "valve",
    "coil",
    "cop",
    "economizer",
    "airflow",
    "duct",
    "filter",
    "backup cooling",
    "conference room",
    "east wing",
    "west wing",
}


class CompoundTaskGenerator:
    """Generate compound planner-eval records with an LLM, then validate locally."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _default_transport

    def generate(self, count: int = 20) -> list[EvalRecord]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _system_prompt()},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": "generate_compound_task_eval_records",
                                "count": count,
                                "allowed_routes": sorted(SUPPORTED_ROUTES),
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                "temperature": 0.4,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        response = self.transport(
            f"{self.base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            body,
            self.timeout_seconds,
        )
        content = str(response["choices"][0]["message"]["content"])
        parsed = _parse_json_object(content)
        if isinstance(parsed, dict):
            candidates = parsed.get("records", [])
        elif isinstance(parsed, list):
            candidates = parsed
        else:
            raise ValueError("LLM compound task payload must be an object or a records list")
        if not isinstance(candidates, list):
            raise ValueError("LLM compound task payload must contain a records list")
        return validate_compound_task_candidates(candidates)


def validate_compound_task_candidates(candidates: list[dict[str, Any]]) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        candidate = _normalize_candidate(candidate)
        try:
            record = EvalRecord.model_validate(candidate)
        except Exception:
            continue
        if record.id in seen_ids:
            continue
        if _is_valid_compound_record(record):
            records.append(record)
            seen_ids.add(record.id)
    return records


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(candidate)
    expected_steps = normalized.get("expected_steps")
    if (
        isinstance(expected_steps, list)
        and expected_steps
        and normalized.get("task_type") in {None, "", "compound", "compound_task"}
    ):
        normalized["task_type"] = expected_steps[-1]
    required_tools = normalized.get("required_tools")
    if isinstance(required_tools, list):
        normalized["required_tools"] = [
            tool
            for tool in required_tools
            if tool in ALLOWED_REQUIRED_TOOLS and tool not in NON_TOOL_MARKERS
        ]
    return normalized


def write_compound_task_dataset(records: list[EvalRecord], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.model_dump(), ensure_ascii=False)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _is_valid_compound_record(record: EvalRecord) -> bool:
    if len(record.expected_steps) < 2 or len(record.expected_steps) > MAX_PLAN_STEPS:
        return False
    if any(step not in SUPPORTED_ROUTES for step in record.expected_steps):
        return False
    if len(set(record.expected_steps)) != len(record.expected_steps):
        return False
    if (
        "policy_recommendation" in record.expected_steps
        and record.expected_steps[-1] != "policy_recommendation"
    ):
        return False
    if record.task_type not in SUPPORTED_ROUTES:
        return False
    if record.task_type != record.expected_steps[-1]:
        return False
    if any(tool not in ALLOWED_REQUIRED_TOOLS for tool in record.required_tools):
        return False
    if not _uses_supported_domain(record):
        return False
    return bool(record.question.strip() and record.gold_answer.strip())


def _uses_supported_domain(record: EvalRecord) -> bool:
    text = " ".join(
        [
            record.question,
            record.gold_answer,
            " ".join(record.expected_keywords),
        ]
    ).lower()
    if any(term in text for term in UNSUPPORTED_DOMAIN_TERMS):
        return False
    return any(term in text for term in SUPPORTED_DOMAIN_TERMS)


def _system_prompt() -> str:
    return (
        "You generate compound_task planner evaluation records for DataCenter-HVAC Copilot. "
        "Return only JSON with key records. Each record must match the eval JSONL schema: "
        "id, question, task_type, gold_answer, required_tools, required_documents, "
        "expected_keywords, expected_steps, expected_output_format. "
        "task_type must be exactly the final expected_steps route; never use compound as task_type. "
        "expected_steps must contain 2 to 3 of these routes only: document_qa, timeseries_query, "
        "anomaly_diagnosis, policy_recommendation. policy_recommendation, if present, must be final. "
        "Prefer 2-3 step questions such as timeseries_query -> anomaly_diagnosis -> "
        "policy_recommendation. required_tools must only use these executable tool names: "
        "query_metric, compare_period, plot_metric_trend, compute_energy_breakdown, "
        "detect_anomaly, rule_based_policy. Do not put document_qa, retrieval, retrieve_policy_doc, "
        "or route names in required_tools. "
        "Stay inside the current demo data schema. Only use these metrics or concepts: "
        "zone_temperature, cooling_power, hvac_power, fan_power, control_action, outdoor_temp, "
        "internal_load, comfort_violation, setpoint, energy breakdown, trend, anomaly, policy. "
        "Use episode_001 and zone_a when a scenario or zone is needed. Do not mention humidity, "
        "pressure, AHU, chiller, server room, return air, supply air, damper, valve, coil, COP, "
        "economizer, airflow, duct, filter, backup cooling, or named rooms/wings. "
        "Use required_documents as an empty list unless the step includes document_qa. "
        "Questions should combine HVAC time-series, anomaly, and policy needs, such as checking "
        "zone_temperature trend before diagnosing anomaly and recommending policy. "
        "Do not include production telemetry claims."
    )


def _parse_json_object(content: str) -> dict[str, Any] | list[Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    return json.loads(stripped)


def _default_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict[str, Any]:
    from urllib import request

    req = request.Request(url=url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
