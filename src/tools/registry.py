from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["read_only", "advisory", "control_boundary"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    route: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    risk_level: RiskLevel = "read_only"
    default_metric: str | None = None
    keywords: tuple[str, ...] = ()
    requires_policy_boundary: bool = False

    @property
    def input_schema(self) -> dict[str, str]:
        return _field_type_map(self.input_model)

    @property
    def output_schema(self) -> dict[str, str]:
        return _field_type_map(self.output_model)

    @property
    def input_json_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    @property
    def output_json_schema(self) -> dict[str, Any]:
        return self.output_model.model_json_schema()


class EmptyInput(BaseModel):
    pass


class EmptyOutput(BaseModel):
    pass


class RAGRetrievalInput(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1)


class RAGRetrievalOutput(BaseModel):
    citations: list[dict[str, Any]]
    retrieved_contexts: list[dict[str, Any]]


class QueryMetricInput(BaseModel):
    metric_name: str = "zone_temperature"
    start_time: str
    end_time: str
    zone_id: str | None = None


class QueryMetricOutput(BaseModel):
    summary: dict[str, Any]
    records: list[dict[str, Any]]


class ComparePeriodInput(BaseModel):
    metric_name: str = "zone_temperature"
    period_a: tuple[str, str]
    period_b: tuple[str, str]
    zone_id: str | None = None


class ComparePeriodOutput(BaseModel):
    delta_mean: float | None = None
    percent_change_mean: float | None = None


class PlotMetricTrendInput(BaseModel):
    metric_name: str = "zone_temperature"
    start_time: str
    end_time: str
    zone_id: str | None = None


class PlotMetricTrendOutput(BaseModel):
    chart_type: str
    series: list[dict[str, Any]]


class ComputeEnergyBreakdownInput(BaseModel):
    start_time: str
    end_time: str


class ComputeEnergyBreakdownOutput(BaseModel):
    components: dict[str, float]
    total: float


class DataQualityCheckInput(BaseModel):
    required_fields: list[str] = Field(
        default_factory=lambda: ["timestamp", "scenario_id", "zone_id", "zone_temperature"]
    )
    expected_frequency: str | None = "1h"


class DataQualityCheckOutput(BaseModel):
    missing_fields: list[str]
    null_counts: dict[str, int]
    quality_score: float
    status: str


class ComfortRiskAssessmentInput(BaseModel):
    temperature_metric: str = "zone_temperature"
    comfort_lower_bound: float = 22.0
    comfort_upper_bound: float = 26.0


class ComfortRiskAssessmentOutput(BaseModel):
    risk_level: str
    violation_count: int
    zone_risks: list[dict[str, Any]]


class ZoneHotspotRankInput(BaseModel):
    metric_name: str = "zone_temperature"
    top_k: int = Field(default=3, ge=1, le=20)


class ZoneHotspotRankOutput(BaseModel):
    ranked_zones: list[dict[str, Any]]


class ControlActionAuditInput(BaseModel):
    action_metric: str = "control_action"
    change_threshold: float = Field(default=0.5, ge=0.0)


class ControlActionAuditOutput(BaseModel):
    large_change_count: int
    stability_status: str
    events: list[dict[str, Any]]


class CoolingEfficiencySummaryInput(BaseModel):
    power_metrics: list[str] | None = None
    temperature_metric: str = "zone_temperature"
    comfort_upper_bound: float = 26.0


class CoolingEfficiencySummaryOutput(BaseModel):
    total_power: float
    comfort_violation_count: int
    efficiency_status: str


class DetectAnomalyInput(BaseModel):
    metric_name: str = "zone_temperature"
    window_size: int = Field(default=2, ge=2)
    threshold: float = Field(default=2.0, gt=0.0)
    zone_id: str | None = None


class DetectAnomalyOutput(BaseModel):
    anomalies: list[dict[str, Any]]


class PolicyRunnerInput(BaseModel):
    state: dict[str, Any]


class PolicyRunnerOutput(BaseModel):
    recommended_action: list[float]
    policy_name: str
    notes: str | None = None


TOOL_SPECS = [
    ToolSpec(
        name="rag_retrieval",
        route="document_qa",
        description="Retrieve cited HVAC and data-center operations documents.",
        input_model=RAGRetrievalInput,
        output_model=RAGRetrievalOutput,
        keywords=("document", "citation", "文档", "引用"),
    ),
    ToolSpec(
        name="query_metric",
        route="timeseries_query",
        description="Query a metric over a time window and optional zone.",
        input_model=QueryMetricInput,
        output_model=QueryMetricOutput,
        default_metric="zone_temperature",
        keywords=("metric", "最大", "最小", "平均", "查询"),
    ),
    ToolSpec(
        name="compare_period",
        route="timeseries_query",
        description="Compare metric means between two periods.",
        input_model=ComparePeriodInput,
        output_model=ComparePeriodOutput,
        default_metric="zone_temperature",
        keywords=("compare", "比较", "对比", "变化"),
    ),
    ToolSpec(
        name="plot_metric_trend",
        route="timeseries_query",
        description="Return serializable line-chart data for a metric trend.",
        input_model=PlotMetricTrendInput,
        output_model=PlotMetricTrendOutput,
        default_metric="zone_temperature",
        keywords=("trend", "趋势", "折线图", "画"),
    ),
    ToolSpec(
        name="compute_energy_breakdown",
        route="timeseries_query",
        description="Sum available cooling, fan, chiller, or aggregate HVAC power fields.",
        input_model=ComputeEnergyBreakdownInput,
        output_model=ComputeEnergyBreakdownOutput,
        default_metric="cooling_power",
        keywords=("energy", "breakdown", "能耗", "构成"),
    ),
    ToolSpec(
        name="data_quality_check",
        route="timeseries_query",
        description="Check missing fields, null values, duplicate timestamps, and time gaps.",
        input_model=DataQualityCheckInput,
        output_model=DataQualityCheckOutput,
        keywords=("quality", "missing", "缺失", "数据质量", "字段"),
    ),
    ToolSpec(
        name="comfort_risk_assessment",
        route="anomaly_diagnosis",
        description="Assess comfort-boundary violations by zone.",
        input_model=ComfortRiskAssessmentInput,
        output_model=ComfortRiskAssessmentOutput,
        risk_level="advisory",
        default_metric="zone_temperature",
        keywords=("comfort", "risk", "过热", "舒适", "越限"),
    ),
    ToolSpec(
        name="zone_hotspot_rank",
        route="timeseries_query",
        description="Rank zones by metric maximum and mean to locate hotspots.",
        input_model=ZoneHotspotRankInput,
        output_model=ZoneHotspotRankOutput,
        default_metric="zone_temperature",
        keywords=("hotspot", "hottest", "top", "最热", "区域", "zone"),
    ),
    ToolSpec(
        name="control_action_audit",
        route="timeseries_query",
        description="Audit control-action jumps and stability.",
        input_model=ControlActionAuditInput,
        output_model=ControlActionAuditOutput,
        risk_level="advisory",
        default_metric="control_action",
        keywords=("control", "action", "震荡", "抖动", "控制"),
    ),
    ToolSpec(
        name="cooling_efficiency_summary",
        route="timeseries_query",
        description="Summarize cooling power versus comfort risk.",
        input_model=CoolingEfficiencySummaryInput,
        output_model=CoolingEfficiencySummaryOutput,
        risk_level="advisory",
        default_metric="cooling_power",
        keywords=("efficiency", "节能", "效率", "能耗", "power"),
    ),
    ToolSpec(
        name="detect_anomaly",
        route="anomaly_diagnosis",
        description="Detect rolling-window z-score anomalies.",
        input_model=DetectAnomalyInput,
        output_model=DetectAnomalyOutput,
        risk_level="advisory",
        default_metric="zone_temperature",
        keywords=("anomaly", "alarm", "异常", "告警"),
    ),
    ToolSpec(
        name="policy_runner",
        route="policy_recommendation",
        description="Run the configured policy backend; LLM may explain but not invent actions.",
        input_model=PolicyRunnerInput,
        output_model=PolicyRunnerOutput,
        risk_level="control_boundary",
        requires_policy_boundary=True,
        keywords=("policy", "strategy", "策略", "建议", "控制"),
    ),
]

TOOL_REGISTRY = {spec.name: spec for spec in TOOL_SPECS}


def tools_for_route(route: str) -> list[ToolSpec]:
    return [spec for spec in TOOL_SPECS if spec.route == route]


def validate_tool_input(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    spec = TOOL_REGISTRY[tool_name]
    try:
        validated = spec.input_model.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Invalid input for {tool_name}: {exc}") from exc
    return validated.model_dump()


def build_planner_tool_prompt() -> str:
    routes = sorted({spec.route for spec in TOOL_SPECS})
    route_lines = []
    for route in routes:
        specs = tools_for_route(route)
        tool_lines = [
            f"{spec.name}(risk={spec.risk_level}, inputs={list(spec.input_schema)})"
            for spec in specs
        ]
        route_lines.append(f"{route}: {', '.join(tool_lines)}")
    return (
        "Allowed routes are exactly: "
        + ", ".join(routes)
        + ". Allowed tools by route: "
        + " | ".join(route_lines)
        + ". If policy_recommendation is included, it must be the final step."
    )


def _field_type_map(model: type[BaseModel]) -> dict[str, str]:
    return {
        name: _annotation_to_name(field.annotation)
        for name, field in model.model_fields.items()
    }


def _annotation_to_name(annotation: Any) -> str:
    return str(annotation).replace("typing.", "")
