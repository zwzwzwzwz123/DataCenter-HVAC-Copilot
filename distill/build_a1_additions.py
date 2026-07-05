"""A1 — append high-quality gold rows targeting the two residual fallback modes.

Phase-4 error analysis (see distill/HANDOFF.md §10) left the distilled A2 model
with 26/100 fallbacks, dominated by:
  * time_window the guard rejects (natural-language windows the model emitted
    but could not be normalized), and
  * illegal tool choices within a route.

A2 patched the *inference* layer (``_normalize_time_window``) as a safety net.
A1 is the *root-cause* fix: teach the model itself, via SFT data, to

  1. read a natural-language time window from the question ("过去一周",
     "last month", "past 3 days", "过去90分钟") and emit the canonical
     hours/minutes label the guard accepts (``last_168_hours`` etc.).
     The gold vocabulary only uses hours/minutes, so days/weeks/months are
     folded into hours (day*24, week*168, month*720) — identical to what
     ``_normalize_time_window`` does, so training and inference agree.

  2. pick the *right* tool within ``timeseries_query`` from the phrasing
     (single value -> query_metric, two periods -> compare_period, a chart ->
     plot_metric_trend, energy split -> compute_energy_breakdown, hotspot
     ranking -> zone_hotspot_rank, efficiency -> cooling_efficiency_summary,
     control history -> control_action_audit, data health -> data_quality_check).

Every row is validated through the SAME ``validate_plan_steps`` guard the online
planner uses; anything that does not pass aborts the append. IDs continue from
gold_0601 and the script is idempotent (re-running is a no-op once appended).

    python -m distill.build_a1_additions            # dry-run: validate + report
    python -m distill.build_a1_additions --append   # append to gold_labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.planner import PlanStep, validate_plan_steps

GOLD_PATH = "distill/data/gold_labeled.jsonl"

# ---------------------------------------------------------------------------
# New rows. Each is {question, steps}; ids are assigned sequentially on append.
# Kept as plain dicts so they read like the checked-in gold_labeled.jsonl.
# ---------------------------------------------------------------------------

# --- Group 1: natural-language time windows -> canonical hours/minutes -------
# Question phrases the window in days/weeks/months/colloquial terms; the label
# carries the guard-legal hours form. This is the behavior A2 bolted on at
# inference; here we push it into the weights.
TIME_WINDOW_ROWS: list[dict] = [
    # days -> hours
    {"question": "过去三天 zone_1 的平均温度是多少？",
     "steps": [{"route": "timeseries_query", "reason": "查询单一指标在时间窗内的统计值。", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_72_hours"}]},
    {"question": "看下过去两天的冷却功率均值。",
     "steps": [{"route": "timeseries_query", "reason": "查询冷却功率的时间窗均值。", "tool": "query_metric", "metric_name": "cooling_power", "time_window": "last_48_hours"}]},
    {"question": "What was the average PUE over the past 3 days?",
     "steps": [{"route": "timeseries_query", "reason": "Single-metric statistic over a time window.", "tool": "query_metric", "metric_name": "pue", "time_window": "last_72_hours"}]},
    {"question": "画一下最近5天 chiller_power 的趋势。",
     "steps": [{"route": "timeseries_query", "reason": "趋势序列用于折线可视化。", "tool": "plot_metric_trend", "metric_name": "chiller_power", "time_window": "last_120_hours"}]},
    {"question": "过去四天 zone_2 湿度画个折线。",
     "steps": [{"route": "timeseries_query", "reason": "湿度趋势可视化。", "tool": "plot_metric_trend", "metric_name": "humidity", "zone_id": "zone_2", "time_window": "last_96_hours"}]},
    # a week -> 168h
    {"question": "过去一周 fan_power 的平均值。",
     "steps": [{"route": "timeseries_query", "reason": "查询风机功率的一周均值。", "tool": "query_metric", "metric_name": "fan_power", "time_window": "last_168_hours"}]},
    {"question": "最近7天 zone_0 的温度趋势图。",
     "steps": [{"route": "timeseries_query", "reason": "一周温度趋势可视化。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "last_168_hours"}]},
    {"question": "Plot the hvac_power trend for the last week.",
     "steps": [{"route": "timeseries_query", "reason": "One-week trend for a line chart.", "tool": "plot_metric_trend", "metric_name": "hvac_power", "time_window": "last_168_hours"}]},
    {"question": "查一下过去一周 it_load 的平均负载。",
     "steps": [{"route": "timeseries_query", "reason": "查询 IT 负载一周均值。", "tool": "query_metric", "metric_name": "it_load", "time_window": "last_168_hours"}]},
    # two weeks -> 336h
    {"question": "过去两周的 cooling_power 均值是多少？",
     "steps": [{"route": "timeseries_query", "reason": "查询冷却功率两周均值。", "tool": "query_metric", "metric_name": "cooling_power", "time_window": "last_336_hours"}]},
    {"question": "Show the last two weeks of outdoor_temp as a chart.",
     "steps": [{"route": "timeseries_query", "reason": "Two-week trend visualization.", "tool": "plot_metric_trend", "metric_name": "outdoor_temp", "time_window": "last_336_hours"}]},
    # a month -> 720h
    {"question": "过去一个月的 pue 平均值。",
     "steps": [{"route": "timeseries_query", "reason": "查询 PUE 一个月均值。", "tool": "query_metric", "metric_name": "pue", "time_window": "last_720_hours"}]},
    {"question": "画出上个月 chiller_power 的走势。",
     "steps": [{"route": "timeseries_query", "reason": "一个月功率趋势可视化。", "tool": "plot_metric_trend", "metric_name": "chiller_power", "time_window": "last_720_hours"}]},
    {"question": "Average zone_1 temperature over the last month?",
     "steps": [{"route": "timeseries_query", "reason": "One-month statistic for a single metric.", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_720_hours"}]},
    {"question": "过去30天 hvac_power 的均值。",
     "steps": [{"route": "timeseries_query", "reason": "30 天折算为一个月的均值查询。", "tool": "query_metric", "metric_name": "hvac_power", "time_window": "last_720_hours"}]},
    # hours phrased colloquially
    {"question": "最近一天的 zone_0 温度均值。",
     "steps": [{"route": "timeseries_query", "reason": "最近一天即 24 小时的均值查询。", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "last_24_hours"}]},
    {"question": "今天的平均 fan_power 是多少？",
     "steps": [{"route": "timeseries_query", "reason": "今天折算为最近 24 小时。", "tool": "query_metric", "metric_name": "fan_power", "time_window": "last_24_hours"}]},
    {"question": "过去半天 cooling_power 的趋势。",
     "steps": [{"route": "timeseries_query", "reason": "半天折算为 12 小时的趋势。", "tool": "plot_metric_trend", "metric_name": "cooling_power", "time_window": "last_12_hours"}]},
    {"question": "过去8小时 zone_2 温度有没有升高，画个图。",
     "steps": [{"route": "timeseries_query", "reason": "8 小时温度趋势可视化。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_2", "time_window": "last_8_hours"}]},
    {"question": "Query it_load for the past 6 hours.",
     "steps": [{"route": "timeseries_query", "reason": "Six-hour metric statistic.", "tool": "query_metric", "metric_name": "it_load", "time_window": "last_6_hours"}]},
    # minutes
    {"question": "过去90分钟的 zone_1 温度均值。",
     "steps": [{"route": "timeseries_query", "reason": "90 分钟窗口的均值查询。", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_90_minutes"}]},
    {"question": "最近45分钟 fan_power 有异常吗？",
     "steps": [{"route": "anomaly_diagnosis", "reason": "45 分钟窗口的异常检测。", "tool": "detect_anomaly", "metric_name": "fan_power", "time_window": "last_45_minutes"}]},
    {"question": "Show cooling_power over the past 15 minutes.",
     "steps": [{"route": "timeseries_query", "reason": "15-minute trend visualization.", "tool": "plot_metric_trend", "metric_name": "cooling_power", "time_window": "last_15_minutes"}]},
    # colloquial named windows
    {"question": "现在 zone_0 的温度是多少？",
     "steps": [{"route": "timeseries_query", "reason": "查询当前最新温度值。", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "latest"}]},
    {"question": "当前 pue 是多少？",
     "steps": [{"route": "timeseries_query", "reason": "查询最新 PUE。", "tool": "query_metric", "metric_name": "pue", "time_window": "latest"}]},
    {"question": "把全部历史的 hvac_power 画成趋势图。",
     "steps": [{"route": "timeseries_query", "reason": "全时段功率趋势可视化。", "tool": "plot_metric_trend", "metric_name": "hvac_power", "time_window": "full_demo_range"}]},
    {"question": "整段数据里 chiller_power 的均值。",
     "steps": [{"route": "timeseries_query", "reason": "全时段冷机功率均值。", "tool": "query_metric", "metric_name": "chiller_power", "time_window": "full_demo_range"}]},
    # multi-step with natural-language windows (both steps carry the window)
    {"question": "对比过去一周和上个月的 zone_temperature，再检测异常，然后给策略。",
     "steps": [{"route": "timeseries_query", "reason": "比较两个时间段的温度均值。", "tool": "compare_period", "metric_name": "zone_temperature", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测温度异常。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "time_window": "last_720_hours"}, {"route": "policy_recommendation", "reason": "给出温控建议。", "tool": "policy_runner"}]},
    {"question": "看过去三天 cooling_power 趋势，检查有没有异常，给节能建议。",
     "steps": [{"route": "timeseries_query", "reason": "三天冷却功率趋势。", "tool": "plot_metric_trend", "metric_name": "cooling_power", "time_window": "last_72_hours"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "cooling_power", "time_window": "last_72_hours"}, {"route": "policy_recommendation", "reason": "给节能策略。", "tool": "policy_runner"}]},
    {"question": "Check the past week of hvac_power for anomalies and recommend a fix.",
     "steps": [{"route": "anomaly_diagnosis", "reason": "One-week anomaly scan on HVAC power.", "tool": "detect_anomaly", "metric_name": "hvac_power", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "Recommend a corrective control policy.", "tool": "policy_runner"}]},
    {"question": "过去两天 zone_1 舒适度风险如何，画个温度趋势佐证，再给策略。",
     "steps": [{"route": "anomaly_diagnosis", "reason": "两天舒适度风险评估。", "tool": "comfort_risk_assessment", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_48_hours"}, {"route": "timeseries_query", "reason": "温度趋势作为佐证。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_48_hours"}, {"route": "policy_recommendation", "reason": "给出控制建议。", "tool": "policy_runner"}]},
]

# --- Group 2: tool disambiguation within timeseries_query --------------------
# Phrasing maps to exactly one tool; these reduce the "illegal / wrong tool"
# fallbacks by making the boundaries between the eight timeseries tools crisp.
TOOL_ROWS: list[dict] = [
    # query_metric: a single aggregate value
    {"question": "chiller_power 的平均值是多少？",
     "steps": [{"route": "timeseries_query", "reason": "只要单一指标的统计均值。", "tool": "query_metric", "metric_name": "chiller_power"}]},
    {"question": "告诉我 zone_2 温度的最大值。",
     "steps": [{"route": "timeseries_query", "reason": "查询单一指标的统计极值。", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_2"}]},
    # compare_period: two periods
    {"question": "对比一下检修前后 fan_power 的变化。",
     "steps": [{"route": "timeseries_query", "reason": "比较两个时间段的指标均值。", "tool": "compare_period", "metric_name": "fan_power"}]},
    {"question": "Compare cooling_power between this week and last week.",
     "steps": [{"route": "timeseries_query", "reason": "Two-period comparison of a metric.", "tool": "compare_period", "metric_name": "cooling_power"}]},
    {"question": "白天和夜间的 pue 差多少？",
     "steps": [{"route": "timeseries_query", "reason": "比较两个时段的 PUE 均值。", "tool": "compare_period", "metric_name": "pue"}]},
    # plot_metric_trend: a chart / line
    {"question": "帮我画 hvac_power 的趋势折线。",
     "steps": [{"route": "timeseries_query", "reason": "用户要趋势序列用于折线可视化。", "tool": "plot_metric_trend", "metric_name": "hvac_power"}]},
    {"question": "把 zone_0 温度可视化出来。",
     "steps": [{"route": "timeseries_query", "reason": "可视化即绘制趋势图。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_0"}]},
    # compute_energy_breakdown: energy split, no single metric
    {"question": "分析一下整体能耗构成。",
     "steps": [{"route": "timeseries_query", "reason": "拆分各分项能耗占比。", "tool": "compute_energy_breakdown"}]},
    {"question": "各项功率分别占多少能耗？",
     "steps": [{"route": "timeseries_query", "reason": "计算能耗分项构成。", "tool": "compute_energy_breakdown"}]},
    {"question": "Break down where the energy is going.",
     "steps": [{"route": "timeseries_query", "reason": "Energy breakdown across components.", "tool": "compute_energy_breakdown"}]},
    # zone_hotspot_rank: rank hottest zones
    {"question": "哪几个 zone 最热，排个序。",
     "steps": [{"route": "timeseries_query", "reason": "按温度对各 zone 做热点排名。", "tool": "zone_hotspot_rank", "metric_name": "zone_temperature"}]},
    {"question": "Rank the zones by temperature to find the hotspots.",
     "steps": [{"route": "timeseries_query", "reason": "Rank zones to surface hotspots.", "tool": "zone_hotspot_rank", "metric_name": "zone_temperature"}]},
    {"question": "找出温度最高的几个区域。",
     "steps": [{"route": "timeseries_query", "reason": "热点区域排名。", "tool": "zone_hotspot_rank", "metric_name": "zone_temperature"}]},
    # cooling_efficiency_summary
    {"question": "评估一下制冷效率怎么样。",
     "steps": [{"route": "timeseries_query", "reason": "汇总制冷效率指标。", "tool": "cooling_efficiency_summary", "metric_name": "cooling_power"}]},
    {"question": "How efficient is the cooling right now?",
     "steps": [{"route": "timeseries_query", "reason": "Summarize cooling efficiency.", "tool": "cooling_efficiency_summary", "metric_name": "cooling_power"}]},
    # control_action_audit
    {"question": "复盘一下最近的控制动作有没有频繁调整。",
     "steps": [{"route": "timeseries_query", "reason": "审计控制动作的变化。", "tool": "control_action_audit", "metric_name": "control_action"}]},
    {"question": "Audit the control_action history for large swings.",
     "steps": [{"route": "timeseries_query", "reason": "Audit control actions for large changes.", "tool": "control_action_audit", "metric_name": "control_action"}]},
    # data_quality_check
    {"question": "检查一下数据有没有缺失。",
     "steps": [{"route": "timeseries_query", "reason": "检查数据完整性与缺失。", "tool": "data_quality_check"}]},
    {"question": "数据质量怎么样，有没有采样间隙？",
     "steps": [{"route": "timeseries_query", "reason": "校验采样频率与数据质量。", "tool": "data_quality_check"}]},
    # anomaly vs comfort disambiguation
    {"question": "zone_1 有没有温度异常？",
     "steps": [{"route": "anomaly_diagnosis", "reason": "检测温度异常点。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "zone_id": "zone_1"}]},
    {"question": "评估 zone_0 的热舒适风险。",
     "steps": [{"route": "anomaly_diagnosis", "reason": "评估舒适度越限风险。", "tool": "comfort_risk_assessment", "metric_name": "zone_temperature", "zone_id": "zone_0"}]},
    # document_qa
    {"question": "规范里温度上限是多少度？",
     "steps": [{"route": "document_qa", "reason": "查询标准文档中的温度阈值。", "tool": "rag_retrieval"}]},
    {"question": "What does the SOP say about humidity limits?",
     "steps": [{"route": "document_qa", "reason": "Retrieve humidity limits from the SOP.", "tool": "rag_retrieval"}]},
    # policy last-step ordering with tool disambiguation
    {"question": "先看能耗构成，再检测异常，最后给节能策略。",
     "steps": [{"route": "timeseries_query", "reason": "能耗分项构成。", "tool": "compute_energy_breakdown"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "hvac_power"}, {"route": "policy_recommendation", "reason": "给节能策略。", "tool": "policy_runner"}]},
    {"question": "查文档标准，评估舒适风险，再给控制建议。",
     "steps": [{"route": "document_qa", "reason": "查标准文档。", "tool": "rag_retrieval"}, {"route": "anomaly_diagnosis", "reason": "评估舒适度风险。", "tool": "comfort_risk_assessment", "metric_name": "zone_temperature"}, {"route": "policy_recommendation", "reason": "给控制建议。", "tool": "policy_runner"}]},
]

# --- Group 3: multi-step compound tasks WITH natural-language windows --------
# The eval set (compound_task_eval) is entirely compound tasks, and the residual
# fallbacks occur there — so the model must normalize windows *while* juggling
# 2-4 steps, not just in isolated single-step queries. These rehearse exactly
# that: a natural-language window in the question, folded to the canonical hours
# label, inside a realistic multi-route plan (policy always last).
COMPOUND_ROWS: list[dict] = [
    {"question": "看过去一周 zone_1 温度趋势，检测异常，再给温控策略。",
     "steps": [{"route": "timeseries_query", "reason": "一周温度趋势可视化。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测温度异常。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给温控建议。", "tool": "policy_runner"}]},
    {"question": "过去三天 cooling_power 的均值查一下，再看有没有异常。",
     "steps": [{"route": "timeseries_query", "reason": "三天冷却功率均值。", "tool": "query_metric", "metric_name": "cooling_power", "time_window": "last_72_hours"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "cooling_power", "time_window": "last_72_hours"}]},
    {"question": "Look at the past month of pue, check for anomalies, and recommend an optimization.",
     "steps": [{"route": "timeseries_query", "reason": "One-month PUE trend.", "tool": "plot_metric_trend", "metric_name": "pue", "time_window": "last_720_hours"}, {"route": "anomaly_diagnosis", "reason": "Detect PUE anomalies.", "tool": "detect_anomaly", "metric_name": "pue", "time_window": "last_720_hours"}, {"route": "policy_recommendation", "reason": "Recommend an efficiency policy.", "tool": "policy_runner"}]},
    {"question": "对比过去一周和更早的 fan_power，再检测异常，给节能建议。",
     "steps": [{"route": "timeseries_query", "reason": "比较两个时间段的风机功率。", "tool": "compare_period", "metric_name": "fan_power", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "fan_power", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给节能策略。", "tool": "policy_runner"}]},
    {"question": "过去两天 zone_0 舒适风险如何，画温度趋势佐证，再给建议。",
     "steps": [{"route": "anomaly_diagnosis", "reason": "两天舒适度风险评估。", "tool": "comfort_risk_assessment", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "last_48_hours"}, {"route": "timeseries_query", "reason": "温度趋势佐证。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "last_48_hours"}, {"route": "policy_recommendation", "reason": "给控制建议。", "tool": "policy_runner"}]},
    {"question": "先查规范里的温度上限，再看最近一周 zone_2 温度趋势，最后给策略。",
     "steps": [{"route": "document_qa", "reason": "查温度上限标准。", "tool": "rag_retrieval"}, {"route": "timeseries_query", "reason": "一周温度趋势对照标准。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_2", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "结合标准给建议。", "tool": "policy_runner"}]},
    {"question": "过去一个月 hvac_power 能耗构成，找异常，给综合优化策略。",
     "steps": [{"route": "timeseries_query", "reason": "一个月能耗分项构成。", "tool": "compute_energy_breakdown", "time_window": "last_720_hours"}, {"route": "anomaly_diagnosis", "reason": "检测 HVAC 功率异常。", "tool": "detect_anomaly", "metric_name": "hvac_power", "time_window": "last_720_hours"}, {"route": "policy_recommendation", "reason": "给综合优化策略。", "tool": "policy_runner"}]},
    {"question": "Check the last 3 days of chiller_power for anomalies and suggest a control change.",
     "steps": [{"route": "anomaly_diagnosis", "reason": "Three-day anomaly scan on chiller power.", "tool": "detect_anomaly", "metric_name": "chiller_power", "time_window": "last_72_hours"}, {"route": "policy_recommendation", "reason": "Suggest a control adjustment.", "tool": "policy_runner"}]},
    {"question": "过去48小时的 zone_1 温度均值，再评估舒适风险，给策略。",
     "steps": [{"route": "timeseries_query", "reason": "48 小时温度均值。", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_48_hours"}, {"route": "anomaly_diagnosis", "reason": "评估舒适度风险。", "tool": "comfort_risk_assessment", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_48_hours"}, {"route": "policy_recommendation", "reason": "给温控建议。", "tool": "policy_runner"}]},
    {"question": "最近一周数据质量如何，有异常吗，需要调整策略吗？",
     "steps": [{"route": "timeseries_query", "reason": "检查一周数据质量。", "tool": "data_quality_check", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测异常。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "按需给策略。", "tool": "policy_runner"}]},
    {"question": "过去两周 cooling_power 效率怎么样，检测异常，给优化建议。",
     "steps": [{"route": "timeseries_query", "reason": "两周制冷效率汇总。", "tool": "cooling_efficiency_summary", "metric_name": "cooling_power", "time_window": "last_336_hours"}, {"route": "anomaly_diagnosis", "reason": "检测效率异常。", "tool": "detect_anomaly", "metric_name": "cooling_power", "time_window": "last_336_hours"}, {"route": "policy_recommendation", "reason": "给优化策略。", "tool": "policy_runner"}]},
    {"question": "看今天各 zone 哪个最热，检测温度异常，再给降温策略。",
     "steps": [{"route": "timeseries_query", "reason": "今天热点区域排名。", "tool": "zone_hotspot_rank", "metric_name": "zone_temperature", "time_window": "last_24_hours"}, {"route": "anomaly_diagnosis", "reason": "检测温度异常。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "time_window": "last_24_hours"}, {"route": "policy_recommendation", "reason": "给降温策略。", "tool": "policy_runner"}]},
    {"question": "过去90分钟 fan_power 有没有异常波动，给建议。",
     "steps": [{"route": "anomaly_diagnosis", "reason": "90 分钟窗口异常检测。", "tool": "detect_anomaly", "metric_name": "fan_power", "time_window": "last_90_minutes"}, {"route": "policy_recommendation", "reason": "给控制建议。", "tool": "policy_runner"}]},
    {"question": "Compare this week vs earlier hvac_power, find anomalies, recommend a policy.",
     "steps": [{"route": "timeseries_query", "reason": "Two-period HVAC power comparison.", "tool": "compare_period", "metric_name": "hvac_power", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "Detect anomalies.", "tool": "detect_anomaly", "metric_name": "hvac_power", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "Recommend a policy.", "tool": "policy_runner"}]},
    {"question": "过去三天审计控制动作，看有没有异常调整，再给建议。",
     "steps": [{"route": "timeseries_query", "reason": "三天控制动作审计。", "tool": "control_action_audit", "metric_name": "control_action", "time_window": "last_72_hours"}, {"route": "anomaly_diagnosis", "reason": "检测控制动作异常。", "tool": "detect_anomaly", "metric_name": "control_action", "time_window": "last_72_hours"}, {"route": "policy_recommendation", "reason": "给控制建议。", "tool": "policy_runner"}]},
    {"question": "最近一个月 it_load 趋势画出来，检测异常，给容量策略。",
     "steps": [{"route": "timeseries_query", "reason": "一个月 IT 负载趋势。", "tool": "plot_metric_trend", "metric_name": "it_load", "time_window": "last_720_hours"}, {"route": "anomaly_diagnosis", "reason": "检测负载异常。", "tool": "detect_anomaly", "metric_name": "it_load", "time_window": "last_720_hours"}, {"route": "policy_recommendation", "reason": "给容量调整策略。", "tool": "policy_runner"}]},
    {"question": "过去半天 zone_2 温度趋势，评估舒适风险，给策略。",
     "steps": [{"route": "timeseries_query", "reason": "12 小时温度趋势。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_2", "time_window": "last_12_hours"}, {"route": "anomaly_diagnosis", "reason": "评估舒适度风险。", "tool": "comfort_risk_assessment", "metric_name": "zone_temperature", "zone_id": "zone_2", "time_window": "last_12_hours"}, {"route": "policy_recommendation", "reason": "给温控建议。", "tool": "policy_runner"}]},
    {"question": "查一下过去5天 outdoor_temp 和 zone_0 温度趋势，检测异常。",
     "steps": [{"route": "timeseries_query", "reason": "5 天温度趋势。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "last_120_hours"}, {"route": "anomaly_diagnosis", "reason": "检测温度异常。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "last_120_hours"}]},
    {"question": "过去一周 humidity 数据质量查一下，有异常给除湿策略。",
     "steps": [{"route": "timeseries_query", "reason": "一周湿度数据质量。", "tool": "data_quality_check", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测湿度异常。", "tool": "detect_anomaly", "metric_name": "humidity", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给除湿策略。", "tool": "policy_runner"}]},
    {"question": "Over the past two weeks, break down energy, check anomalies, and give a policy.",
     "steps": [{"route": "timeseries_query", "reason": "Two-week energy breakdown.", "tool": "compute_energy_breakdown", "time_window": "last_336_hours"}, {"route": "anomaly_diagnosis", "reason": "Detect power anomalies.", "tool": "detect_anomaly", "metric_name": "hvac_power", "time_window": "last_336_hours"}, {"route": "policy_recommendation", "reason": "Give an energy policy.", "tool": "policy_runner"}]},
]

# --- Group 4: 24 more multi-step compound tasks (distinct routes) -----------
# Bring the A1 batch to 100 rows, all multi-step, all with distinct routes so
# the guard never collapses a tool. Mixes natural-language windows, tool
# disambiguation, 2/3/4-step plans, EN/ZH, and every route ordering constraint
# (policy always last). These are the highest-value shape for the eval set,
# which is entirely compound tasks.
COMPOUND_ROWS_2: list[dict] = [
    # 4-step: doc -> quality -> anomaly -> policy
    {"question": "先查文档标准，再检查数据质量，检测温度异常，最后给策略。",
     "steps": [{"route": "document_qa", "reason": "查标准文档。", "tool": "rag_retrieval"}, {"route": "timeseries_query", "reason": "检查数据质量。", "tool": "data_quality_check"}, {"route": "anomaly_diagnosis", "reason": "检测温度异常。", "tool": "detect_anomaly", "metric_name": "zone_temperature"}, {"route": "policy_recommendation", "reason": "给综合策略。", "tool": "policy_runner"}]},
    {"question": "过去一周先看能耗构成，检查数据质量，检测功率异常，给节能策略。",
     "steps": [{"route": "timeseries_query", "reason": "一周能耗分项。", "tool": "compute_energy_breakdown", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "hvac_power", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给节能策略。", "tool": "policy_runner"}]},
    {"question": "Check the SOP humidity limits, assess comfort risk over the past 3 days, then recommend a fix.",
     "steps": [{"route": "document_qa", "reason": "Retrieve humidity limits.", "tool": "rag_retrieval"}, {"route": "anomaly_diagnosis", "reason": "Three-day comfort risk assessment.", "tool": "comfort_risk_assessment", "metric_name": "humidity", "time_window": "last_72_hours"}, {"route": "policy_recommendation", "reason": "Recommend a fix.", "tool": "policy_runner"}]},
    # timeseries (query) + anomaly, no policy
    {"question": "过去一个月 pue 的均值查一下，再看有没有效率异常。",
     "steps": [{"route": "timeseries_query", "reason": "一个月 PUE 均值。", "tool": "query_metric", "metric_name": "pue", "time_window": "last_720_hours"}, {"route": "anomaly_diagnosis", "reason": "检测效率异常。", "tool": "detect_anomaly", "metric_name": "pue", "time_window": "last_720_hours"}]},
    {"question": "对比过去两周和更早的 chiller_power，再检测异常。",
     "steps": [{"route": "timeseries_query", "reason": "两周功率对比。", "tool": "compare_period", "metric_name": "chiller_power", "time_window": "last_336_hours"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "chiller_power", "time_window": "last_336_hours"}]},
    # efficiency -> anomaly -> policy
    {"question": "过去一周制冷效率如何，检测异常，给优化建议。",
     "steps": [{"route": "timeseries_query", "reason": "一周制冷效率汇总。", "tool": "cooling_efficiency_summary", "metric_name": "cooling_power", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测效率异常。", "tool": "detect_anomaly", "metric_name": "cooling_power", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给优化策略。", "tool": "policy_runner"}]},
    {"question": "Look at cooling efficiency over the last month, check anomalies, and suggest an optimization.",
     "steps": [{"route": "timeseries_query", "reason": "One-month efficiency summary.", "tool": "cooling_efficiency_summary", "metric_name": "cooling_power", "time_window": "last_720_hours"}, {"route": "anomaly_diagnosis", "reason": "Detect efficiency anomalies.", "tool": "detect_anomaly", "metric_name": "cooling_power", "time_window": "last_720_hours"}, {"route": "policy_recommendation", "reason": "Suggest an optimization.", "tool": "policy_runner"}]},
    # hotspot -> anomaly -> policy
    {"question": "过去三天哪个 zone 最热，检测异常，给降温策略。",
     "steps": [{"route": "timeseries_query", "reason": "三天热点区域排名。", "tool": "zone_hotspot_rank", "metric_name": "zone_temperature", "time_window": "last_72_hours"}, {"route": "anomaly_diagnosis", "reason": "检测温度异常。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "time_window": "last_72_hours"}, {"route": "policy_recommendation", "reason": "给降温策略。", "tool": "policy_runner"}]},
    # control audit -> anomaly -> policy
    {"question": "过去一周控制动作审计一下，检测异常波动，给稳定控制建议。",
     "steps": [{"route": "timeseries_query", "reason": "一周控制动作审计。", "tool": "control_action_audit", "metric_name": "control_action", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测控制动作异常。", "tool": "detect_anomaly", "metric_name": "control_action", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给稳定控制建议。", "tool": "policy_runner"}]},
    # doc -> query -> policy with window
    {"question": "查规范里的温度上限，再看过去两天 zone_1 温度均值，给策略。",
     "steps": [{"route": "document_qa", "reason": "查温度上限标准。", "tool": "rag_retrieval"}, {"route": "timeseries_query", "reason": "两天温度均值对照标准。", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_48_hours"}, {"route": "policy_recommendation", "reason": "结合标准给建议。", "tool": "policy_runner"}]},
    {"question": "Check the SOP temperature limit, plot the past week of zone_0 temperature, then advise.",
     "steps": [{"route": "document_qa", "reason": "Retrieve temperature limit.", "tool": "rag_retrieval"}, {"route": "timeseries_query", "reason": "One-week temperature trend.", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "Advise based on the standard.", "tool": "policy_runner"}]},
    # quality -> anomaly (no policy)
    {"question": "过去一个月 humidity 数据质量如何，有没有异常。",
     "steps": [{"route": "timeseries_query", "reason": "一个月湿度数据质量。", "tool": "data_quality_check", "time_window": "last_720_hours"}, {"route": "anomaly_diagnosis", "reason": "检测湿度异常。", "tool": "detect_anomaly", "metric_name": "humidity", "time_window": "last_720_hours"}]},
    # comfort -> query trend -> policy
    {"question": "评估 zone_2 过去一周的舒适风险，画温度趋势佐证，再给策略。",
     "steps": [{"route": "anomaly_diagnosis", "reason": "一周舒适度风险评估。", "tool": "comfort_risk_assessment", "metric_name": "zone_temperature", "zone_id": "zone_2", "time_window": "last_168_hours"}, {"route": "timeseries_query", "reason": "温度趋势佐证。", "tool": "plot_metric_trend", "metric_name": "zone_temperature", "zone_id": "zone_2", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给温控建议。", "tool": "policy_runner"}]},
    # energy compare + anomaly + policy
    {"question": "对比这周和上周的能耗，检测功率异常，给节能建议。",
     "steps": [{"route": "timeseries_query", "reason": "两周能耗对比。", "tool": "compare_period", "metric_name": "hvac_power", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "hvac_power", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给节能策略。", "tool": "policy_runner"}]},
    # minutes window multi-step
    {"question": "过去30分钟 fan_power 趋势看一下，有异常给建议。",
     "steps": [{"route": "timeseries_query", "reason": "30 分钟功率趋势。", "tool": "plot_metric_trend", "metric_name": "fan_power", "time_window": "last_30_minutes"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "fan_power", "time_window": "last_30_minutes"}, {"route": "policy_recommendation", "reason": "按需给建议。", "tool": "policy_runner"}]},
    {"question": "最近一小时 zone_1 温度均值查下，再评估舒适风险。",
     "steps": [{"route": "timeseries_query", "reason": "1 小时温度均值。", "tool": "query_metric", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_1_hours"}, {"route": "anomaly_diagnosis", "reason": "评估舒适度风险。", "tool": "comfort_risk_assessment", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_1_hours"}]},
    # it_load related
    {"question": "过去一周 it_load 趋势，检测负载异常，给容量策略。",
     "steps": [{"route": "timeseries_query", "reason": "一周 IT 负载趋势。", "tool": "plot_metric_trend", "metric_name": "it_load", "time_window": "last_168_hours"}, {"route": "anomaly_diagnosis", "reason": "检测负载异常。", "tool": "detect_anomaly", "metric_name": "it_load", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给容量调整策略。", "tool": "policy_runner"}]},
    # outdoor_temp context + zone
    {"question": "过去两天 outdoor_temp 均值查一下，再看 zone_0 温度有没有异常。",
     "steps": [{"route": "timeseries_query", "reason": "两天室外温度均值。", "tool": "query_metric", "metric_name": "outdoor_temp", "time_window": "last_48_hours"}, {"route": "anomaly_diagnosis", "reason": "检测区域温度异常。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "zone_id": "zone_0", "time_window": "last_48_hours"}]},
    # full-range multi-step
    {"question": "全时段能耗构成算一下，检测功率异常，给综合优化策略。",
     "steps": [{"route": "timeseries_query", "reason": "全时段能耗分项。", "tool": "compute_energy_breakdown", "time_window": "full_demo_range"}, {"route": "anomaly_diagnosis", "reason": "检测功率异常。", "tool": "detect_anomaly", "metric_name": "hvac_power", "time_window": "full_demo_range"}, {"route": "policy_recommendation", "reason": "给综合优化策略。", "tool": "policy_runner"}]},
    {"question": "Summarize the whole-range cooling efficiency, detect anomalies, and recommend a policy.",
     "steps": [{"route": "timeseries_query", "reason": "Full-range efficiency summary.", "tool": "cooling_efficiency_summary", "metric_name": "cooling_power", "time_window": "full_demo_range"}, {"route": "anomaly_diagnosis", "reason": "Detect anomalies.", "tool": "detect_anomaly", "metric_name": "cooling_power", "time_window": "full_demo_range"}, {"route": "policy_recommendation", "reason": "Recommend a policy.", "tool": "policy_runner"}]},
    # doc + anomaly (no policy)
    {"question": "查一下告警阈值文档，再看过去一周 zone_2 温度有没有超限。",
     "steps": [{"route": "document_qa", "reason": "查告警阈值标准。", "tool": "rag_retrieval"}, {"route": "anomaly_diagnosis", "reason": "检测温度越限。", "tool": "detect_anomaly", "metric_name": "zone_temperature", "zone_id": "zone_2", "time_window": "last_168_hours"}]},
    # compare + policy directly
    {"question": "对比过去三天和更早的 zone_1 温度，给温控调整建议。",
     "steps": [{"route": "timeseries_query", "reason": "三天温度对比。", "tool": "compare_period", "metric_name": "zone_temperature", "zone_id": "zone_1", "time_window": "last_72_hours"}, {"route": "policy_recommendation", "reason": "给温控调整建议。", "tool": "policy_runner"}]},
    # quality + policy
    {"question": "过去一周数据质量检查一下，如果没问题给个常规巡检策略。",
     "steps": [{"route": "timeseries_query", "reason": "一周数据质量检查。", "tool": "data_quality_check", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "给常规巡检策略。", "tool": "policy_runner"}]},
    # hotspot + policy
    {"question": "Rank the hottest zones over the past week and recommend a cooling policy.",
     "steps": [{"route": "timeseries_query", "reason": "One-week hotspot ranking.", "tool": "zone_hotspot_rank", "metric_name": "zone_temperature", "time_window": "last_168_hours"}, {"route": "policy_recommendation", "reason": "Recommend a cooling policy.", "tool": "policy_runner"}]},
]

NEW_ROWS: list[dict] = TIME_WINDOW_ROWS + TOOL_ROWS + COMPOUND_ROWS + COMPOUND_ROWS_2


def _to_plan_steps(raw_steps: list[dict]) -> list[PlanStep]:
    return [
        PlanStep(
            route=s["route"],
            reason=s.get("reason", "hand-labeled plan step"),
            tool=s.get("tool"),
            metric_name=s.get("metric_name"),
            zone_id=s.get("zone_id"),
            time_window=s.get("time_window"),
        )
        for s in raw_steps
    ]


def _load_existing(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def _next_id(existing: list[dict]) -> int:
    nums = [int(r["id"].split("_")[1]) for r in existing if r.get("id", "").startswith("gold_")]
    return (max(nums) + 1) if nums else 1


def build(append: bool) -> None:
    existing = _load_existing(GOLD_PATH)
    existing_questions = {r["question"].strip() for r in existing}
    existing_ids = {r["id"] for r in existing}

    # Validate every new row through the live guard and enforce uniqueness.
    # Rows whose question is already in the file are SKIPPED (idempotent
    # re-runs / incremental batches); only genuine defects are errors.
    validated: list[dict] = []
    errors: list[str] = []
    skipped = 0
    seen_questions: set[str] = set()
    for i, row in enumerate(NEW_ROWS):
        q = row["question"].strip()
        if not q:
            errors.append(f"row {i}: empty question")
            continue
        if q in seen_questions:
            errors.append(f"row {i}: duplicate question within batch: {q!r}")
            continue
        seen_questions.add(q)
        if q in existing_questions:
            skipped += 1
            continue
        try:
            validated_steps = validate_plan_steps(_to_plan_steps(row["steps"]))
        except ValueError as exc:
            errors.append(f"row {i} ({q!r}): {exc}")
            continue
        # The guard dedupes steps by route (keeps the first per route), so a
        # plan with two steps sharing a route silently loses the second tool —
        # the label would then not match the question. Reject that here so a
        # multi-step row must use distinct routes.
        if len(validated_steps) != len(row["steps"]):
            errors.append(
                f"row {i} ({q!r}): {len(row['steps'])} authored steps collapsed to "
                f"{len(validated_steps)} (duplicate route drops a tool); use distinct routes"
            )
            continue
        validated.append(row)

    print(
        f"[a1] candidates={len(NEW_ROWS)} new-valid={len(validated)} "
        f"already-present={skipped} errors={len(errors)}"
    )
    for e in errors:
        print(f"[invalid] {e}")
    if errors:
        raise SystemExit("A1 rows have validation/uniqueness errors; fix before appending.")

    if not validated:
        print("[a1] nothing new to append (all rows already present).")
        return

    start = _next_id(existing)
    assigned = []
    for offset, row in enumerate(validated):
        gid = f"gold_{start + offset:04d}"
        assert gid not in existing_ids, f"id collision: {gid}"
        assigned.append({"id": gid, "question": row["question"], "steps": row["steps"]})

    print(f"[a1] would append {len(assigned)} rows as {assigned[0]['id']}..{assigned[-1]['id']}")
    if not append:
        print("[a1] dry-run (pass --append to write).")
        return

    with Path(GOLD_PATH).open("a", encoding="utf-8") as f:
        for row in assigned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[a1] appended {len(assigned)} rows to {GOLD_PATH} (total now {len(existing) + len(assigned)}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append", action="store_true", help="write rows to gold_labeled.jsonl")
    args = parser.parse_args()
    build(args.append)


if __name__ == "__main__":
    main()
