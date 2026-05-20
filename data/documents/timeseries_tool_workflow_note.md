# Time-Series Tool Workflow Note

## 适用边界

本文说明本项目时序工具的使用顺序。它适用于 BEAR HVAC 仿真轨迹、BEAR sample CSV 或 mock fallback，不代表真实生产监控平台。

## 工具流程

当问题询问最近、某区域、最大值、平均值或数据覆盖范围时，优先调用 query_metric。当问题询问前后变化、维护窗口前后差异或策略前后差异时，调用 compare_period。当问题询问异常升高、离群点或告警候选时，调用 detect_anomaly。

当问题询问能耗构成时，调用 compute_energy_breakdown。当问题需要展示趋势或 demo 图表时，调用 plot_metric_trend。回答应保留工具名称、时间范围、zone_id、count 和 summary。

## 可检索要点

- query_metric 适合最大值、平均值和覆盖范围。
- compare_period 适合前后窗口对比。
- detect_anomaly 适合异常点和告警候选。
- plot_metric_trend 适合趋势图展示。
