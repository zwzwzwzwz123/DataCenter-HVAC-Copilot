# Sensor Missing Data Quality Note

## 适用边界

本文用于说明传感器缺失、采样间隔不齐和数据质量问题。它适用于 BEAR-like 仿真轨迹和样例 CSV 的质量检查，不代表真实生产监控系统。

## 诊断要点

时序分析前应检查 timestamp、zone_id、metric_name 和 count。若某个区域缺少观测值，query_metric 应返回结构化摘要，回答应说明数据不足，而不是推断真实设备状态。

缺失数据可能导致异常检测误报或漏报。对于 sensor missing、NaN、采样窗口过短等情况，应先报告样本数量和时间范围，再考虑 compare_period 或 detect_anomaly。

## 可检索要点

- missing data 需要先看 count 和时间范围。
- 异常诊断应报告数据不足和证据缺口。
- 不应从缺失观测值推断真实设备故障。
