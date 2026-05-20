# Redundancy Maintenance Alarm Note

## 适用边界

本文用于说明冷却冗余、维护窗口和告警优先级。它是运维领域说明，可用于构造 RAG 与异常诊断评测，不代表真实数据中心工单。

## 运行要点

当冷却系统进入维护窗口时，N+1 redundancy 或 backup cooling capacity 会影响告警优先级。单个 zone_temperature 高点不一定代表容量不足，需要结合持续时间、相邻区域、control_action、comfort violation 和是否存在维护窗口。

如果冗余能力下降且异常持续，系统应优先调用 detect_anomaly 或 compare_period 定位证据，再解释风险。LLM 不应编造设备故障或维护记录。

## 可检索要点

- redundancy 和 maintenance window 会改变告警优先级。
- 告警判断需要持续时间、相邻区域和控制动作证据。
- 不应编造真实工单或设备故障。
