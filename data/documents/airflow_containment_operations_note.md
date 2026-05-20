# Airflow Containment Operations Note

Source type: project internal note.
Published at: 2026.
Category: thermal_management.

冷热通道封闭的目标是减少 supply air 与 return air 的混合。对于数据中心冷却叙事，containment 证据不能单独说明系统健康，仍需要结合 zone temperature trend、internal load、airflow balance 和 recent control action 变化。

当出现局部 hotspot 时，应优先检查区域温度趋势、负载变化和控制响应是否同步。如果风量或冷却动作增加后温度没有下降，可能存在回流、旁通气流、传感器位置偏差或响应滞后。该判断必须来自时序证据和文档证据的组合，而不是只看单点平均温度。

本说明是项目内部测试文档，用于构造与 thermal_management_note 相似的检索主题，帮助比较 keyword retrieval 与 BM25-style hybrid retrieval 在相近主题文档上的表现。
