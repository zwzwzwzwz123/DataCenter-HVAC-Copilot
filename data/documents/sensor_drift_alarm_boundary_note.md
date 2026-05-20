# Sensor Drift Alarm Boundary Note

## 适用边界

本文用于解释传感器漂移、告警误判和异常诊断证据。项目中的 BEAR 轨迹来自 HVAC 仿真或样例 CSV fallback，不能写成真实生产遥测。

## 诊断思路

当某个 zone_temperature 或 alarm_flag 显示异常时，系统不应只依据单个传感器读数给出结论。更稳妥的做法是先调用 detect_anomaly 定位异常时间点，再比较相邻区域、outdoor_temp、internal_load 和 control_action。如果高温点只出现在单个通道，并且负载与控制动作没有同步变化，应考虑 sensor drift、采样缺失或告警阈值误判。

如果证据不足，回答应明确说明 uncertainty，不应编造设备故障、真实工单或生产事件。异常诊断要保留工具结果和引用上下文。

## 可检索要点

- sensor drift 需要和相邻区域及控制动作交叉验证。
- alarm false positive 不能只看单点温度。
- detect_anomaly 是异常诊断的第一层证据。
