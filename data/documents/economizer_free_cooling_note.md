# Economizer Free Cooling Note

## 适用边界

本文用于解释数据中心冷却类问题中的 economizer、free cooling 和室外空气利用。它可辅助分析 BEAR HVAC 仿真轨迹上的冷却策略问题，但 BEAR 在本项目中仍是物理仿真/可控代理场景，不是真实数据中心生产遥测。

## 运行要点

economizer 或 free cooling 可能在室外温度较低、湿度风险可控时降低机械制冷能耗。判断时不能只看 outdoor_temp，还应结合 zone_temperature、humidity 是否为可解释字段、airflow containment、filter pressure 或告警风险。

如果数据集中没有可复现的湿度或空气质量字段，应明确说明证据缺口。Agent 可以解释何时需要 free cooling 证据，但控制建议仍必须来自 policy 工具或 offline replay。

## 可检索要点

- economizer 需要结合 outdoor_temp 和区域温度趋势。
- free cooling 风险包括湿度、空气质量和过滤压差。
- BEAR 仿真轨迹不能被表述为真实生产遥测。
