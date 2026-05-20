# Return Air Delta-T Operations Note

## 适用边界

本文用于说明机柜或区域维度的 Delta-T、回风温度和热交换效率。它是领域知识说明，可辅助解释 BEAR HVAC 仿真轨迹，但不是生产数据中心测点说明。

## 运行含义

rack delta-t 或 return air differential 反映送风与回风之间的温差关系。较低的 Delta-T 可能意味着气流旁路、过度送风或冷热气流混合；较高的 Delta-T 可能意味着负载集中、局部热点或风量不足。判断时需要同时查看 supply/return temperature delta、zone_temperature trend、fan_power 或 control_action。

Delta-T 证据适合与 containment 证据一起使用。若冷通道封闭改善后 Delta-T 上升且热点没有增加，可能说明热交换效率改善；若 Delta-T 异常升高并伴随 comfort violation，则应优先定位热点风险。

## 可检索要点

- return air differential 需要结合 zone_temperature trend。
- rack delta-t 可提示气流旁路、局部热点或风量不足。
- control_action 和 fan_power 是判断 Delta-T 变化原因的重要证据。
