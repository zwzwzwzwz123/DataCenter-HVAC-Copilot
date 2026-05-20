# Liquid Air Hybrid Cooling Note

## 适用边界

本文讨论液冷与风冷混合过渡场景，用于数据中心冷却知识检索。当前项目没有真实液冷生产测点，BEAR 仍只作为 HVAC 仿真和可控代理场景。

## 运行要点

液冷与风冷混合时，air-side cooling 的 zone_temperature 仍有价值，但不能单独代表冷板、CDU 或液路状态。若用户询问 liquid cooling、CDU 或 coolant loop，回答应说明当前数据契约没有这些原生字段，除非后续 adapter 明确接入。

混合冷却评估可以比较风侧负载下降、局部热点是否减少、fan_power 是否下降，以及 comfort violation 是否变化。控制动作仍必须来自 policy adapter 或 offline replay。

## 可检索要点

- liquid cooling 字段不是当前 BEAR 原生字段。
- hybrid cooling 需要区分风侧证据和液路证据。
- CDU、coolant loop 和冷板状态不能由 LLM 编造。
