# Supply Air Reset Risk Note

## 适用边界

本文是数据中心冷却类问题的领域说明，可用于解释 BEAR HVAC 仿真轨迹上的送风温度设定点问题。BEAR 在本项目中是物理仿真和可控代理场景，不是真实数据中心生产遥测。

## 关键判断

提高 supply air reset 或 supply temperature setpoint 可能降低冷却能耗，但风险不应只用平均温度判断。需要同时查看 zone temperature trend、comfort violation、hot spot evidence 和 recent control_action。如果温度上限附近已经出现持续波动，继续上调设定点可能扩大热风险。

节能建议不能由 LLM 直接生成控制动作。Agent 只能说明任务路由、证据和不确定性；recommended_action 必须来自 rule-based policy、MPC-like policy、DiffFNO / Guided-DiffFNO adapter 或 offline replay。

## 可检索要点

- supply air reset 需要结合 comfort violation。
- supply temperature setpoint 的节能收益必须和 hot spot risk 同时解释。
- policy 工具边界比自然语言建议更重要。
