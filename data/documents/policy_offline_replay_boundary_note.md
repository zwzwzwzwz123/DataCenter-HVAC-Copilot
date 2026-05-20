# Policy Offline Replay Boundary Note

## 适用边界

本文说明 policy 工具、offline replay 和 diffusion adapter 的边界。Agent 不直接生成控制动作，也不向 BEAR 环境写回动作。

## 策略边界

当真实 DiffFNO / Guided-DiffFNO 后端未接入时，系统应使用 rule-based fallback、MPC-like placeholder 或 offline replay。offline replay 只能读取已保存实验结果，不能让 LLM 伪造 recommended_action、estimated_energy 或 comfort risk。

如果用户要求具体控制数值，回答应说明该数值必须来自 policy tool output。Agent 可以解释 policy_name、notes、estimated risk 和 fallback 原因，但不能把自然语言建议当成控制器。

## 可检索要点

- offline replay 读取已保存策略结果。
- DiffFNO adapter 未接入时必须显式说明 limitation。
- LLM 不直接生成或写回控制动作。
