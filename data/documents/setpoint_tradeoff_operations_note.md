# Setpoint Tradeoff Operations Note

Source type: project internal note.
Published at: 2026.
Category: control_policy.

提高 supply temperature setpoint 往往可以降低部分 cooling energy，但必须同时监控 comfort violation、hotspot risk、zone temperature upper bound 和控制动作变化。节能建议不能只由 LLM 直接生成控制动作，应由 rule-based policy、MPC-like adapter、DiffFNO / Guided-DiffFNO adapter 或 offline replay 工具给出。

如果真实 diffusion policy 后端尚未接入，系统应明确说明 limitation，并使用 configured fallback 或 offline replay。Agent 的职责是 route task、collect evidence 和 explain tool result，而不是训练模型或向 BEAR 环境直接写入动作。

本说明是项目内部测试文档，用于提供与 control_policy_note 和 sample_hvac_guidance 相近但更长的主题材料，便于后续扩展检索压力测试和评测集。
