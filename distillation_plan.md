# 计划：Planner 决策蒸馏（SFT + DPO）

## 目标

把项目里 DeepSeek 充当 route planner / ReAct controller 的决策能力，蒸馏进一个可本地运行的小模型（Qwen2.5-1.5B-Instruct 为主，8B 为可选对比），并在**已有的 benchmark** 上对比：微调小模型 vs DeepSeek vs deterministic planner。

产出一条完整能力链：**数据构造 → SFT → DPO → 接回现有评测**。这一条链同时覆盖 agent 岗（受控 planner / tool use）和大模型训练岗（数据、SFT、DPO、评测）关注点，并把已有的 RL / 扩散 / ICCC 论文背景落到 LLM 上。

## 为什么这个方案资源可控

- **不做 PPO 式 RLHF**：无需 reward model + critic + 在线采样（四模型），只做 DPO——policy + 冻结 reference 两模型，偏好对离线构造，显存和工程量都低一个量级。
- **LoRA / QLoRA + 小模型**：1.5B QLoRA 在单张 12–16G 卡即可；8B QLoRA 需一张 24G 租用卡。
- **数据不用重新标**：复用项目已有的 planner trace 作为教师信号。

---

## 阶段 0 —— 定位与准备（0.5 天，无需 GPU）

**0.1 把 RL / 扩散 / 论文提到台前（最高 ROI，先做）**
- 简历与项目描述中显式写出：RL 训练经验、扩散模型、ICCC 论文，并连接到本项目的 policy 边界设计（DROPT / diffusion adapter）。
- 一句话叙事参考：「具备 RL 后训练底层能力，本项目中将其落地为受控 policy 边界；并通过 SFT+DPO 把云端 LLM 的 planner 决策蒸馏进本地小模型。」

**0.2 环境**
- 新建独立子目录 `distill/`，与主项目解耦，避免污染现有依赖。
- 训练依赖单独装：`transformers`、`trl`、`peft`、`datasets`、`accelerate`、`bitsandbytes`（QLoRA 用）。
- 建议用 TRL 的 `SFTTrainer` / `DPOTrainer`，成熟、面试可讲、recipe 公开。

---

## 阶段 1 —— 数据构造（3–5 天，无需 GPU，核心工序）

这是整个项目最能体现工程能力的部分，务必认真做。

**1.1 SFT 数据（planner 模仿）**
- 输入：`question`（+ 可选 `task_type`）。
- 输出：受控 plan（route + tool + time_window + reason），即项目 schema 里 `PlanStep` 序列的 JSON。
- 来源：
  - 现成 50 条 `real_eval.jsonl` 的 `question → route/tools` 只是种子，**量不够**。
  - 扩充：用项目已有的 DeepSeek planner，批量对更多 question 生成 plan（可从文档/时序/异常/策略四类各造一批问题，或对现有 question 做改写扩增），把 DeepSeek 的合法输出作为教师标签。
  - **只保留通过 `validate_plan_steps` 校验的样本**（复用 `src/agent/planner.py` 的校验函数）——这一步天然保证了 SFT 标签质量，也是很好的面试点。
- 目标规模：SFT 800–2000 条足够小模型学会 schema。
- 划分 train/val（如 90/10），val 用于早停和报告。

**1.2 DPO 偏好数据（利用你的 RL 视角）**
- 偏好对 `(chosen, rejected)` 构造思路（三选一或混合）：
  1. **guard 信号法（推荐，贴合项目）**：同一 question 生成多个候选 plan，用项目已有的本地 guard（`bounded_react.py` 的 duplicate guard / policy deadline / budget guard）判定。通过 guard 且合法 = chosen；被拦截或非法 = rejected。这直接把「本地 runtime 裁决」变成偏好信号，叙事极强。
  2. **教师 vs 学生法**：DeepSeek 的 plan = chosen，SFT 初版模型犯错的 plan = rejected。
  3. **规则退化法**：合法且 policy step 在最后 = chosen，顺序错误 / 超预算 = rejected。
- 目标规模：DPO 500–1500 对。
- 产出脚本：`distill/build_sft_data.py`、`distill/build_dpo_data.py`，均调用现有 planner / 校验函数，保证数据与线上行为一致。

**阶段 1 验收**：两份 jsonl 数据集 + 数据构造脚本 + 一页数据卡（规模、来源、构造方法、合法率）。此时代码已能全部本地跑通，不需要 GPU。

---

## 阶段 2 —— SFT（2–3 天，需 GPU）

**2.1 基座**：Qwen2.5-1.5B-Instruct（主）。QLoRA（4-bit）+ LoRA adapter。
**2.2 训练**：TRL `SFTTrainer`，chat template 对齐 Qwen 格式；1–3 epoch；跟踪 val loss。
**2.3 显存参考**：
- 1.5B QLoRA：单张 12–16G 卡可行。
- 8B QLoRA：需 24G，开 gradient checkpointing、短 max_len、batch=1 + 梯度累积。
**2.4 产出**：`distill/train_sft.py` + LoRA 权重 + 训练日志。

**阶段 2 验收**：SFT 后模型能对 val 问题输出**合法** plan（用 `validate_plan_steps` 统计合法率），合法率应显著高于未微调基座。

---

## 阶段 3 —— DPO（2–3 天，需 GPU）

**3.1** 以 SFT 后的模型为起点（policy），SFT 模型冻结副本为 reference。
**3.2** TRL `DPOTrainer` + 阶段 1.2 的偏好数据；关注 `beta`、chosen/rejected reward margin。
**3.3** 你有 RL 背景，这里能讲深：DPO 的隐式 reward、与 PPO 的等价性直觉、为什么不需要显式 reward model——**这些正是训练岗爱问的**。
**3.4 产出**：`distill/train_dpo.py` + DPO 权重 + reward margin 曲线。

**阶段 3 验收**：DPO 后在 val 上，plan 合法率 / guard 通过率 / 与教师 plan 一致率相比 SFT 有提升（哪怕小幅），且能解释提升来源。

---

## 阶段 4 —— 接回现有评测（2–3 天）

这一步把整个工作变成闭环，是 README/面试的落点。

**4.1** 把微调模型包装成一个 `RoutePlanner` 实现（复用 `src/agent/planner.py` 的 `RoutePlanner` Protocol），使其能像 DeepSeek/deterministic 一样插进现有 pipeline。
**4.2** 在已有 benchmark 上跑三方对比：
- **planned_step_accuracy / order_accuracy / policy_final_step_rate**（compound task 评测已有）
- plan 合法率、guard 通过率、平均延迟、（可选）本地推理显存/速度
- 对象：`deterministic` vs `DeepSeek(教师)` vs `SFT` vs `SFT+DPO`（+ 可选 8B）
**4.3** 产出一张对比表，核心结论期望是：**1.5B 蒸馏模型在 planner 指标上接近 DeepSeek，但可本地运行、延迟/成本大幅下降**。

**阶段 4 验收**：`docs/distillation_report.md` 一份，含方法、数据卡、三方对比表、结论与局限。

---

## 阶段 5 —— 呈现（1 天）

- 主 README 增一节「Planner Distillation (SFT + DPO)」，指向 `docs/distillation_report.md`。
- 简历项目描述更新：数据构造 + SFT + DPO + 评测闭环 + 本地小模型替代云端 LLM。

---

## 时间与硬件小结

| 阶段 | 时长 | GPU |
| --- | --- | --- |
| 0 定位准备 | 0.5 天 | 否 |
| 1 数据构造 | 3–5 天 | 否 |
| 2 SFT | 2–3 天 | 是 |
| 3 DPO | 2–3 天 | 是 |
| 4 接回评测 | 2–3 天 | 推理即可 |
| 5 呈现 | 1 天 | 否 |

总计约 2–3 周实工时，落在你 1–3 个月的窗口内且有余量做 8B 对比。

**硬件建议**：1.5B 为主，租 AutoDL/Colab 一张 16–24G 卡，SFT+DPO 全程预算约几十元。8B 作为可选对比再单独租 24G 卡跑一轮。

---

## 关键原则（贯穿全程）

1. **数据构造复用现有 planner 与校验函数**，保证蒸馏数据与线上行为一致——这是本方案区别于「孤立练手项目」的核心。
2. **偏好信号来自项目已有的本地 guard**，把「runtime 裁决」直接变成 DPO 训练信号，叙事闭环。
3. **每阶段有可验证的验收标准**，不靠感觉推进。
4. **先做阶段 0 和阶段 1**：不需要 GPU 就能锁定方案地基，确认数据够用再租卡，避免浪费预算。
