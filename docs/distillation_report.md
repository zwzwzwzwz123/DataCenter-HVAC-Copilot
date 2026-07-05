# Planner Distillation — Stage 4 Comparison

Eval set: `data/eval/compound_task_eval.jsonl` (100 records; step metrics use the 3-way rows below over records that have `expected_steps`).

| planner | step_acc | order_acc | req_recall | policy_final | non_fb | avg_lat |
| --- | --- | --- | --- | --- | --- | --- |
| eval_pred_deterministic | 14.0% | 14.0% | 56.3% | 72.7% | 100/100 | 0.000s |
| eval_pred_distilledA2 | 68.0% | 68.0% | 85.5% | 93.5% | 74/100 | 0.791s |
| eval_pred_teacher | 63.0% | 63.0% | 85.7% | 93.5% | 72/100 | 18.353s |

**指标说明**：
- `step_acc` planned_step_accuracy：预测 route 集合 == 期望集合的比例
- `order_acc` planned_step_order_accuracy：预测 route 顺序完全一致的比例
- `req_recall` required_step_recall：期望 route 被覆盖的平均比例
- `policy_final`：含 policy 的任务中 policy 步落在最后的比例
- `non_fb`：未走 deterministic fallback 的计划数（越高说明该 planner 自身产出越多有效计划）
- `avg_lat`：单条平均规划延迟

---

## 结论解读

**核心结果：蒸馏后的本地 1.5B 模型在全部四项规划指标上都显著优于规则基线（step_acc 68% vs 14%，约 4.9 倍），并已追平云端 DeepSeek 教师（68% vs 63%），同时本地运行、单条延迟 0.8s、无 API 成本、比教师快约 23 倍（0.8s vs 18s）。** 这证明 SFT 让小模型学到了规则关键词匹配学不到的多步复合任务拆解能力。

### 三方定位（A2 归一化后）
| planner | 角色 | step_acc | 延迟 | 部署 |
| --- | --- | --- | --- | --- |
| deterministic | 低基线（规则关键词路由） | 14.0% | ~0s | 本地 |
| **distilled 1.5B** | **蒸馏学生（本工作产物）** | **68.0%** | 0.8s | 本地 GPU |
| DeepSeek 教师 | 云端大模型 | 63.0% | 18s | API |

### A2（time_window 归一化）前后对比
评测第一版暴露出一个共性问题：模型（教师与蒸馏都）倾向输出自然语言时间窗（`"past 7 days"`、`"last month"`、`"7d"`），而线上校验守卫 `validate_plan_steps` 只接受受限词表（`full_demo_range` / `last_24_hours` / `latest` 等），导致大量计划被拒后回退到 deterministic。

A2 在解析层加了 `_normalize_time_window`（`src/agent/planner.py`），把这些自然语言表达映射回规范词表（天/周/月折算成小时，与 gold 训练格式一致），教师与蒸馏两条线共用同一解析函数，因此同时受益：

| 指标 | 蒸馏 A2 前 | 蒸馏 A2 后 | 教师 A2 前 | 教师 A2 后 |
| --- | --- | --- | --- | --- |
| step_acc | 47.0% | **68.0%** | 21.0% | **63.0%** |
| order_acc | 47.0% | **68.0%** | 21.0% | **63.0%** |
| req_recall | 74.7% | **85.5%** | 60.0% | **85.7%** |
| policy_final | 84.4% | **93.5%** | 74.0% | **93.5%** |
| 非fallback | 51/100 | **74/100** | 13/100 | **72/100** |
| 因 time_window 回退 | 34 | **8** | 83 | **25** |

**A2 是纯推理层改动，不需要重新训练**。蒸馏 step_acc +21 个点、教师 +42 个点。教师提升幅度更大，是因为它此前几乎全被 time_window 问题卡死（13/100）；蒸馏本身已学到部分规范格式（SFT 用经守卫校验的 gold 训练），所以基数更高、提升相对温和但仍显著。

> 注：教师 A2 前后是两次独立 API 调用（`deepseek-v4-flash`，temperature=0 但长文本仍有微小随机性，且两次超时的条数不同），故 13→72 的提升中，主体是 A2 归一化的功劳，但不是 100% 纯归一化贡献。蒸馏两版则是同一 adapter、同一确定性解码，差异可完全归因于 A2。

### 蒸馏模型自身的残余局限（如实记录）
蒸馏 A2 版 100 条中 74 条非 fallback，其余 26 条回退，原因：
- **10 条 JSON 格式错**（`max_new_tokens` 截断长计划或输出非法 JSON）——可调大 token 上限或优化停止条件缓解；
- **8 条 tool 非法**（选了该 route 不允许的工具）；
- **8 条 time_window 仍无法归一化**（字典形式 `{'start_time':...}`、或错把 episode_id 填进时间窗等——这类是真正无法可靠映射的输入，A2 有意不强行猜，交给守卫拒绝）。

这三类是下一步（A1 补 gold 数据 / 阶段3 DPO）最该针对的短板。

### 一句话总结
> 经 SFT 蒸馏 + A2 归一化后的本地 1.5B 学生模型，在本项目 compound-task 规划上达到 **68% step_acc**，是规则基线（14%）的近 5 倍，并追平云端 DeepSeek 教师（63%）；且本地运行、0.8s 延迟（教师的 1/23）、零 API 成本。剩余待改进项集中在 JSON 截断、tool 选择、以及少数无法归一化的复杂时间窗，属数据/解码层面可继续优化的问题。

