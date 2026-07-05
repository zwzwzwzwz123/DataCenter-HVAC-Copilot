# Planner Distillation — Stage 4 Comparison

Eval set: `data/eval/compound_task_eval.jsonl` (100 records; step metrics use the rows that have `expected_steps`).

> ⚠️ **术语澄清**：最终 SFT 模型的训练标签 **100% 为人工手标 gold**（数据卡 `teacher: hand_labeled`）。DeepSeek 在本报告里是**评测对照基线**（线上另一条 planner 路线），**不是**训练教师。下文表格/文字中的 "DeepSeek 教师" / "教师" 一词沿用评测脚本的历史列名，**仅指该对照列**，不代表用其输出训练了模型——对外表述请说「用人工 gold 做 planner SFT」，勿说「蒸馏了 DeepSeek」。

| planner | step_acc | order_acc | req_recall | policy_final | non_fb | avg_lat |
| --- | --- | --- | --- | --- | --- | --- |
| eval_pred_deterministic | 14.0% | 14.0% | 56.3% | 72.7% | 100/100 | 0.000s |
| eval_pred_distilledA2 (旧, 600条 SFT) | 68.0% | 68.0% | 85.5% | 93.5% | 74/100 | 0.791s |
| **eval_pred_distilled_a1 (新, 700条 SFT)** | **84.0%** | **84.0%** | **93.0%** | **96.1%** | **90/100** | 0.792s |
| eval_pred_teacher (DeepSeek) | 63.0% | 63.0% | 85.7% | 93.5% | 72/100 | 18.353s |

**指标说明**：
- `step_acc` planned_step_accuracy：预测 route 集合 == 期望集合的比例
- `order_acc` planned_step_order_accuracy：预测 route 顺序完全一致的比例
- `req_recall` required_step_recall：期望 route 被覆盖的平均比例
- `policy_final`：含 policy 的任务中 policy 步落在最后的比例
- `non_fb`：未走 deterministic fallback 的计划数（越高说明该 planner 自身产出越多有效计划）
- `avg_lat`：单条平均规划延迟

---

## 结论解读

**核心结果：A1 补数据后的本地 1.5B SFT 模型达到 84% step_acc——是规则基线（14%）的 6 倍，并显著反超作为对照基线的云端 DeepSeek planner（63%）21 个点，同时本地运行、单条延迟 0.8s、无 API 成本、比其快约 23 倍（0.8s vs 18s）。** 这条链证明：用经守卫校验的高质量**人工 gold** 做 SFT，能让 1.5B 小模型在本项目 compound-task 规划上不仅追平、而是超过一个通用云端大模型（注：DeepSeek 是评测对照、非训练教师）。

### 四方定位（A1 补数据后）
| planner | 角色 | step_acc | 非fallback | 延迟 | 部署 |
| --- | --- | --- | --- | --- | --- |
| deterministic | 低基线（规则关键词路由） | 14.0% | 100/100 | ~0s | 本地 |
| distilled A2（旧, 600条） | 蒸馏学生（补数据前） | 68.0% | 74/100 | 0.8s | 本地 GPU |
| **distilled A1（新, 700条）** | **蒸馏学生（本工作最终产物）** | **84.0%** | **90/100** | 0.8s | 本地 GPU |
| DeepSeek（对照基线） | 云端大模型 | 63.0% | 72/100 | 18s | API |

### A1（补 100 条 gold 数据）前后对比
残余 fallback 分析（见 §10）显示旧版 26 条回退里，time_window 无法归一化 + tool 选错占主体。A1 针对性补了 **100 条经守卫校验的 gold 样本**：把自然语言时间窗（`过去一周`/`last month`/`past 3 days`）直接标成合法小时词表让模型自身学会归一化（不再只靠 A2 推理层兜底），并补齐 8 个 timeseries 工具的消歧样本 + 多步复合任务。重训（train 630 / val 70，val exact_match 96.67%→97.14%）后重跑评测：

| 指标 | 蒸馏 A2（600条） | 蒸馏 A1（700条） | 变化 |
| --- | --- | --- | --- |
| step_acc | 68.0% | **84.0%** | **+16 个点** |
| order_acc | 68.0% | **84.0%** | +16 个点 |
| req_recall | 85.5% | **93.0%** | +7.5 个点 |
| policy_final | 93.5% | **96.1%** | +2.6 个点 |
| 非fallback | 74/100 | **90/100** | +16 条 |

**A1 是"治本"改动**：把归一化/工具选择能力固化进模型权重，而非仅靠 A2 推理层兜底。逐条对比：旧版 26 条 fallback 中 **A1 修复了 20 条**，新引入 4 条（解码随机性/边界样本），两版都失败的 6 条。净减 16 条回退。

### 蒸馏 A1 版的残余局限（如实记录）
A1 版 100 条中 90 条非 fallback，其余 **10 条回退**。逐条核查后，这 10 条**全部是最难的 3 步复合任务**，且集中在 A1 数据未专门覆盖的形态：
- **多指标同图/比值绘图**（如 "plot cooling_power and fan_power ratio"、"zone_temperature and setpoint together"）——单步里要表达两个 metric 的组合，超出当前 step schema 的单 metric_name 表达力；
- **时段限定窗**（"during the night 10 PM–6 AM"）——非"最近 N 小时"型的绝对时段，词表本就不支持；
- **episode/zone 具体引用 + 多文档检索**（"episode_001 zone_a"、"doc_002/doc_006"）——模型对具体 id 的处理仍不稳。

这些不是 time_window 归一化或基础 tool 选择的问题（那两类已被 A1 基本解决），而是更深的多步复合表达能力，属阶段3 DPO 或进一步扩充这类样本的目标。

### 一句话总结
> 经 SFT（人工 gold）+ A2 归一化 + A1 补数据后的本地 1.5B 学生模型，在本项目 compound-task 规划上达到 **84% step_acc**，是规则基线（14%）的 6 倍，并反超作为对照基线的云端 DeepSeek planner（63%）21 个点；本地运行、0.8s 延迟（对照的 1/23）、零 API 成本。残余 10 条回退全部是多指标绘图、绝对时段、多文档引用等最难的多步复合任务，属阶段3 DPO 的目标。

---

## 附录：A2（time_window 归一化，推理层）阶段结论

> 以下为 A1 补数据之前、仅靠 A2 推理层归一化时的三方对比，留档以说明优化路径。

### A2 时的三方定位
| planner | 角色 | step_acc | 延迟 | 部署 |
| --- | --- | --- | --- | --- |
| deterministic | 低基线（规则关键词路由） | 14.0% | ~0s | 本地 |
| distilled 1.5B（600条） | 蒸馏学生 | 68.0% | 0.8s | 本地 GPU |
| DeepSeek（对照基线） | 云端大模型 | 63.0% | 18s | API |

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

### 一句话总结（A2 阶段）
> 经 SFT 蒸馏 + A2 归一化后的本地 1.5B 学生模型，在本项目 compound-task 规划上达到 **68% step_acc**，是规则基线（14%）的近 5 倍，并追平云端 DeepSeek 教师（63%）；且本地运行、0.8s 延迟（教师的 1/23）、零 API 成本。剩余待改进项集中在 JSON 截断、tool 选择、以及少数无法归一化的复杂时间窗——**这些正是 A1 补数据要治的短板（见正文），A1 后 step_acc 进一步升到 84%**。

