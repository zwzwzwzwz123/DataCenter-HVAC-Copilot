# Planner Distillation — Stage 4 Comparison

Eval set: `data/eval/compound_task_eval.jsonl` (100 records; step metrics use the 3-way rows below over records that have `expected_steps`).

| planner | step_acc | order_acc | req_recall | policy_final | non_fb | avg_lat |
| --- | --- | --- | --- | --- | --- | --- |
| eval_pred_deterministic | 14.0% | 14.0% | 56.3% | 72.7% | 100/100 | 0.000s |
| eval_pred_distilled | 47.0% | 47.0% | 74.7% | 84.4% | 51/100 | 0.812s |
| eval_pred_teacher | 21.0% | 21.0% | 60.0% | 74.0% | 13/100 | 25.207s |

**指标说明**：
- `step_acc` planned_step_accuracy：预测 route 集合 == 期望集合的比例
- `order_acc` planned_step_order_accuracy：预测 route 顺序完全一致的比例
- `req_recall` required_step_recall：期望 route 被覆盖的平均比例
- `policy_final`：含 policy 的任务中 policy 步落在最后的比例
- `non_fb`：未走 deterministic fallback 的计划数（越高说明该 planner 自身产出越多有效计划）
- `avg_lat`：单条平均规划延迟

---

## 结论解读

**核心结果：蒸馏后的本地 1.5B 模型在全部四项规划指标上都显著优于规则基线，step/order 精确率是 deterministic 的 3.4 倍（47% vs 14%）。** 这证明 SFT 让小模型学到了规则关键词匹配学不到的多步复合任务拆解能力，且完全本地运行、单条延迟 0.8s，无 API 成本。

### 三方定位
| planner | 角色 | step_acc | 延迟 | 部署 |
| --- | --- | --- | --- | --- |
| deterministic | 低基线（规则关键词路由） | 14.0% | ~0s | 本地 |
| **distilled 1.5B** | **蒸馏学生（本工作产物）** | **47.0%** | 0.8s | 本地 GPU |
| DeepSeek 教师 | 云端大模型 | 21.0% | 25s | API |

### 为什么教师的 step_acc 看起来只有 21%？——不是模型能力弱，是 schema 合规问题
教师那一列的低分**几乎全部来自输出被线上校验守卫（`validate_plan_steps`）拒绝后回退**，而非规划思路错误。100 条中只有 13 条通过校验，回退原因分布：

- **83 条 `unsupported time_window`**：DeepSeek 倾向输出自然语言时间窗（如 `"past 7 days"`、`"last month"`、`"last 2 hours"`），而守卫只接受受限词表（`full_demo_range` / `last_24_hours` / `latest` 等，见 `TIME_WINDOW_PATTERN`）。教师的系统提示**未约束该词表**，故大量被拒。
- 2 条超时、1 条 tool 非法、1 条 JSON 解析错。

**这恰恰是蒸馏的价值所在**：SFT 用 600 条经守卫校验的 gold 数据训练，让学生模型学会了直接产出**符合线上 schema** 的计划——这是教师（未针对本项目 schema 微调）做不到的。换句话说，蒸馏不只是"压缩"教师，而是把"符合本系统约束"这一隐性要求固化进了模型。

### 蒸馏模型自身的局限（如实记录）
蒸馏模型 100 条中 51 条非 fallback，其余 49 条回退，原因：
- **34 条 `bad_time_window`**：与教师同类问题——训练数据里多步任务的时间窗表达仍不够覆盖，模型对部分自然语言时间窗未归一化到合法词表。**这是下一轮扩充 gold 数据 / 阶段 3 DPO 最该针对的短板。**
- 5 条 tool 非法、约 10 条 JSON 截断或格式错（`max_new_tokens` 截断长计划所致，可调大或优化停止条件）。

即便算上这 49 条回退（回退时退化为 deterministic 的 14% 水平），整体仍达到 47%，说明非 fallback 子集上的规划质量相当高。若能把 time_window 归一化问题解决（数据层面即可，无需换模型），非 fallback 比例和总分还有明显上升空间。

### 一句话总结
> 蒸馏后的 1.5B 学生模型，在本项目 compound-task 规划上以 **47% step_acc** 大幅超越规则基线（14%），本地运行、0.8s 延迟、零 API 成本；相比之下云端教师因输出不符合线上 schema 词表，实际可用率反而低。主要待改进项是 time_window 归一化，属数据层面可解决的问题。

> 备注：teacher 的 25s 平均延迟含少量 API 超时；其低通过率主要是 schema 词表不匹配而非规划能力，评估时应结合上文 fallback 分析而非仅看表格数字。

