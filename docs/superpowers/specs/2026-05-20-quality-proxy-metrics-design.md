# 回答质量代理指标设计

## 背景

当前评测报告已经覆盖检索命中、上下文召回、关键词覆盖、工具选择、工具执行和证据覆盖。49 条 eval 样例也包含人工维护的 `expected_keywords`。不过现有指标仍偏向检索和词面覆盖，不能直接说明回答是否覆盖关键事实，也不能说明回答是否避免违反数据边界或工具边界。

## 目标

本轮新增本地、确定性、无模型依赖的回答质量代理指标：

- `answer_correctness_proxy`：衡量回答是否覆盖人工标注的关键事实。
- `faithfulness_proxy`：衡量回答是否基于证据，并避免命中人工标注的禁用表述。

这些指标是轻量代理，不宣称等价于完整人工 correctness / faithfulness 评审。

## 非目标

- 不接入 LLM judge。
- 不引入 cross-encoder、评测服务或新框架。
- 不要求一次性人工精标全部样例。
- 不改变 LLM / Agent 与控制工具的边界。
- 不让 LLM 直接生成或写回控制动作。

## 数据结构

在 `EvalRecord` 中新增两个可选字段：

- `must_include: list[str] = []`
- `must_not_include: list[str] = []`

含义：

- `must_include` 表示回答中应该出现的关键事实或术语。
- `must_not_include` 表示回答中不应出现的违反边界、幻觉或误导性表述。

现有 JSONL 记录不需要一次性全部补齐。没有标注的样例不参与对应代理指标计算。

## 指标定义

### answer_correctness_proxy

只评估包含 `must_include` 的记录。对每条记录计算回答命中 `must_include` 的比例，再对记录取平均。

该指标偏保守，适合衡量 extractive baseline 是否覆盖关键事实，但不能替代人工判断。

### faithfulness_proxy

只评估包含 `must_include` 或 `must_not_include` 的记录。每条记录从 1.0 开始：

1. 如果回答命中任何 `must_not_include`，该条得 0.0。
2. 如果记录需要文档引用或工具调用，但预测没有 citation 或 tool result，该条最高 0.5。
3. 如果存在 `must_include`，再乘以对应命中比例。

该指标用于捕捉两类风险：没有证据支撑，以及出现明确禁用表述。

## 实现范围

修改：

- `src/evaluation/dataset.py`
- `src/evaluation/metrics.py`
- `src/evaluation/runner.py`
- `src/evaluation/report.py`
- `data/eval/hvac_eval.jsonl`
- `tests/test_evaluation.py`
- `tests/test_experiment_report.py`
- `README.md`
- `docs/system_design.md`
- `docs/stage_1_handoff.md`

新增标注优先覆盖以下代表性样例：

- BEAR 数据边界类问题。
- Agent / policy 控制边界类问题。
- 检索证据不足或不能编造类问题。
- supply air reset / sensor drift / Delta-T 等新增领域样例。

## 测试策略

遵循 TDD：

1. 先写 dataset loader 测试，证明 `must_include` / `must_not_include` 字段可被读取。
2. 先写 metrics 测试，覆盖 correctness proxy 和 faithfulness proxy 的正常、缺证据、禁用表述三种情况。
3. 运行测试确认失败。
4. 实现最小代码。
5. 更新 runner/report，使两项指标进入整体和按任务类型表。
6. 运行 `python -m pytest` 和 `python scripts/run_eval.py`。

## 验收标准

- `EvalRecord` 能读取新字段且兼容旧记录。
- 新指标出现在 `baseline_comparison.json` 的 `summary` 和 `by_task_type` 中。
- `docs/experiment_report.md` 的全局表和按任务类型表包含新指标。
- 至少一批代表性 eval 样例含 `must_include` / `must_not_include`。
- `python -m pytest` 通过。
- `python scripts/run_eval.py` 成功。
- 所有文档继续明确 BEAR 是 HVAC 仿真/可控代理场景，不是生产遥测。
