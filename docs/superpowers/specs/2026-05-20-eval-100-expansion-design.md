# Eval 100 扩展设计

## 背景

当前项目已有 49 条 eval 样例、13 篇开发文档、baseline comparison、quality proxy 指标和 Streamlit 评测摘要展示。下一步要提高项目可信度和最终完成度，需要把评测集扩展到 100 条左右，并补充更多领域风格文档和质量代理标注。

## 目标

- 将 `data/eval/hvac_eval.jsonl` 从 49 条扩展到 100 条。
- 让 100 条样例全部包含 `expected_keywords`。
- 至少 40 条样例包含 `must_include` 或 `must_not_include`。
- 新增 6 篇 UTF-8 Markdown 领域文档，用于更真实的近义和干扰检索。
- 继续保持 BEAR 是 HVAC 仿真/可控代理场景，不能表述为真实生产遥测。

## 样例分布

目标分布：

- `document_qa`：40 条
- `timeseries_query`：20 条
- `anomaly_diagnosis`：20 条
- `policy_recommendation`：20 条

新增样例优先覆盖：

- 液冷/风冷混合过渡的边界说明。
- economizer / free cooling 风险。
- 冷却冗余、维护窗口和告警优先级。
- 传感器缺失、漂移和多区域交叉验证。
- policy 工具、offline replay 和 diffusion adapter 边界。
- 时序工具的周期对比、趋势图、能耗拆分和异常诊断。

## 文档扩展

新增 6 篇 `data/documents/*.md`：

- `economizer_free_cooling_note.md`
- `redundancy_maintenance_alarm_note.md`
- `liquid_air_hybrid_cooling_note.md`
- `sensor_missing_data_quality_note.md`
- `policy_offline_replay_boundary_note.md`
- `timeseries_tool_workflow_note.md`

文档全部作为领域说明或评测说明，不宣称来自真实数据中心生产遥测。

## 测试与报告

遵循 TDD：

1. 先把 eval 数量测试改为 100，任务类型分布测试改为目标分布，并要求质量代理标注不少于 40 条。
2. 运行测试确认失败。
3. 新增文档和 eval 样例。
4. 运行 `python -m pytest -q`。
5. 运行 `python scripts/run_eval.py` 生成最新报告。
6. 更新 README、system_design、stage_1_handoff。

## 非目标

- 不接 LLM judge。
- 不引入新框架。
- 不改变 API 契约。
- 不让 LLM 直接生成或写回控制动作。
- 不部署真实 DiffFNO / Guided-DiffFNO 推理。

## 验收标准

- `data/eval/hvac_eval.jsonl` 恰好 100 条。
- 任务类型分布为 40 / 20 / 20 / 20。
- 100 条全部包含 `expected_keywords`。
- 至少 40 条包含质量代理标注。
- 新文档能被 loader 自动加载。
- `python -m pytest -q` 通过。
- `python scripts/run_eval.py` 成功。
