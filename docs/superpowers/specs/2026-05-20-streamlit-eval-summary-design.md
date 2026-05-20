# Streamlit 评测摘要增强设计

## 背景

当前项目已经在后端评测报告中加入 `answer_correctness_proxy` 和 `faithfulness_proxy`，但 Streamlit 的评测摘要页仍按普通 metric 列表展示。这样 demo 页面无法突出检索、回答覆盖、工具调用和质量代理之间的差异，和最新评测体系脱节。

## 目标

增强 Streamlit 评测摘要 tab，让 demo 能更清楚展示当前 baseline 的关键结果：

- 将指标按 Retrieval、Answer、Tool、Quality Proxy 分组展示。
- 保留通用指标表，便于查看全部数值。
- 在 prediction preview 中增加证据相关字段，便于演示工具和引用覆盖。
- 明确说明 quality proxy 是本地确定性弱指标，不等价于人工评审或 LLM judge。

## 非目标

- 不改变 FastAPI `/eval/run` 响应结构。
- 不引入新前端框架。
- 不启动真实 LLM judge。
- 不将 BEAR 仿真轨迹描述成真实生产遥测。
- 不新增控制动作生成能力。

## 设计

在 `app/streamlit_app.py` 中新增纯函数 helper，便于单元测试：

- `group_eval_metrics(metrics: dict) -> dict[str, list[tuple[str, float]]]`
- `build_prediction_preview(predictions: list[dict]) -> list[dict]`

指标分组：

- Retrieval：`citation_hit_rate`、`context_recall`
- Answer：`expected_keyword_coverage`、`lexical_answer_coverage`
- Tool：`tool_selection_accuracy`、`tool_execution_success_rate`、`evidence_coverage`
- Quality Proxy：`answer_correctness_proxy`、`faithfulness_proxy`

prediction preview 增加：

- `has_citation`
- `has_tool_result`
- `answer_length`

UI 仍使用 Streamlit 原生组件：`st.metric`、`st.dataframe`、`st.caption`、`st.expander`。不做大规模视觉改造。

## 测试

新增或扩展 `tests/test_streamlit_client.py`：

- 测试指标分组能把新旧指标放入正确组。
- 测试 prediction preview 能计算 citation/tool/answer 字段。

不需要启动 Streamlit 服务；测试纯 helper 即可。

## 文档同步

更新：

- `README.md`
- `docs/system_design.md`
- `docs/stage_1_handoff.md`

说明 Streamlit 评测摘要已展示质量代理指标和证据相关 preview 字段。

## 验收标准

- `python -m pytest -q` 通过。
- `python scripts/run_eval.py` 成功。
- Streamlit 评测摘要 tab 能按组展示新增质量代理指标。
- 文档继续说明 BEAR 是 HVAC 仿真/可控代理场景，不是生产遥测。
