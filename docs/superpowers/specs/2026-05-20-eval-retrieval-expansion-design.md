# 评测集与检索重排扩展设计

## 背景

当前 DataCenter-HVAC Copilot 已具备 BEAR 仿真轨迹标准化、时序工具、policy adapter、轻量 RAG、deterministic router、FastAPI、Streamlit demo 和 37 条 eval baseline。最新瓶颈主要在文档问答的 citation/context 与回答覆盖率；`rag_hybrid_rerank` 与 `rag_hybrid` 指标持平，说明重排接口已有，但现有样例不足以验证重排策略价值。

## 目标

本轮推进目标是扩展更接近领域真实表达的文档与评测样例，并加入一个轻量、确定性、无新依赖的 metadata-aware reranker 改进，使 baseline 报告能更清楚展示检索与重排差异。

## 非目标

- 不引入 cross-encoder、LLM judge 或在线模型依赖。
- 不把 BEAR 仿真轨迹表述为真实数据中心生产遥测。
- 不让 LLM 直接生成或写回控制动作。
- 不重构 Agent 为复杂 LangGraph 工作流。
- 不大改 Streamlit UI。

## 设计范围

### 文档扩展

在 `data/documents/` 新增 3-4 篇 UTF-8 Markdown 文档，主题覆盖：

- 冷通道封闭与气流短路。
- 送风温度设定点与节能/风险权衡。
- 机柜 Delta-T、回风温度和热交换效率。
- 传感器漂移、告警误判或仿真评测边界。

每篇文档都保留可追溯标题和主题边界，避免把仿真数据误写为生产遥测。

### Eval 扩展

在 `data/eval/hvac_eval.jsonl` 新增约 12 条样例，优先覆盖 `document_qa`，并少量覆盖 `anomaly_diagnosis` 与 `policy_recommendation`。每条样例包含：

- `question`
- `task_type`
- `gold_answer`
- `required_tools`
- `required_documents`
- `expected_output_format`
- `expected_keywords`

新增样例要刻意包含近义表达和相似主题干扰，让 keyword、hybrid 和 rerank 的差异可被报告捕捉。

### Reranker 改进

增强 `RerankingRetriever` 的候选打分，除正文覆盖率、短语命中和 base score 外，加入 citation metadata 的轻量词面匹配：

- `title`
- `section`
- `source_id`

实现保持确定性、纯 Python、无新依赖。默认权重保守，避免 metadata 命中完全压过正文证据。

### 测试策略

遵循 TDD：

1. 先新增 reranker 单元测试，构造长噪声正文与短目标文档，证明 metadata-aware rerank 能提升目标文档排序。
2. 运行该测试并确认失败原因是当前未使用 metadata。
3. 实现最小 reranker 改动。
4. 运行相关测试和全量测试。

文档和 eval 扩展不需要复杂生产代码，但必须通过现有 dataset loader、retrieval pipeline 和 baseline runner 测试。

### 报告与文档同步

如改动 eval、baseline 或报告数据，运行：

```bash
python -m pytest
python scripts/run_eval.py
```

自动更新：

- `data/eval/baseline_predictions.jsonl`
- `data/eval/baseline_comparison.json`
- `docs/experiment_report.md`

同步手动更新：

- `README.md`
- `docs/system_design.md`
- `docs/stage_1_handoff.md`

## 验收标准

- 新增文档能被现有 UTF-8 loader 自动加载。
- eval 记录总数从 37 条增加到约 49 条，且所有新增样例包含 `expected_keywords`。
- `RerankingRetriever` 有明确的 metadata-aware 单元测试覆盖。
- `python -m pytest` 通过。
- `python scripts/run_eval.py` 成功生成最新报告。
- 文档继续明确 BEAR 是 HVAC 仿真/可控代理场景，不是生产遥测。
