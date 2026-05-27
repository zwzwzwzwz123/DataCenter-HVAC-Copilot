本文件回答：从 70-80% 项目补到更完整简历作品，当前漏洞具体在哪里。

# 完整度审计

### A. 工程基础设施

**当前评分**：6/10

**已经做到的**：
- CI 存在，Ubuntu + Python 3.12 + `pip install -e ".[dev]"` + ruff + pytest，见 `.github/workflows/ci.yml:20`。
- `pyproject.toml` 配置 pytest 和 ruff，见 `pyproject.toml:31`、`pyproject.toml:36`。
- 测试规模不小：`pytest --collect-only -q` 收集 259 tests；测试代码 6708 非空非注释行。按文件名粗略映射，`app` 关联 576 行测试，`scripts` 关联 45 行测试。
- Dockerfile 和 compose 存在，API/Streamlit 双服务配置清楚，见 `Dockerfile:1`、`docker-compose.yml:1`。
- 配置管理有 `.env.example`，且 `.env` loader 不覆盖已有环境变量，见 `.env.example:17`、`src/core/env.py:7`。

**明显缺失的**：
- 没有 coverage 工具或覆盖率报告；只能说“测试多”，不能说覆盖率百分比。
- 没有 mypy/pyright；类型标注有但未静态验证。
- Docker 无法在本机验证：`docker` 命令不存在，因此“一键跑通”未被确认。
- 依赖声明不完整：`app/api_client.py:5` import `httpx`，`src/policies/dropt_adapter.py:9` import `torch`，但 `pyproject.toml:7` 依赖列表没有 `httpx`/`torch`。
- 无结构化 logging；全仓 `rg logger|logging|getLogger` 没有命中，脚本只用 `print`（如 `scripts/run_eval.py:208`）。

**质量不够的（有但不达标）**：
- `ruff` 当前失败 1 处：`B904` at `src/knowledge/service.py:349`，CI 会因此失败。
- 完整 `python -m pytest -q` 当前 259 tests 中 1 fail：`tests/test_query_rewrite.py:63`，rewrite RAG 未检索到预期上下文。
- 错误处理抽样：较好的是 memory storage/retrieval 分离上报（`src/api/app.py:141`、`src/api/app.py:159`）、knowledge ingest 失败记录 failed document（`src/knowledge/service.py:126`）、FAISS rebuild 失败恢复 backup（`src/knowledge/indexer.py:86`）；较弱的是 DeepSeek/Ollama 捕获所有 Exception 后静默 fallback（`src/agent/deepseek_generator.py:64`、`src/agent/ollama_generator.py:56`），没有错误原因返回或日志。

### B. 文档与展示

**当前评分**：7/10

**已经做到的**：
- README 覆盖系统边界、架构、启动、评测、LLM 配置、knowledge/memory/DROPT 等，关键信息密度高（例如 `README.md:46`、`README.md:421`）。
- 有 demo walkthrough，5-8 分钟路径写得清楚，见 `docs/demo_walkthrough.md:1`、`docs/demo_walkthrough.md:83`。
- 有实验报告 `docs/experiment_report.md:1` 和 human eval guide `docs/human_evaluation_guide.md`。
- 核心文件有一些 docstring，例如 `AgentTaskExecutor`（`src/agent/executor.py:27`）、`DeterministicAnswerGenerator`（`src/agent/answer_generator.py:32`）、`ContextManager`（`src/memory/context_manager.py:15`）。

**明显缺失的**：
- README 没有 hero 图、架构图截图、Streamlit demo 截图或 GIF 素材。
- API 文档主要依赖 FastAPI 自动 docs；没有单独的 OpenAPI 示例文档或 curl 请求/响应集。
- 文档与现实有不同步：README 多处写 100 条 eval（`README.md:25`、`README.md:459`），实验报告实际 108 条（`docs/experiment_report.md:5`）。

**质量不够的（有但不达标）**：
- README 很长，给面试官读会累；外部 reviewer 更需要一页式架构图和 demo 截图。
- docstring 覆盖不均：`src/api/app.py` 大量闭包 endpoint 缺 docstring；`src/knowledge/service.py` 复杂但注释少。

### C. 评测体系

**当前评分**：6.5/10

**已经做到的**：
- 主 eval 当前 108 条：document_qa 40、timeseries_query 20、anomaly_diagnosis 20、policy_recommendation 28；见 `data/eval/hvac_eval.jsonl` 实际统计与 `docs/experiment_report.md:5`。
- baseline summary 当前 15 个：`llm_only`、keyword、dense、hybrid、rewrite、HyDE、rag_tool_agent、langgraph、react 等，见 `docs/experiment_report.md:18` 到 `docs/experiment_report.md:32`。
- 指标真实实现：`citation_hit_rate` 检查 required docs subset（`src/evaluation/metrics.py:10`），`tool_selection_accuracy` 检查 required tools（`src/evaluation/metrics.py:46`），`faithfulness_proxy` 检查 must_not_include 和 evidence presence（`src/evaluation/metrics.py:158`）。
- 真实 dense baseline 用 `sentence-transformers` + FAISS，见 `docs/experiment_report.md:10`。
- safety adversarial 有 29 条并生成 hit rate，见 `docs/experiment_report.md:107`。

**明显缺失的**：
- 人工标注为空：24 条 `human_review_annotations.jsonl` 的 `correctness_score`、`faithfulness_score`、`safety_boundary` 全是 null，见 `data/eval/human_review_annotations.jsonl:1`；报告显示 labeled_count=0（`docs/experiment_report.md:99`）。
- eval 样本真实性偏 demo/AI 生成风险：数据来自项目内 JSONL 和 notes；没有真实运维问答或外部标注来源说明。此项基于经验推测，证据是 `data/eval/hvac_eval.jsonl` 使用项目内维护的 `gold_answer` / `expected_keywords` 字段。
- `scripts/run_eval.py` 可跑但当前完整测试失败，削弱“评测脚本稳定”印象。

**质量不够的（有但不达标）**：
- README 里的旧数字与报告数字不一致。
- `answer_correctness_proxy`/`faithfulness_proxy` 是关键词代理，不能替代人工正确性；代码也只是字符串覆盖率，见 `src/evaluation/metrics.py:144`。
- Safety Audit 对抗集 hit rate 0.586，translation 0.000（`docs/experiment_report.md:111`、`docs/experiment_report.md:120`），需要降级叙述。

### D. 可演示性

**当前评分**：7/10

**已经做到的**：
- Streamlit 是完整页面，不只是 toy UI；有 Copilot、evaluation、knowledge base 等入口，main 在 `app/streamlit_app.py:1241`。
- API client 封装 ask/eval/knowledge upload/status，见 `app/api_client.py:12`、`app/api_client.py:40`、`app/api_client.py:86`。
- walkthrough 有 5-8 分钟流程，见 `docs/demo_walkthrough.md:36`、`docs/demo_walkthrough.md:52`、`docs/demo_walkthrough.md:67`。
- Streamlit 默认 LangGraph，workflow 切换配置在 `app/streamlit_app.py:37`。

**明显缺失的**：
- 没有 demo 视频/GIF/截图素材。
- 没有“5 分钟一键演示脚本”，仍需手动启动 API/Streamlit、选择 case。
- 本机未验证 Streamlit 视觉表现；本次未启动浏览器截图，故 UI 视觉质量只能基于代码和 README 推断。

**质量不够的（有但不达标）**：
- Streamlit 文件 1428 行，CSS/HTML/helper 混在一起；展示效果可能好，但维护性一般。
- `httpx` 未在依赖中声明，新的环境下 Streamlit client 可能直接 import 失败（`app/api_client.py:5`）。

### E. 数据与可信度

**当前评分**：7/10

**已经做到的**：
- README 明确不能把 BEAR 说成真实生产遥测，见 `README.md:14`。
- demo 数据加载优先级清晰：processed CSV -> BEAR sample CSV -> mock，见 `src/api/demo_factory.py:99`。
- 当前 processed rollout 文件存在，README 写 2016 行 BEAR rollout，见 `README.md:592`；loader 会把 data_source 写入响应（`src/api/demo_factory.py:104`）。
- BEAR 导出脚本是真接口：构建 env 后调用 `export_bear_rollout`，见 `scripts/export_bear_data.py:30`、`src/ingestion/bear_adapter.py:63`。

**明显缺失的**：
- 没有独立数据卡/dataset card，说明哪些文档是人工写、哪些 AI 生成、哪些来自 BEAR。
- 没有自动验证 `scripts/export_bear_data.py` 在外部 BEAR repo 上可跑；本仓 vendored BEAR 目录未必等同 README 的外部 clone 流程。
- 没有对 `data/documents` 的来源、许可证、生成方式逐条说明。

**质量不够的（有但不达标）**：
- README 的“数据中心冷却优化”叙述容易让 reviewer 误以为是真实数据中心数据；虽然边界写了，但简历措辞必须持续强调 BEAR 仿真。
- Safety Audit 的生产遥测防误述在中文 paraphrase 好，英文/间接表达弱。

## 漏洞优先级排序

按“简历加分 / 工作量”排序：

1. S：修复 `ruff` B904 和当前 1 个 pytest 失败；这是 CI 可信度底线。
2. S：补齐 `pyproject.toml` 依赖中的 `httpx`、`torch`，并说明 torch 是否 optional。
3. S：统一 README/报告里的 eval 数字：100 vs 108。
4. S：给 README 加 1 张架构图和 2-3 张 Streamlit 截图/GIF。
5. S：把 Safety Audit 文案降级，明确 hit rate 与已知漏检类别。
6. S：填 8-12 条 human review annotation，哪怕小样本，也比 24 条全空强。
7. M：增加 `pytest-cov` 或 coverage report，至少输出模块级覆盖率。
8. M：拆出“5 分钟 demo script/checklist”，包含启动、health check、样例问题、预期输出。
9. M：为 `/ask`、knowledge upload、eval run 写 3 个 curl 请求/响应示例文档。
10. M：加基本 logging，至少覆盖 LLM fallback、knowledge refresh、memory indexing、eval run。
11. M：补一份 data card，解释 BEAR rollout、demo docs、eval JSONL 的来源和边界。
12. L：做更可靠的 safety audit 或 LLM judge calibration；对简历有用，但不必追工业级。
