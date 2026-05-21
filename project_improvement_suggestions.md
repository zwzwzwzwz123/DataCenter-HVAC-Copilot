# 🔍 DataCenter-HVAC Copilot 项目改进建议

> **目标**：将项目从当前「超 MVP 但未完全面试就绪」的状态提升到**简历级别的高质量可展示项目**。

---

## 一、项目现状评估

### ✅ 已完成（做得不错的部分）

| 维度 | 现状 | 评分 |
|---|---|---|
| **项目定位** | 明确的 B 路线：RAG + Tool Agent + Evaluation，不是普通 ChatPDF | ⭐⭐⭐⭐⭐ |
| **架构完整性** | 10 个子模块分层清晰（core/ingestion/retrieval/tools/policies/agent/evaluation/api） | ⭐⭐⭐⭐ |
| **数据边界意识** | BEAR 仿真 vs 真实生产数据的边界约束、Safety Audit | ⭐⭐⭐⭐⭐ |
| **评测体系** | 100 条 eval JSONL + 7 个 baseline + 9 项指标 + 按任务类型分组 | ⭐⭐⭐⭐ |
| **文档覆盖** | README、Project Spec、System Design、Experiment Report、Demo Walkthrough | ⭐⭐⭐⭐ |
| **前端 Demo** | Streamlit 深色控制台布局 + Copilot/评测摘要双 Tab | ⭐⭐⭐⭐ |
| **测试覆盖** | 23 个测试文件覆盖各模块 | ⭐⭐⭐⭐ |
| **策略集成** | DROPT/Guided-DiffFNO checkpoint adapter + rule-based + MPC-like + offline replay | ⭐⭐⭐⭐ |

### ⚠️ 需要改进的关键差距

| 维度 | 问题 | 影响 |
|---|---|---|
| **无 LangGraph** | Agent 仍用 deterministic router，没有真正的 LLM 工作流 | 面试时「Agent」能力难以讲深 |
| **无真实 Embedding** | 默认使用 deterministic hash embedding，非真正语义检索 | RAG 效果无法体现，面试被追问时暴露 |
| **无 Docker** | 没有容器化，启动步骤多 | 工程完整性不够，不便展示 |
| **无截图/录屏** | README 和文档全是文字，没有视觉展示 | 面试官/简历筛选时第一印象差 |
| **代码规模偏小** | 源码约 4000 行（不含测试），偏轻量 | 作为核心项目需要更多实质内容 |
| **Git 历史太少** | 仅 10 个 commit，且像是一次性生成 | 显得不像真实开发过程 |
| **人工标注为空** | human_review_annotations.jsonl 全部 pending | 评测可信度打折扣 |
| **bear_rollout.csv 为空** | bear_processed 目录只有 .gitkeep | 主要数据链路无法真正运行 |

---

## 二、改进建议（按优先级排序）

### 🔴 P0 — 必须做（直接影响面试通过率）

#### 1. 引入 LangGraph 替换 Deterministic Router

> [!IMPORTANT]
> 当前 `src/agent/orchestrator.py` 使用硬编码的 if-else 路由，在面试中被追问「你的 Agent 工作流是怎么实现的」时会非常被动。

**具体改动：**
- 用 LangGraph 重构 `src/agent/` 为有状态工作流图
- 节点：`intent_classifier` → `retriever` / `timeseries_tool` / `policy_tool` → `evidence_aggregator` → `answer_generator` → `safety_audit`
- 边：条件路由（基于 LLM 意图判断，而非正则匹配）
- 保留 deterministic router 作为 fallback baseline

**面试加分点：**
- 可以画出工作流图，讲 LangGraph state graph 的设计
- 可以对比 deterministic routing vs LLM-based routing 的准确率
- 展示 LangGraph trace/visualization

**预估工作量：** 2-3 天

---

#### 2. 启用真实语义检索（Sentence-Transformers + FAISS）

> [!IMPORTANT]
> 当前默认的 `rag_dense` 用的是 hash embedding（确定性占位），不是真正的语义向量检索。面试时 RAG 是高频追问点，必须有真实 embedding 效果数据。

**具体改动：**
- 在 `src/retrieval/dense_retriever.py` 中启用真实 `sentence-transformers` + `FAISS`
- 推荐使用 `BAAI/bge-small-zh-v1.5`（中文场景，本地推理，不需 API）
- 重新跑评测，对比 hash embedding vs 真实 embedding 的指标差异
- 更新 experiment_report.md 中的 dense retrieval 指标

**面试加分点：**
- 可以讲 embedding 模型选型（BGE vs E5 vs text-embedding）
- 可以讲 FAISS IndexFlatIP vs IVF 的 trade-off
- 有真实的 dense vs hybrid vs hybrid+rerank 指标对比

**预估工作量：** 1 天（代码已预留接口）

---

#### 3. 补充 Demo 截图和架构图

> [!IMPORTANT]
> README 完全没有截图。在 GitHub 页面或简历附件中，纯文字项目的第一印象远不如有截图的项目。

**具体改动：**
- 截取 Streamlit Copilot Tab 的完整界面（含深色控制台布局）
- 截取 3 个典型 walkthrough case 的运行结果
- 截取评测摘要 Tab 的指标展示
- 用 Mermaid 或 draw.io 画一张正式的系统架构图（替代 ASCII art）
- 将截图放到 `docs/images/` 并在 README 中引用

**面试加分点：**
- GitHub README 一打开就能看到项目全貌
- 面试时可以直接用截图引导讲解

**预估工作量：** 半天

---

#### 4. Docker 容器化

> [!WARNING]
> 当前没有 Dockerfile。容器化是工程能力的基本体现，简历上「Docker」也是标配关键词。

**具体改动：**
- 添加 `Dockerfile` 和 `docker-compose.yml`
- 一个容器跑 FastAPI，一个跑 Streamlit
- 添加 `Makefile` 或 `justfile` 统一启动命令
- README 添加「一键启动」说明

```dockerfile
# 示例
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install -e ".[dev,dense]"
COPY . .
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**预估工作量：** 半天

---

### 🟡 P1 — 强烈建议（显著提升项目深度）

#### 5. 完成人工评测标注

> [!WARNING]
> `human_review_annotations.jsonl` 中 24 条全是 `null`。实验报告显示 `pending_human_review`，在面试时被问到「你的评测数据靠谱吗」时会减分。

**具体改动：**
- 按 `docs/human_evaluation_guide.md` 标注 24 条样例的 correctness / faithfulness / safety boundary
- 重新运行 `python scripts/run_eval.py` 更新报告
- 在实验报告中展示 human calibration 结果 vs deterministic proxy 的一致性

**面试加分点：**
- 展示评测不只是代码自动化，还有人工校准
- 可以讲 proxy 指标与人工标注的相关性

**预估工作量：** 2-3 小时

---

#### 6. 生成真实 BEAR Rollout 数据

> [!IMPORTANT]
> `data/bear_processed/` 目录为空（只有 .gitkeep）。虽然有 BEAR sample CSV fallback，但主要数据链路缺失会让项目显得「没跑通」。

**具体改动：**
- 使用 `scripts/export_bear_data.py` 从 `BEAR/` 导出真实 rollout
- 生成 `data/bear_processed/bear_rollout.csv`
- 确保 demo 优先使用 processed CSV 而非 mock

**预估工作量：** 1-2 小时（取决于 BEAR 环境依赖安装）

---

#### 7. 增强 LLM 回答生成能力

**具体改动：**
- 当前 deterministic generator 是模板拼接，效果生硬
- 集成更好的 LLM 生成器：除 DeepSeek 外，支持 Qwen API / 本地 Ollama
- 在 `src/agent/answer_generator.py` 中实现 `OllamaAnswerGenerator`
- 对比不同生成器的回答质量

**面试加分点：**
- 可以讲多 LLM 后端适配的设计
- 可以讲 deterministic vs LLM 生成器的 trade-off

**预估工作量：** 1 天

---

#### 8. 增加 Query Rewrite / HyDE

**具体改动：**
- 在 `src/retrieval/` 中增加 query rewrite 模块
- 实现 HyDE（Hypothetical Document Embeddings）
- 对比 raw query vs rewrite vs HyDE 的检索效果
- 更新 baseline comparison

**面试加分点：**
- RAG 面试高频考点
- 有实验数据支撑

**预估工作量：** 1 天

---

### 🟢 P2 — 加分项（锦上添花）

#### 9. 代码质量提升

- **添加 type hints**：当前部分函数缺少类型标注
- **添加 logging**：用 `logging` 替代 `print`，便于调试和演示
- **添加 pre-commit hooks**：`ruff` / `black` / `mypy`
- **GitHub Actions CI**：自动跑 `pytest`，README 加个 CI badge

**预估工作量：** 半天

---

#### 10. 扩展评测集到 150-200 条

**具体改动：**
- 当前 100 条已经不错，但 Project Spec 建议 150-200 条
- 重点补充：异常诊断的复杂案例、多工具协作案例、边界测试案例
- 增加 cross-lingual 样例（中英混合提问）

**预估工作量：** 1 天

---

#### 11. 添加 API 文档（OpenAPI/Swagger）

- FastAPI 自带 Swagger UI，但需要完善 endpoint 的 description、example
- 在 README 中展示 Swagger 截图

**预估工作量：** 2 小时

---

#### 12. 完善 Git 历史

> [!TIP]
> 当前只有 10 个 commit，建议后续开发过程中保持正常的 commit 粒度和 message 规范，展示真实的开发迭代过程。

---

## 三、改进优先级总览

| 优先级 | 改进项 | 工作量 | 面试影响 |
|---|---|---|---|
| 🔴 P0 | LangGraph 替换 deterministic router | 2-3 天 | ⭐⭐⭐⭐⭐ |
| 🔴 P0 | 启用真实语义检索 | 1 天 | ⭐⭐⭐⭐⭐ |
| 🔴 P0 | 补充 Demo 截图和架构图 | 半天 | ⭐⭐⭐⭐⭐ |
| 🔴 P0 | Docker 容器化 | 半天 | ⭐⭐⭐⭐ |
| 🟡 P1 | 完成人工评测标注 | 2-3h | ⭐⭐⭐⭐ |
| 🟡 P1 | 生成真实 BEAR Rollout 数据 | 1-2h | ⭐⭐⭐⭐ |
| 🟡 P1 | 增强 LLM 回答生成 | 1 天 | ⭐⭐⭐ |
| 🟡 P1 | Query Rewrite / HyDE | 1 天 | ⭐⭐⭐⭐ |
| 🟢 P2 | 代码质量 (logging, CI, lint) | 半天 | ⭐⭐⭐ |
| 🟢 P2 | 扩展评测集 | 1 天 | ⭐⭐⭐ |
| 🟢 P2 | API 文档 | 2h | ⭐⭐ |
| 🟢 P2 | Git 历史规范 | 持续 | ⭐⭐ |

**总预估：如果全部完成 P0 + P1，约需 7-8 天全力投入。**

---

## 四、面试叙事优化建议

### 当前叙事问题
你的 career_plan 中将这个项目定位为「LLM Agent 调度 RL/扩散模型」的项目，但实际实现更偏向「RAG + Tool Agent + Evaluation」。两者需要对齐。

### 建议叙事

> 我构建了 DataCenter-HVAC Copilot，一个面向数据中心冷却优化场景的 RAG + Tool Agent 系统。系统基于 BEAR HVAC 物理仿真环境，支持四类任务：文档问答、时序查询、异常诊断和策略建议。
>
> **核心设计**：
> 1. **不是普通 ChatPDF**：Agent 会根据问题类型路由到不同工具链——文档检索、时序分析工具、或策略工具
> 2. **控制边界清晰**：LLM 只做证据整合和解释生成，控制动作必须来自 RL/扩散策略工具，有 Safety Audit 保障
> 3. **可复现评测**：100 条评测集 + 7 个 baseline 对比 + 9 项指标，不只是 demo
>
> **技术亮点**：
> - LangGraph 工作流编排（加入后）
> - Hybrid 检索 + Rerank + Evidence-Grounded Generation
> - DROPT/Guided-DiffFNO checkpoint 推理适配器（桥接论文方向）
> - DeepSeek 可选接入 + deterministic fallback
> - FastAPI + Streamlit 专业 Demo

### 面试必备的 5 个技术深度问题准备

1. **「你的检索用了什么方案？对比过哪些？」**
   → 展示 keyword vs dense vs hybrid vs hybrid+rerank 的实验数据

2. **「你的 Agent 是怎么做路由的？」**
   → 展示 LangGraph state graph（改进后）+ deterministic baseline 对比

3. **「为什么不让 LLM 直接控制 HVAC？」**
   → 展示 Safety Audit 机制 + policy adapter 架构

4. **「你的评测怎么保证可信？」**
   → 展示 deterministic metrics + human calibration + 可选 LLM judge 的三层评测

5. **「你的扩散模型/RL 在这里什么角色？」**
   → 展示 DROPT adapter 如何将论文模型封装为工具，被 Agent 调度

---

## 五、简历一句话表达（改进版）

**当前版本：**
> 构建 DataCenter-HVAC Copilot：基于 BEAR HVAC 仿真轨迹，设计 RAG + Tool Agent + Evaluation 系统...

**建议版本：**
> 构建 DataCenter-HVAC Copilot：基于 LangGraph 的 RAG + Tool Agent 系统，面向数据中心 HVAC 仿真场景，集成文档检索（Hybrid + Rerank）、时序分析工具、RL/扩散策略推理适配器和 Safety Audit；在 100 条评测集上对比 7 组 baseline，工具路由准确率 100%，证据覆盖率 91%，检索召回率从 keyword 的 55.4% 提升到 hybrid+rerank 的 60.0%。技术栈：LangGraph + FAISS + DeepSeek + FastAPI + Streamlit + Docker。

---

*最后更新：2026.05.21*
