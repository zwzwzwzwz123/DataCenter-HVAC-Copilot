# DataCenter-HVAC Copilot · 改进进展评估 & 简历含金量评估 & 新一轮建议

> **评估日期**：2026-05-22
> **对照文件**：[project_improvement_suggestions.md](project_improvement_suggestions.md)（2026-05-21 制定）
> **职业目标参考**：[career_plan.md](career_plan.md) — 大模型应用算法岗（RAG + Agent 方向）
> **验证方法**：以代码与产物为准（直接读 `src/`、`tests/`、`data/eval/`、配置文件、git log），不被进度文档误导

---

## 一、上轮 12 项建议逐条核验

> 评分 0-10，以**当前代码与产物**为唯一依据。文档里写"已完成"但代码里看不到的，按未完成处理。

| # | 建议 | 优先级 | 现状判断 | 评分 |
|---|---|---|---|---|
| 1 | LangGraph 替换 deterministic router | 🔴 P0 | ✅ **真实落地** | 8/10 |
| 2 | 真实语义检索（ST + FAISS） | 🔴 P0 | ✅ **真实落地** | 9/10 |
| 3 | Demo 截图 + 架构图 | 🔴 P0 | ❌ **完全未做** | 1/10 |
| 4 | Docker 容器化 | 🔴 P0 | ⚠️ 基础完成 | 6/10 |
| 5 | 完成人工评测标注 | 🟡 P1 | ❌ **24/24 全 null** | 0/10 |
| 6 | 真实 BEAR rollout 数据 | 🟡 P1 | ✅ 已生成（有空列） | 7/10 |
| 7 | 增强 LLM 回答生成 | 🟡 P1 | ⚠️ 接口完成，无质量对比 | 6/10 |
| 8 | Query Rewrite / HyDE | 🟡 P1 | ✅ deterministic 完成 | 7/10 |
| 9 | 代码质量（logging / lint / CI） | 🟢 P2 | ❌ 几乎全部缺失 | 3/10 |
| 10 | 评测集扩展到 150-200 条 | 🟢 P2 | ❌ 仍 100 条 | 0/10 |
| 11 | API 文档 / Swagger | 🟢 P2 | ❌ 无 description | 2/10 |
| 12 | 完善 Git 历史 | 🟢 P2 | ❌ 仅 12 个 commit | 2/10 |

**统计：12 项里 4 项真实完成（1/2/6/8），2 项部分完成（4/7），6 项明显欠账（3/5/9/10/11/12）。整体进度约 51/120 = 42%。**

> [!IMPORTANT]
> 与 [project_progress_evaluation.md](project_progress_evaluation.md) 略有差异：进度文档把 LLM 回答增强和 HyDE 写得偏乐观。本评估以"是否有真实对比实验/真实 LLM 调用结果落库"为标准，更严格。

---

### 1. ✅ LangGraph — 8/10

**核验代码**：[src/agent/langgraph_workflow.py](src/agent/langgraph_workflow.py) 共 203 行。

- 用 `langgraph.graph.StateGraph` 构建 7 节点图：`intent_classifier` → 条件边 → `retrieval / timeseries_tool / anomaly_tool / policy_tool` → `evidence_aggregator` → `answer_audit` → END
- 定义了 `WorkflowState(TypedDict)`，每个节点都写入 `workflow_trace`
- intent 节点支持注入 `RuleBasedIntentClassifier` / `LLMIntentClassifier` (DeepSeek) / `OllamaIntentClassifier`，见 [src/agent/intent_classifier.py](src/agent/intent_classifier.py)
- 与 `BaselineOrchestrator` 通过共享的 `AgentTaskExecutor`（[src/agent/executor.py](src/agent/executor.py)）解耦工具执行，不再调用私有方法
- API `/ask` 支持 `workflow_engine=langgraph|deterministic`，Streamlit 有 trace 面板
- 评测里独立出 `langgraph_tool_agent` baseline，并新增 [scripts/run_intent_eval.py](scripts/run_intent_eval.py) 输出 [intent_routing_comparison.json](data/eval/intent_routing_comparison.json)（rule_based accuracy 0.640）

**扣分点**：rule-based 在 100 条上 accuracy 仅 0.64，document_qa 类有 18/40 被错路由（confusion matrix 显示常被错判到 timeseries）。`intent_routing_comparison.json` 里目前**只跑了 rule_based**，没有 DeepSeek/Ollama 的真实对比数据落库——所以"LangGraph + LLM intent"的差异化优势还是叙事，不是数据。

---

### 2. ✅ 真实语义检索 — 9/10

**核验代码**：[src/retrieval/embeddings.py](src/retrieval/embeddings.py)、[src/retrieval/faiss_retriever.py](src/retrieval/faiss_retriever.py)

- `SentenceTransformerEmbeddingProvider` 真实加载模型，默认 `all-MiniLM-L6-v2`
- `FaissDenseRetriever` 用 `faiss.IndexFlatIP`
- `pyproject.toml` 有 `[dense]` extras（`faiss-cpu>=1.8`、`sentence-transformers>=3.0`）
- `run_eval.py` 支持 `--dense-provider sentence-transformers --dense-backend faiss --dense-model BAAI/bge-small-zh-v1.5`
- [docs/experiment_report.md](docs/experiment_report.md) 顶部"运行配置"明确写了 `dense_provider: sentence-transformers / dense_backend: faiss / dense_model: BAAI/bge-small-zh-v1.5`，并落了真实指标

**这是这一轮最有价值的成果**：`rag_dense` citation_hit_rate 0.692 / context_recall 0.692 / answer_correctness_proxy 0.654，**全面优于 keyword (0.554) 和 hybrid+rerank (0.600)**。这是面试时可以直接拿出来的硬数据。

**扣分点**：dense/faiss 是 optional extras，[Dockerfile](Dockerfile) 只装 `[dev]`，容器跑评测会回退到 hash embedding。

---

### 3. ❌ Demo 截图 + 架构图 — 1/10

- `docs/images/` 不存在
- README **零图片**（grep `![` 无结果）
- README 系统架构仍是 ASCII art（只在 LangGraph 段插了一张小 Mermaid）

**这是上轮成本最低、回报最高的事，本轮零行动。** 简历筛选阶段，纯文字 GitHub 项目第一印象就输一截，这个必须立即补。

---

### 4. ⚠️ Docker — 6/10

- [Dockerfile](Dockerfile) 16 行，[docker-compose.yml](docker-compose.yml) 双服务（api + streamlit）
- 有 `.dockerignore`

**问题**：
- 装的是 `[dev]` 不是 `[dense]`，镜像里跑 dense 评测会失败
- 没有 healthcheck / non-root user / restart policy
- 没有数据 volume 挂载
- 没有 Makefile / justfile 一键命令

---

### 5. ❌ 人工评测标注 — 0/10

`grep -c '"correctness_score": null' data/eval/human_review_annotations.jsonl` = **24/24 全 null**，[experiment_report.md](docs/experiment_report.md) 仍是 `pending_human_review`。

> [!WARNING]
> 这是最容易被面试官一击致命的弱点：你写了 `human_evaluation_guide.md`、留了 schema、起了"human calibration"这个名字，**结果一条都没标**。被追问"deterministic proxy 和人工到底一致不一致"时，唯一诚实回答是"还没标"。这比根本不做这个机制还尴尬。

**只需 2-3 小时**，自己按 1-5 分把 24 条标完，是这一轮最高 ROI 的动作之一。

---

### 6. ✅ 真实 BEAR Rollout — 7/10

[data/bear_processed/bear_rollout.csv](data/bear_processed/bear_rollout.csv) 2017 行 + 18 列 + 多 zone。

**扣分点**：`humidity / it_load / cooling_power / fan_power / chiller_power / hvac_power / pue` 多列为空。这本身没错（BEAR 场景就不提供），但 **README / experiment_report 里没有显式声明哪些列是 BEAR 原生、哪些是衍生、哪些场景缺失**。面试被问"PUE 数据来源"时容易失分。

---

### 7. ⚠️ LLM Answer Generator — 6/10

- ✅ [src/agent/deepseek_generator.py](src/agent/deepseek_generator.py) + [src/agent/ollama_generator.py](src/agent/ollama_generator.py)
- ✅ `build_answer_generator_from_env()` 支持 `deterministic / deepseek / ollama` 三态
- ❌ `baseline_comparison.json` 里**没有 DeepSeek 答和 Ollama 答的 side-by-side 指标**——只有 deterministic generator 的结果
- ❌ 没有任何"LLM-as-Judge 真实运行结果"落库（`DeterministicKeywordJudge` 是确定性 proxy，不是 LLM judge）

**所以这条是"接口完成 + 零真实对比实验"**。简历声称"多 LLM 后端 + 质量对比"会被一问就破。

---

### 8. ✅ Query Rewrite / HyDE — 7/10

- [src/retrieval/query_rewrite.py](src/retrieval/query_rewrite.py) 实现了 `RuleBasedHVACQueryRewriter` / `TemplateHyDEGenerator` / `RewriteRAGPipeline` / `HyDERAGPipeline`
- 评测里跑出了真实数据：

| Baseline | citation_hit_rate | context_recall | expected_keyword_coverage |
|---|---:|---:|---:|
| rag_keyword | 0.554 | 0.554 | 0.372 |
| rag_dense | **0.692** | **0.692** | 0.528 |
| rag_rewrite | 0.646 | 0.646 | **0.584** |
| rag_hyde | 0.246 | 0.246 | 0.182 |
| rag_hyde_rerank | 0.338 | 0.338 | 0.195 |

**亮点**：`rag_hyde` **比 keyword 还低**——这反而是面试的好素材，可以讲"deterministic template HyDE 在小语料 + 中文领域会引入 query drift"，证明你做过实验、有判断力，而不是无脑套技巧。

**扣分点**：还是 deterministic 模板，没接 DeepSeek/Ollama 真实生成 hypothetical document 的对比。

---

### 9. ❌ 代码质量 — 3/10

- ✅ Type hints / Protocol / TypedDict 用得不错
- ❌ `grep -r "import logging" src/` = **0 hit**——零日志
- ❌ 没有 `.github/workflows/` 目录
- ❌ 没有 `Makefile` / `justfile`
- ❌ 没有 `.pre-commit-config.yaml`
- ❌ `pyproject.toml` 里 dev 依赖只有 `pytest`，没有 ruff / black / mypy
- ❌ `logs/` 目录是空的（虽然存在）

> [!WARNING]
> "工程能力" 是 [career_plan.md](career_plan.md) 明确说的最大短板。这一栏 3/10 直接打脸。CI / lint / logging 是简历"工程"两个字的最低门槛。

---

### 10-12. ❌ 评测集 / API 文档 / Git 历史

- 评测仍 100 条（[hvac_eval.jsonl](data/eval/hvac_eval.jsonl)）
- FastAPI [src/api/app.py](src/api/app.py) endpoint 无 `summary` / `description`，[src/api/schemas.py](src/api/schemas.py) 也没 `Field(description=...)`
- `git log --oneline` = **12 个 commit**，比上一轮多了 2 个，且最近 2 个都是上传文档；本轮 P0 的代码改动**没有以单独 commit 形式留下来**——这从面试官角度看，是"项目又是一次性丢上来的"

---

## 二、整体完成度评估

### 项目当前真实状态

| 维度 | 评分 | 评语 |
|---|---|---|
| **项目定位 & 差异化** | ⭐⭐⭐⭐⭐ | RAG + Tool Agent + RL/Diffusion 策略适配器 + Safety Audit，赛道很稀缺 |
| **架构完整性** | ⭐⭐⭐⭐ | 10 个模块、47 个 .py、4740 行源码、27 个测试文件、3960 行测试 |
| **核心功能落地** | ⭐⭐⭐⭐ | LangGraph + 真实 dense + Query Rewrite + 多 LLM 后端 都有真实代码和指标 |
| **评测可信度** | ⭐⭐⭐ | 11 组 baseline 真实跑通，但人工标注 0/24，intent eval 只跑了 rule_based |
| **可视化展示** | ⭐⭐ | 没有任何截图，README 第一眼是 ASCII art |
| **工程基线** | ⭐⭐ | 无 CI / lint / logging，dev 依赖只 pytest |
| **数据真实性** | ⭐⭐⭐⭐ | BEAR rollout CSV 落库，但部分列空 + 来源未声明 |
| **简历可讲性** | ⭐⭐⭐⭐ | 已具备"5 分钟深度版"的素材，但叙事还有过度声明的风险 |

**整体完成度：约 65-70%。** 离"面试就绪、随便挑 5 分钟讲都不慌"还差一段——主要是**展示层 + 工程层 + 评测可信度补丁**这三个不在核心代码里的事。

### 与 career_plan 时间线对照

[career_plan.md](career_plan.md) 写的 2026.05 里程碑是"DataCenter-HVAC Copilot 超 MVP 完成"，2026.06 是"项目升级：LangGraph + 真实 Embedding + Docker"。

**事实：**
- ✅ LangGraph 已落地（提前完成 6 月里程碑）
- ✅ 真实 Embedding 已落地（提前完成）
- ✅ Docker 基础已落地（提前完成）
- 📅 距离 2027.01 投递日还有 8 个月——**核心代码层已基本完成，剩下的都是"打磨 + 周边"**

> [!TIP]
> 你跑得比原计划快。这意味着接下来 3 个月（5-7 月）应该把重心从"加新功能"切到"打磨 + 占领差异化叙事"，而不是继续堆模块。新功能堆得再多，没截图、没 CI、没人工标注、Git 历史一次性，简历筛选阶段就过不了。

---

## 三、放在简历上的含金量评估

### 当前可信叙事下的简历强度

按 [career_plan.md](career_plan.md) 第三梯队（央国企 / 电网 AI 实验室）→ 第二梯队（独角兽 / 金融科技）→ 第一梯队（字节豆包 / 阿里通义）的顺序看：

| 目标段位 | 含金量 | 评语 |
|---|---|---|
| **国家电网 / 南网 AI 实验室、运营商研究院** | 🟢 **强** | 你的领域 + 学历几乎是 perfect match，这个项目即使现在的状态也够拿面试 |
| **银行科技子公司、央企智能化平台** | 🟢 **强** | RAG + Agent + 工程闭环 + 论文背景，简历筛选过线没问题 |
| **DeepSeek / 智谱 / MiniMax / 月暗** | 🟡 **中等偏上** | 项目能讲，但 AI 公司面试官对 LangGraph / RAG 细节会问得很深，目前评测可信度短板会被抓住 |
| **字节豆包 / 阿里通义 / 腾讯混元** | 🟡 **中等** | 项目方向对，但**没有大厂实习**才是最大瓶颈，项目本身只是"够进面试"，不是"决胜面试" |
| **互联网大厂 SP / SSP（40w+ TC）** | 🔴 **不够** | 需要至少一段大厂实习 + 项目 + 面试表现都顶才有戏，目前两条腿一长一短 |

### 简历一句话（按当前真实状态、不夸大）

```
DataCenter-HVAC Copilot — 基于 LangGraph StateGraph 的 RAG + Tool Agent 系统（个人项目）
- 面向 BEAR HVAC 物理仿真场景，路由文档问答 / 时序查询 / 异常诊断 / 策略建议四类任务
- 11 组检索 baseline 对比：rag_dense (BGE-small-zh + FAISS) citation 0.692 / context_recall 0.692，
  rag_rewrite expected_keyword_coverage 0.584；rag_hyde 出现 query drift（0.246），
  作为 deterministic baseline 的反例
- LangGraph 7 节点工作流 + workflow_trace；intent classifier 可切 rule-based / DeepSeek / Ollama
- 集成 DROPT / Guided-DiffFNO checkpoint 推理适配器，LLM 仅做证据整合 + Safety Audit 边界
- FastAPI + Streamlit 双服务 + Docker Compose，pytest 27 个测试文件
- 技术栈：Python 3.12 / LangGraph / FAISS / sentence-transformers (BGE-small-zh) /
  DeepSeek API / Ollama / FastAPI / Streamlit / Docker / pytest
```

**为什么不能说更多**：
- ❌ 不能写"100 条评测集 + 人工校准"——人工标注 0/24
- ❌ 不能写"多 LLM 后端质量对比"——没有真实 LLM 答案级 side-by-side 指标
- ❌ 不能写"LangGraph + LLM intent routing 提升 X%"——`intent_routing_comparison.json` 只跑了 rule_based
- ❌ 不能写"Docker 一键启动包含真实语义检索"——dense 是 optional extra

### 含金量提升的杠杆（按 ROI 排）

1. **截图 + 架构图（2 小时，回报最大）** — 简历附 GitHub link 时，README 第一眼有图就赢一半
2. **24 条人工标注（3 小时）** — 直接把"不能说人工校准"变成"能说，且和 proxy 一致性 X%"
3. **真实跑一次 DeepSeek / Ollama intent 对比并落库（1 小时）** — 把"LLM intent 节点"从叙事变数据
4. **CI badge + ruff + 一行 logging 配置（2 小时）** — 工程印象分立刻拉一档
5. **简历叙事替换：把"提升 X%"换成"识别出 HyDE drift + 用 dense 解决"** — 体现实验判断力，比堆数字更高级

---

## 四、新一轮改进建议（按 ROI 排序，2026-05-22 起）

> 时间约束：career_plan 写的 2027-01 投递。从今天起还有 ~8 个月。
> 策略：**5-6 月做"打磨周"+ 占领叙事，7 月起把主力切回刷题 + 论文 + 实习投递。**

### 🔴 P0 · 本周内做完（合计 ~8 小时）

#### P0-1. 截图 + Mermaid 架构图（2 小时）

```
docs/images/
├── readme_hero.png            # README 顶部 hero 图，Streamlit 全貌
├── streamlit_copilot_tab.png  # 含工作流 trace 面板
├── streamlit_walkthrough_doc_qa.png
├── streamlit_walkthrough_anomaly.png
├── streamlit_eval_summary.png
├── swagger_docs.png           # /docs OpenAPI
└── architecture.png           # 用 Mermaid 渲染或导出
```

README 头部加：

```markdown
![Streamlit Demo](docs/images/readme_hero.png)

[![CI](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)](...)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
```

把 README 里 ASCII art 系统架构换成扩展版 Mermaid（包含 Safety Audit / DROPT adapter / 多 LLM 后端三个亮点）。

#### P0-2. 完成 24 条人工评测标注（3 小时）

按 `docs/human_evaluation_guide.md` 给 24 条样本填：
- `correctness_score`: 1-5
- `faithfulness_score`: 1-5
- `safety_boundary`: pass / fail
- `reviewer_notes`: 1 句中文

跑完后重新执行：
```bash
python scripts/run_eval.py
```
让 [experiment_report.md](docs/experiment_report.md) 的 `Human Calibration` 段从 `pending_human_review` 变成 `labeled`。

**关键产出**：实验报告里多一行"人工标注 mean_correctness X，与 deterministic proxy Pearson r = Y"。简历 / 面试瞬间多一张牌。

#### P0-3. 跑一次 DeepSeek intent eval 并落库（1 小时）

```bash
# 假设 .env 里有 DEEPSEEK_API_KEY
python scripts/run_intent_eval.py --providers rule_based deepseek
```

把 `intent_routing_comparison.json` 里 `deepseek` 那一栏填上真实 accuracy / fallback_rate / confusion_matrix。
**没有 DeepSeek 配额时，至少装个本地 Ollama 跑 qwen2.5:7b 也行**（你有 RTX 3070 8GB，qwen2.5:3b 或 7b-q4 都能跑）。

把对比写进 [experiment_report.md](docs/experiment_report.md) 末尾的"Intent Routing"小节。

#### P0-4. CI + Lint + Logging 工程基线（2 小时）

**(a) `pyproject.toml` 补依赖**：
```toml
dev = [
  "pytest>=7.4",
  "ruff>=0.8",
  "mypy>=1.14",
  "pre-commit>=4.0",
]
```

**(b) 新增 `.github/workflows/ci.yml`**：跑 `ruff check` + `pytest tests/` + 上传 baseline_comparison artifact。

**(c) `.pre-commit-config.yaml`**：ruff fix + ruff format。

**(d) 新增 `src/core/logging_config.py`**：统一 `logging.getLogger`，至少在 [orchestrator.py](src/agent/orchestrator.py) / [langgraph_workflow.py](src/agent/langgraph_workflow.py) / [retriever.py](src/retrieval/retriever.py) 三个核心模块加 `logger.info` 关键节点。

**(e) 新增 `Makefile`**：
```makefile
.PHONY: dev test lint eval docker-up docker-down
test:      ; pytest -q
lint:      ; ruff check src/ tests/
eval:      ; python scripts/run_eval.py
docker-up: ; docker compose up --build -d
```

**(f) README 顶部加 CI badge。**

> [!IMPORTANT]
> 这一组做完，简历"工程能力"那一栏从 3/10 直接拉到 6/10。这是最便宜的提升。

---

### 🟡 P1 · 本月内做完（合计 ~2-3 天）

#### P1-1. 拆分 Git 历史，让本轮改进留下足迹（持续）

> [!WARNING]
> 当前 12 个 commit 几乎全是粗粒度的 feat: / docs:。
> **不要 squash 历史**（破坏性 + 无收益），但**接下来每一个改进都按功能粒度提交**：
>
> - `feat(viz): add streamlit screenshots and architecture diagram`
> - `feat(eval): annotate 24 human review samples`
> - `feat(eval): add deepseek intent routing comparison run`
> - `chore(ci): add github actions and ruff pre-commit`
> - `feat(logging): introduce structured logging in agent core`
>
> 目标：到 2026-12 底前累积 30-50 个 commit，让 git history graph 看起来像"持续迭代"，而不是"两次大爆炸"。

#### P1-2. Dockerfile 升级到生产可用（1 小时）

- 改装 `[dense]` 而非 `[dev]`（或多阶段构建：dev 阶段跑测试，runtime 阶段只装 dense + runtime 依赖）
- 加 non-root user
- 加 healthcheck (`HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1`)
- docker-compose 加 restart policy + data volume
- 写一个 `make docker-up` 一键体验

#### P1-3. API endpoint description + Swagger 截图（1 小时）

- [src/api/app.py](src/api/app.py) 每个 endpoint 加 `summary` / `description` / `responses` 例子
- [src/api/schemas.py](src/api/schemas.py) Pydantic 字段加 `Field(description=...)`
- 截一张 `/docs` 的 Swagger 图放进 `docs/images/swagger_docs.png`

#### P1-4. README 数据来源声明 + BEAR 列说明（30 分钟）

`README.md` "数据边界"小节明确：

```
| 字段 | 来源 | 状态 |
|---|---|---|
| zone_temperature, outdoor_temp, ... | BEAR rollout 原生 | ✅ 已填充 |
| reward, control_action | BEAR 派生 | ✅ 已填充 |
| pue, humidity, it_load, chiller_power | BEAR 不提供 | ⚠️ 当前为空，需通过外部模型估算 |
```

避免面试被追"PUE 数据从哪来的"时被一击即破。

#### P1-5. 评测集补到 130 条（半天）

不需要硬怼到 200，但 100 条覆盖度真的不够。重点补：

- 跨意图边界样本（让 router 挨打 5-10 条）：例如"最近一小时温度趋势是否提示需要调整 setpoint？"——同时涉及 timeseries + policy
- 中英混合 5-10 条
- 安全边界 prompt（例如直接要求 LLM 写控制动作）5-10 条 → Safety Audit 必须命中

补完后 `rag_tool_agent` 的指标会更稳，confusion matrix 会更有深度可讲。

#### P1-6. 跑一次 DeepSeek answer generator 的端到端 baseline（1 天）

新增一个 `rag_tool_agent_deepseek` baseline，让 [experiment_report.md](docs/experiment_report.md) 里能直接对比：
- deterministic generator vs DeepSeek generator 在同一份证据下的 `expected_keyword_coverage` 和 `lexical_answer_coverage`
- 平均回答长度 / 平均 latency

**这是把"多 LLM 后端"从"接口完成"升级到"有数据"的关键**。

---

### 🟢 P2 · 7-9 月可做（不阻塞投递）

| # | 改进 | 工作量 | 说明 |
|---|---|---|---|
| P2-1 | LangGraph 加一个 `replan` 节点 | 2 天 | 当 retriever 召回低 / Safety Audit 警告时回到 intent 重新分类，让 graph 不只是"线性图" |
| P2-2 | 把 baseline 跑成参数化 sweep | 1 天 | top_k / chunk_size / rerank_k 的小型消融实验，写进 experiment_report 附录 |
| P2-3 | 写 1-2 篇技术博客 | 持续 | career_plan 里建议过；选题"为什么 deterministic HyDE 在 HVAC 中文场景反而掉点"，是别人没做过的角度 |
| P2-4 | 把 `dropt_adapter` 真实接论文 checkpoint 跑通 | 视论文进度 | 当前是 stub，能跑通就把"集成自研论文模型"从声明变事实 |
| P2-5 | 自部署 demo | 半天 | Streamlit Community Cloud 或 HuggingFace Spaces，简历贴 live demo URL，比 GitHub 截图再加一档 |

---

## 五、优先级 / 时间表对齐 career_plan

```
2026.05 (本周)        P0-1~P0-4：截图 + 人工标注 + intent 真跑 + CI/Lint/Logging   [8h]
2026.05 (本月)        P1-1~P1-3：Git 拆分 + Dockerfile 升级 + Swagger              [3-4h]
2026.06               P1-4~P1-6：数据声明 + 评测扩到 130 条 + DeepSeek 答案对比    [2-3 天]
2026.07               写 2 篇技术博客（HyDE drift + Safety Audit 设计）             [4-6h]
2026.08               简历定稿 v1，开始模拟面试                                     [持续]
2026.09-10            八股文集中突破（career_plan 已规划）                          [持续]
2026.11               全力刷题 + 模拟面试                                           [持续]
2026.12               最终检查、补漏                                                [持续]
2027.01               🚀 投递实习                                                  [持续]
```

> [!TIP]
> 6 月底前所有"项目相关"的事必须收尾。7 月开始**项目只做小修补**，主力切到刷题 + 八股 + 论文 + 投递准备。再继续在项目上加功能是反 ROI 的——招聘官打开你的 GitHub 看 30 秒就走，他们不会在意你又加了第 12 个 baseline。

---

## 六、面试叙事修订（基于当前真实状态）

### ✅ 现在可以自信讲

- "用 LangGraph StateGraph 编排 7 节点工作流，intent 节点支持 DeepSeek / Ollama 切换，默认保留 rule-based 作为可复现 baseline"
- "真实 BGE-small-zh-v1.5 + FAISS dense retrieval，对比 11 组 baseline，dense citation_hit_rate 0.692 优于 keyword 0.554"
- "deterministic template HyDE 在中文 HVAC 小语料上反而掉点（0.246 vs keyword 0.554），所以用 query rewrite (0.646) 而不是 HyDE 作为生产路径"
- "Tool Agent 工具选择和执行成功率 100%，证据覆盖率 91%"
- "Safety Audit 用确定性规则而非 LLM，因为安全边界不能依赖概率模型"

### ⚠️ 现在需要小心讲（P0 做完前不要主动提）

- 不要主动说"人工评测校准过"——24/24 还是 null
- 不要主动说"多 LLM 后端质量对比"——只有接口，没有 side-by-side
- 不要主动说"LangGraph + LLM intent 大幅提升路由准确率"——只有 rule_based 数据
- 不要主动说"一键 Docker 启动含真实语义检索"——dense 在镜像里不会装

### 🔥 高级反问（P0 做完后可以主动展示）

- "你看 rag_hyde 比 keyword 还低，这其实是好结果——说明 deterministic template 在中文小语料上会引入 query drift。这个反例让我决定生产路径用 query rewrite 不用 HyDE。"
- "rule-based intent 在 document_qa 上的混淆主要走向 timeseries（18 条里有 7 条），这反映了关键词分类的天花板，所以我把 intent 节点设计成可注入接口，方便切到 DeepSeek 后实测准确率提升。"

---

## 七、一句话总结

> 你这个月把上轮 P0 的硬骨头（LangGraph + 真实 dense + Docker）啃下来了，**核心代码层基本到位**；剩下的差距全在"展示层 / 评测可信度补丁 / 工程基线"——加起来 8-12 小时的事，但不做就会让前面那些硬功夫被"看起来不专业"拖住。本周把 P0 这四件做完，含金量从"能进央国企面试"提到"能进 AI 独角兽面试"。

---

*最后更新：2026-05-22*
