# DataCenter-HVAC Copilot · 用项目反向学 Agent 的 8 周路线

> **背景前提**：项目用 AI 搭起来了，你没读过代码。这份计划就是把"读代码 + 补理论 + 自己重写关键片段 + 面试讲法"四件事捏成 8 周可执行的工作流。
> **目标**：8 周后能在面试中被追问任意一个核心模块都答得出"为什么这么设计 / 自己能不能写 / 重新做会改什么"——也就是 [career_plan.md](career_plan.md) 里那条最重要的红线："真正拥有你的项目"。
> **节奏**：每周 10-15 小时，对应 career_plan 里"每周 10-20h"的承诺。每周末交一次"自检产出"。
> **核心原则**：每个模块都按 `读代码 → 补理论 → 不看代码手写 → 写一段 100-200 字的"我会怎么改"` 四步走。手写不出来的就是没掌握。

> [!IMPORTANT]
> **2026-05-23 更新**：经过 4 轮代码审计 + Tier 1 优化（详见 [tier1_progress_review_v2.md](tier1_progress_review_v2.md) / [final_review.md](final_review.md)），项目现在比写这份计划时多了：
> - `react_agent.py`（W5/W6 多一个对照）
> - `GroundedRAGPipeline` + 三组对照（W4 已有真实 trade-off 数据）
> - `safety_adversarial.jsonl` + `policy_benchmark.py`（W7 多两个测试模块）
> - 24/24 人工标注完成（W7 从"做标注"改为"重测信度盲打分"）
> - GitHub Actions CI / 21 个 git commit / 178 测试全绿
>
> 路线节奏不变，只是每周读的具体文件和"实操"部分的对比数据**已经更新到当前真实指标**——你不会再看到过期的 0.554 或者写"未来要做"的 baseline。


---

## 学习总览：8 周分配

| 周次 | 主题 | 核心代码 | 补的理论 | 自检产出 |
|---|---|---|---|---|
| **W1** | 项目跑通 + 心智模型 | 整个调用链一遍 | RAG / Agent 概念地图 | 一张架构图 + 一段 30s 讲述 |
| **W2** | RAG 检索基础 | retrieval/* | BM25 / TF-IDF / 中文分词 | 手写一个 BM25 retriever |
| **W3** | Dense 检索 + 向量库 | embeddings / faiss_retriever / dense | Embedding / FAISS / 余弦 | 手写一个 mini FAISS-like 索引 |
| **W4** | Hybrid + Rerank + Query Rewrite | rag.py / query_rewrite | RRF / 重排器 / HyDE | 复现 hybrid 融合公式 + 写实验报告 |
| **W5** | Agent 路由 + 工具调用 | agent/router / executor | ReAct / Function Calling / Tool Use | 手写一个 mini agent loop |
| **W6** | LangGraph 工作流 | langgraph_workflow / intent_classifier | StateGraph / 条件边 / 状态机 | 自己用 LangGraph 写一个新场景 |
| **W7** | 评测体系 + Safety Audit | evaluation/* / answer_audit | RAGAS / LLM-as-Judge / 边界 | 给 24 条做人工标注 + 写评测博客 |
| **W8** | 面试串讲 + 整体优化 | 全部 | 八股文串联 | 5min/3min/30s 三个版本讲述 |

> [!IMPORTANT]
> 这个表是骨架。每周展开如下，**重点是"自检产出"必须做出来**——产出做不出来就是没学会，不要往下走。

---

## 🟢 第 1 周：跑通 + 建立心智模型（10h）

**目标**：你能在不看代码的情况下，画出"问题进来 → 答案出去"的完整流程图，并知道每个模块在哪个文件。

### 第 1-2 天：环境 + 跑通（4h）

按 [README.md](README.md) 装环境（推荐 conda + Python 3.12），跑：

```bash
# 1. 装核心 + dense 依赖（dense 可选但强烈装上，否则没真正的语义检索）
pip install -e ".[dev,dense]"

# 2. 跑测试
pytest tests/ -q

# 3. 起 API
uvicorn src.api.app:app --reload --port 8000
# 浏览器访问 http://localhost:8000/docs

# 4. 起 Streamlit（另开终端）
streamlit run app/streamlit_app.py

# 5. 跑评测，看 11 组 baseline 结果
python scripts/run_eval.py
```

**自检**：
- [ ] `/health` 返回 200
- [ ] Streamlit 三种 walkthrough case 都能点完
- [ ] `data/eval/baseline_comparison.json` 重新生成且与 git 里一致
- [ ] 看一遍 `docs/experiment_report.md` 表格，能说出每行的含义

### 第 3-4 天：调用链跟踪（4h）

**这是最重要的一步。** 选 2 个具体问题，用调试器 / `print` / `logger` 跟一遍：

1. **问题 A（doc_qa）**："数据中心冷却系统为什么可能出现高能耗？"
2. **问题 B（policy）**："建议如何调整 HVAC 控制策略？"

按这条调用链一行一行看：

```
app/streamlit_app.py
  → app/api_client.py             (HTTP POST /ask)
  → src/api/app.py                (FastAPI handler)
  → src/api/demo_factory.py       (build_demo_orchestrator — 注意它装配了什么)
  → src/agent/langgraph_workflow.py  (LangGraphOrchestrator.run)
      → src/agent/intent_classifier.py     (classify)
      → src/agent/executor.py              (run_xxx)
          → src/retrieval/rag.py           (ExtractiveRAGPipeline / GroundedRAGPipeline)
          → src/tools/timeseries.py        (各种 query_xxx)
          → src/policies/*.py              (策略工具，含 dropt_adapter)
      → src/agent/answer_generator.py
      → src/agent/answer_audit.py
```

**追加**：第三个问题（multi-hop）跑一遍 `src/agent/react_agent.py`：

3. **问题 C（multi-hop）**："最近一小时温度趋势是否提示需要调整 HVAC 控制策略？"（来自 `data/eval/hvac_eval.jsonl` 里 `multihop_001` 这一类）

观察 ReAct planner 怎么先走 timeseries_query 再走 policy_recommendation——这是项目里**唯一会触发两步执行**的代码路径。

**做法**：开 VS Code 设断点（或在每个文件开头加 `print(f">>> entered {__name__}")`），跑一次问题 A 看打印顺序；再跑问题 B 看分歧点。

**自检**：
- [ ] 在白纸上画出问题 A 和问题 B 的调用链，标出**关键分叉点**（intent 决策、是否调工具、Safety Audit 触发的条件）
- [ ] 答出："`demo_factory` 决定了这个 demo 用 keyword 还是 dense 还是 hybrid 检索器"——找出来在哪一行
- [ ] 答出："为什么问题 B 即使被路由到 policy_recommendation，LLM 也不能直接生成控制动作？"——在 `answer_audit.py` 里找对应的规则

### 第 5-7 天：补 RAG / Agent 概念地图（2h）

读以下内容，每篇做一句话笔记：

- 李宏毅 2024 GenAI 课程（B站搜"李宏毅 2024 生成式 AI"）：第 1-3 讲（Transformer / Attention / 大模型范式）
- Anthropic 官方文档 "Building effective agents"（搜 anthropic effective agents）：理解 workflow vs agent 的区别
- LangChain RAG 教程首页（concept 部分）：理解 retrieval + augment + generate 三步
- Anthropic Cookbook 的 "Tool Use" 章节：理解 function calling 的本质

**自检产出 1（必须做完）**：
1. 用 Mermaid 画一张完整架构图（取代 README 里的 ASCII），保存到 `docs/images/architecture.md`
2. 写一段 30 秒电梯版项目介绍（约 100 字）
3. 列出 5 个你这周想清楚的问题（如"为什么 router 用关键词不用 LLM？"），下周开始一个个答

> [!TIP]
> 这周不追求懂细节，**追求能找到代码 + 能复述大流程**。8 周路线最容易死的是第 1 周——很多人卡在装环境就放弃了。装不上就发我截图，我帮你 debug。

---

## 🟡 第 2 周：RAG 检索基础（BM25 / Keyword）（12h）

**目标**：能不看代码手写一个 BM25 检索器，能讲清楚 TF-IDF 和 BM25 的区别。

### 必读代码

- [src/retrieval/schemas.py](src/retrieval/schemas.py) — `DocumentChunk` 数据结构
- [src/retrieval/loader.py](src/retrieval/loader.py) — 文档加载
- [src/retrieval/chunking.py](src/retrieval/chunking.py) — 分块逻辑
- [src/retrieval/retriever.py](src/retrieval/retriever.py) — `KeywordRetriever` + `HybridRetriever`（重点）
- [src/retrieval/rag.py](src/retrieval/rag.py) — `ExtractiveRAGPipeline`
- [tests/test_retrieval_pipeline.py](tests/test_retrieval_pipeline.py) — 跑通这个测试，理解期望的输入输出

### 必补理论（4h）

- **TF-IDF**：词频 × 逆文档频率，理解为什么需要 IDF（防止常见词主导）
- **BM25**：在 TF-IDF 基础上加饱和函数 (k1) 和长度归一化 (b)。读 wikipedia BM25 词条 + 1 篇中文博客
- **中文分词的坑**：项目用了 `[A-Za-z0-9_\-一-鿿]+` 正则——这是把每个中文字符当一个 token，不是真正的分词。**找出来这个正则在 [retriever.py:11](src/retrieval/retriever.py#L11)**
- **chunking 策略**：固定长度 vs 语义分块 vs 递归分块。看 LangChain 的 `RecursiveCharacterTextSplitter` 文档
- **召回 / 精确率 / Recall@k**：理解评测里 `context_recall` 是什么意思

### 实操（6h）

1. **完全不看 [retriever.py](src/retrieval/retriever.py)**，自己用 Python 写一个 BM25 检索器：
   - 输入：`list[DocumentChunk]` + query
   - 输出：top-k 个 chunk + score
   - 限制：只用标准库（不能用 scikit-learn / rank_bm25）

2. 写完后对比项目里的实现，回答：
   - 项目里的 `_score` 函数是 TF-IDF 还是 BM25？提示：看有没有 k1 / b 参数（答案：是 TF-IDF 变体，不是严格 BM25）
   - 为什么 [retriever.py:54](src/retrieval/retriever.py#L54) 的 IDF 用 `log((1 + N) / (1 + df)) + 1`？这个 +1 平滑是干嘛的？
   - HybridRetriever 的"BM25-style"和真正的 BM25 差在哪？

3. 跑一次评测，看 `rag_keyword` 的 `citation_hit_rate=0.554`，问自己：**如果换成真正的 BM25（带 k1/b）会更好吗？写一段假设并设计实验**。

**自检产出 2**：
- 一个独立的 `my_bm25.py` 文件，能跑通自己写的几个 unit test
- 一段 200 字的"BM25 vs TF-IDF vs project 当前实现"对比

### 面试可复用知识点
- "我的 keyword retriever 是 TF-IDF 变体而非严格 BM25——这是 deterministic baseline 的故意取舍，要的是可复现而非 SOTA。如果要上 BM25，会引入 k1=1.5、b=0.75 这两个常用参数，并把 token 长度归一化加进去"

---

## 🟡 第 3 周：Dense 检索 + 向量库（12h）

**目标**：能讲清楚 embedding 是什么、FAISS 在做什么、为什么 dense 在你这个场景比 keyword 好。

### 必读代码

- [src/retrieval/embeddings.py](src/retrieval/embeddings.py) — `DeterministicHashEmbeddingProvider` + `SentenceTransformerEmbeddingProvider`
- [src/retrieval/dense.py](src/retrieval/dense.py) — 纯 Python 内存版 dense
- [src/retrieval/faiss_retriever.py](src/retrieval/faiss_retriever.py) — FAISS 包装
- [tests/test_dense_retrieval.py](tests/test_dense_retrieval.py)

### 必补理论（4h）

- **Word Embedding → Sentence Embedding**：从 word2vec / GloVe 到 BERT [CLS] 到 sentence-transformers 的演进
- **对比学习的直觉**：为什么 sentence-transformers 用 triplet/MNR loss 训练，目的是什么
- **BGE / E5 / text-embedding-3 的区别**：看 MTEB leaderboard，理解中文场景为什么用 BGE
- **FAISS 索引类型**：IndexFlatIP（精确）vs IVF（聚类近似）vs HNSW（图索引）的 trade-off
- **余弦相似度 vs 内积**：normalize 之后两者等价。理解 [embeddings.py:62](src/retrieval/embeddings.py#L62) 的 `_normalize` 在干嘛

### 实操（6h）

1. **不看代码**，用 numpy 实现一个 mini "FAISS-like" 索引：
   - 接受 `list[list[float]]` 向量
   - 实现 `add` 和 `search(query_vec, k)`，返回 top-k 索引和距离
   - 用 `np.dot` 做内积；不准用 faiss / scikit-learn

2. 在 Python 里手动跑一遍：
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
   vecs = model.encode(["数据中心冷却", "HVAC 控制策略", "今天天气真好"])
   # 自己算余弦相似度，验证前两句相似度 > 后两句
   ```

3. 阅读项目的 `DeterministicHashEmbeddingProvider`，回答：
   - 为什么这个不是真正的语义嵌入？提示：看 [embeddings.py:48-55](src/retrieval/embeddings.py#L48-L55)
   - 它能 work 吗？跑评测时 hash embedding 的 `rag_dense` 指标 vs 真实 BGE 的差距是多少？
   - 这个 hash provider 的存在是不是水分？答：**不是**，它的价值是"无依赖 fallback + 测试时不需要下载模型"——这是个工程设计，可以在面试里讲

4. 比较 `rag_keyword` (0.554) vs `rag_dense` (0.692)，写一段 200 字解释：**为什么 dense 在中文 HVAC 文档上赢了 ~0.14？哪些查询是 dense 赢的？哪些是 keyword 赢的？**（提示：去 `baseline_predictions.jsonl` 里找具体例子）

**自检产出 3**：
- `my_mini_faiss.py` + 跑通的 demo
- 一段 300 字的 "Hash Embedding vs BGE 在 HVAC 文档上的实测对比"

### 面试可复用知识点
- "我做了 hash embedding (deterministic) 和 BGE-small-zh (真实语义) 的对比，发现 dense citation_hit_rate 从 0 → 0.692，证明语义检索在中文小语料上的必要性。FAISS 用 IndexFlatIP 因为我们文档量只有 ~100 chunk，不需要 IVF/HNSW 的近似"

---

## 🟡 第 4 周：Hybrid + Rerank + Query Rewrite（12h）

**目标**：理解"多路召回 + 融合"为什么不一定比单路好；能讲清 HyDE 和 Query Rewrite 的区别。

### 必读代码

- [src/retrieval/retriever.py](src/retrieval/retriever.py) `HybridRetriever` 部分（BM25-style + 长度归一化）
- [src/retrieval/rag.py](src/retrieval/rag.py) — 重点看 **`ExtractiveRAGPipeline` vs `GroundedRAGPipeline` 的差异**
- [src/retrieval/query_rewrite.py](src/retrieval/query_rewrite.py) — `RuleBasedHVACQueryRewriter` + `TemplateHyDEGenerator`
- [tests/test_query_rewrite.py](tests/test_query_rewrite.py)
- [tests/test_grounded_rag.py](tests/test_grounded_rag.py)

### 必补理论（4h）

- **Reciprocal Rank Fusion (RRF)**：最常见的多路融合公式 `1 / (k + rank)`，看一篇博客
- **Cross-Encoder 重排器**：BGE-Reranker / Cohere Rerank 的原理，为什么比 bi-encoder 准但慢
- **HyDE 论文核心**："Precise Zero-Shot Dense Retrieval without Relevance Labels"。读 abstract + 第 3 节
- **Query Expansion vs HyDE**：前者加同义词/术语，后者生成假设答案再去检索

### 实操（6h）

1. 看你的实验数据（**已跑出 grounded 三组对照 + BGE 真实数据**）：

| Baseline | citation_hit_rate | grounding_rate | expected_keyword_coverage |
|---|---:|---:|---:|
| rag_keyword | 0.554 | 0.000 | 0.353 |
| rag_keyword_grounded | 0.554 | **0.708** | 0.344 |
| rag_dense | **0.692** | 0.000 | 0.502 |
| rag_dense_grounded | 0.692 | **1.000** | 0.492 |
| rag_rewrite | 0.646 | 0.000 | 0.566 |
| rag_rewrite_grounded | 0.646 | **1.000** | 0.477 |
| rag_hyde | **0.246** | 0.000 | 0.174 |

2. 写 300 字解释 **三个反直觉**——这是面试最好的素材：
   - 为什么 hybrid 不如纯 dense？（hybrid 里的 dense 用 hash 拖了后腿）
   - 为什么 HyDE 反而最差？（template 让 query 漂移到模板词）
   - **为什么 grounded 版本 grounding_rate 上去了，expected_keyword_coverage 反而下降**？（拼接式答案 keyword 召回更高 → grounded 不是免费午餐，是 trade-off）

3. **手写 RRF 融合**：把 keyword 和 dense 的结果融合成一个排序，对比项目里的加权融合（看 HybridRetriever 的实现），哪个更好？

4. **设计一个新的 Rewrite 策略**（可选挑战）：用 DeepSeek 真实 LLM 生成 hypothetical document，跑一遍看比 template HyDE 强多少。

**自检产出 4**：
- 一篇 800 字的"实验报告"博客草稿，标题就叫《为什么我的 HyDE 跑得比 keyword 还差》——这是 career_plan 里推荐的博客选题之一
- 一段 RRF 实现对比
- 一段对 grounded trade-off 的反思（"我以为 grounded 一定更好，结果发现…"）

### 面试可复用知识点（高级）
- "我跑了 11 组检索 baseline，反直觉的发现是 deterministic template HyDE 反而比 keyword 还差（0.246 vs 0.554）。这让我意识到 HyDE 在中文小语料上会引入 query drift——模板词稀释了原始关键词。"
- "更深的发现：grounded 三组对照里 grounding_rate 1.0 但 keyword coverage 反而下降——所以 grounded 不是免费午餐，是 trade-off。下一步是混合策略：grounded 时显式保留检索证据里的 expected keyword。"
- 这是面试官最想听的"我做过实验、有判断、不迷信论文"的故事

---

## 🟠 第 5 周：Agent 路由 + 工具调用（12h）

**目标**：能讲清"什么是 Agent"、"为什么用 Agent 而不是单纯 RAG"、能手写一个 mini agent loop。

### 必读代码

- [src/agent/router.py](src/agent/router.py) — 关键词路由（看下面贴的实现）
- [src/agent/intent_classifier.py](src/agent/intent_classifier.py) — Rule / DeepSeek / Ollama 三种 intent
- [src/agent/executor.py](src/agent/executor.py) — `AgentTaskExecutor`，工具执行的中枢
- [src/agent/orchestrator.py](src/agent/orchestrator.py) — `BaselineOrchestrator`（确定性版本）
- [src/agent/react_agent.py](src/agent/react_agent.py) — **`ReActOrchestrator` + `DeterministicReActPlanner`，项目里唯一的 multi-step 路径**
- [src/tools/timeseries.py](src/tools/timeseries.py) — 5 个时序工具
- [src/policies/](src/policies/) — 策略工具

### 必补理论（4h）

- **ReAct 论文核心**：`Thought → Action → Observation → Thought ...` 循环，理解为什么要把推理和行动交替
- **Function Calling 原理**：OpenAI / Anthropic 怎么把 function schema 喂给模型，模型怎么"决定"调哪个
- **Tool Use 的三种范式**：
  - 关键词路由（你项目的默认）—— 简单但脆弱
  - LLM 单步决策（你项目的 LLMIntentClassifier）—— 灵活但贵
  - ReAct 多步循环（项目暂未实现）—— 强大但难调
- **AnthropOphic "Building effective agents"**：精读这篇，理解 workflow（确定性图）vs agent（LLM 自主决策）的边界——你的项目是 workflow，不是真正的 agent

### 实操（6h）

1. 读 [router.py](src/agent/router.py)，看 17 行的关键词路由（很短），回答：
   - 为什么 anomaly 关键词放第一位检测？
   - 如果用户问"温度有异常吗"会被分到哪一类？是不是错的？（这是你 confusion matrix 里 timeseries → anomaly 误判的来源）

2. **手写一个 mini agent loop**（不看 [executor.py](src/agent/executor.py)）：
   ```python
   tools = {
       "search_doc": lambda q: "...",
       "query_metric": lambda metric: "...",
       "detect_anomaly": lambda: "...",
   }
   def my_agent(question: str, max_steps=3):
       # 1. 决定调哪个工具
       # 2. 调用 + 拿 observation
       # 3. 拿 observation 回去判断要不要再调 / 直接答
       ...
   ```
   先用关键词路由实现，再用 DeepSeek API 实现 LLM 版本，对比两者效果。

3. 读 `intent_routing_comparison.json`，看 rule_based 准确率 0.64，confusion matrix 显示 document_qa 经常被错路由。**写 200 字假设：如果换成 LLM 路由，准确率能到多少？为什么？**

4. 跑一次 `python scripts/run_intent_eval.py --providers rule_based deepseek`（如果你有 API key），把数据补到 `intent_routing_comparison.json`。**这条单独做就能让你在简历里多写一行真实数据**。

**自检产出 5**：
- `my_mini_agent.py` 能跑通，至少支持 3 个 mock 工具
- 一段 300 字的"ReAct vs 单步 LLM 决策 vs 关键词路由"对比

### 面试可复用知识点
- "我的 Agent 是 workflow 形式（确定性图 + LLM 在 intent 节点单步决策），不是 ReAct 多步循环。这是因为 HVAC 控制场景每多一次 LLM 决策就多一次不确定性，而我的工具是确定性的（time-series query / policy adapter），不需要 ReAct 的反思链"
- "rule_based intent accuracy 0.64，混淆主要在 document_qa → timeseries（关键词'温度'被两类共享）。LLM intent 节点是为这种边界 case 设计的"

---

## 🟠 第 6 周：LangGraph 工作流（12h）

**目标**：能不看代码用 LangGraph 写一个新场景，能讲清 StateGraph 的设计哲学。

### 必读代码

- [src/agent/langgraph_workflow.py](src/agent/langgraph_workflow.py) — 全文 203 行，一行不漏读
- 对比 [src/agent/orchestrator.py](src/agent/orchestrator.py) deterministic 版本

### 必补理论（4h）

- **LangGraph 官方教程**：完整跑一遍 introduction 和 quickstart（约 1.5 小时）
- 核心概念：**State / Node / Edge / Conditional Edge / Compile**
- **TypedDict 状态**：为什么 `WorkflowState(TypedDict, total=False)`，total=False 是干嘛
- **可观测性**：`workflow_trace` 在每个节点 append 自身行为——这就是 LangSmith / LangFuse 替代品的雏形
- **Checkpointer / Memory**：项目暂未用，但要知道 LangGraph 支持持久化中断后恢复

### 实操（6h）

1. **不看代码**用 LangGraph 写一个新场景：
   - 假设场景：FAQ 客服 Agent
   - 节点：`classify`（分到 billing / technical / general）→ 各自的 handler → `quality_check` → END
   - 状态：`question / category / answer / quality_score`
   - 必须用条件边

2. 看完 [langgraph_workflow.py](src/agent/langgraph_workflow.py) 后回答：
   - `_select_route_node` 在哪一步执行？返回字符串后 langgraph 怎么用？（提示：`add_conditional_edges` 的第三个 dict 参数）
   - 为什么 [langgraph_workflow.py:69](src/agent/langgraph_workflow.py#L69) 4 个工具节点都连到 `evidence_aggregator`？
   - 如果想加一个 "replan"（当 retrieval 召回为 0 时回到 intent_classifier 重判），怎么改？提示：加一个 conditional edge from evidence_aggregator

3. **加一个真实改进**（可选挑战）：把 `replan` 节点加进 langgraph，让低召回的问题自动尝试 query rewrite。这是 P2 改进里写过的，但你自己写出来比 AI 写的有面试价值得多。

4. 比较 `rag_tool_agent`、`langgraph_tool_agent`、`react_agent` 三组指标：
   - 100 条单步问题上：三组**指标完全相同**（同一份 task_executor）
   - 但 `react_agent` 在 8 条 multi-hop policy 子集上 tool_selection_accuracy 从 71.4% 提到 89.3%（+25%）
   - **回答：那 langgraph 的价值是什么？**（答：StateGraph 编排可视化、workflow_trace 可观测、intent 节点可插拔——这些是工程价值。**ReAct 的价值是 multi-hop**——这是 multi-step 才能解决的场景）

**自检产出 6**：
- 你独立写的 FAQ Agent LangGraph，要能跑
- 一张你画的 langgraph_workflow 状态转移图（用 Mermaid）

### 面试可复用知识点
- "用 LangGraph 是为了三件事：StateGraph 可视化、workflow_trace 可观测、intent 节点可插拔。它和我的 deterministic baseline 在指标上完全相同——这是有意为之，因为换路由实现不应该改工具行为，否则评测就被污染了"
- "如果让我重做，我会加一个 replan 节点处理低召回情况——但这是 P2 优化，不是 MVP 必须"

---

## 🔴 第 7 周：评测体系 + Safety Audit（12h）

**目标**：能设计一个新场景的评测集；理解为什么 Safety Audit 用规则不用 LLM。

### 必读代码

- [src/evaluation/dataset.py](src/evaluation/dataset.py) — eval JSONL 加载
- [src/evaluation/metrics.py](src/evaluation/metrics.py) — 9 项指标定义（核心，含 `grounding_rate`）
- [src/evaluation/runner.py](src/evaluation/runner.py) — baseline runner
- [src/evaluation/llm_judge.py](src/evaluation/llm_judge.py) — `DeterministicKeywordJudge`
- [src/evaluation/human_review.py](src/evaluation/human_review.py)
- [src/evaluation/safety_adversarial.py](src/evaluation/safety_adversarial.py) — **对抗鲁棒性测试模块**
- [src/evaluation/policy_benchmark.py](src/evaluation/policy_benchmark.py) — **DROPT 独立 baseline 跑 latency / action 分布**
- [src/agent/answer_audit.py](src/agent/answer_audit.py) — Safety Audit 规则
- [data/eval/hvac_eval.jsonl](data/eval/hvac_eval.jsonl) 前 10 条
- [data/eval/safety_adversarial.jsonl](data/eval/safety_adversarial.jsonl) — 29 条对抗 prompt
- [data/eval/baseline_comparison.json](data/eval/baseline_comparison.json) 完整结构

### 必补理论（4h）

- **RAG 评测三个维度**：检索质量 / 生成质量 / 端到端正确性
- **RAGAS 框架核心指标**：context_recall / context_precision / faithfulness / answer_relevancy。对照你项目的指标，哪些是真 RAGAS、哪些是 proxy
- **LLM-as-Judge 的坑**：position bias / length bias / 自我吹捧 bias。读一篇综述
- **人工标注的价值**：唯一不被 model bias 污染的 ground truth；通常用 inter-annotator agreement 衡量
- **Safety / Guardrails**：为什么不能让 LLM 自己审计自己——读一篇 Anthropic 的 Constitutional AI 简介

### 实操（6h）

1. **复核 24 条人工标注**（5/22 已经标完，本周做的是"重新读自己的标注 + 校准"）：
   - 打开 [data/eval/human_review_annotations.jsonl](data/eval/human_review_annotations.jsonl)
   - **不看自己当时填的分数**，重新对每条样本评一次（盲打分）
   - 对比两次评分的差异——**这就是你自己的 inter-rater agreement**，是面试可以讲的"我做了重测信度"故事
   - 跑 `python scripts/run_eval.py` 看 experiment_report.md 里 Human Calibration 段的数据
   - 计算你的人工分和 deterministic proxy 的 Pearson 相关系数

2. 读 [metrics.py](src/evaluation/metrics.py)，回答：
   - `citation_hit_rate` 和 `context_recall` 在你的实现里为什么数值相同？（看实现细节）
   - `expected_keyword_coverage` 是怎么算的？为什么这个比 citation 更接近"答案对不对"？
   - `answer_correctness_proxy` 和 `faithfulness_proxy` 是怎么算的？为什么叫 proxy？
   - `grounding_rate` 是怎么算的？为什么 extractive baseline 一律是 0？

3. 读 [answer_audit.py](src/agent/answer_audit.py) + [safety_adversarial.jsonl](data/eval/safety_adversarial.jsonl)，回答：
   - 三类 Safety 违规分别是什么？
   - **跑过的 29 条对抗 prompt 里 translation 类为什么 0/4 全军覆没？**（答：risky_phrases 字典是中文）
   - 为什么 paraphrase 反而 8/8 全命中？
   - 写 100 字答案：为什么 Safety Audit 用确定性规则而非 LLM？

4. **读 [policy_benchmark.py](src/evaluation/policy_benchmark.py)** + experiment_report.md 末尾的 DROPT 数据：
   - 28/28 推理成功、6.5ms latency 这些数字怎么来的？
   - "sub-10ms 适合实时控制循环"这个论断的依据是什么？
   - 如果让你给 DROPT 加一个新指标，你会加什么？（提示：action diversity / std）

5. **设计挑战**：给一个新场景（比如智能家居控制）设计一个 30 条的评测集 + 5 项指标 + Safety 规则。**不写代码，只写设计文档（500 字）**。

**自检产出 7**：
- 你的两次盲打分对比 + 一致性分析（200 字）
- 一段 200 字的"我的人工分和 proxy 的相关性"分析
- 一篇 1500 字博客《如何为垂直领域 RAG Agent 设计 100 条评测集》草稿

### 面试可复用知识点
- "我的评测有三层：deterministic proxy（自动跑、可复现）、optional LLM judge（可信但贵）、人工标注（24 条 ground truth，与 proxy Pearson r=X，证明 proxy 可信）"
- "Safety Audit 用确定性规则不用 LLM，是因为安全边界不能依赖概率模型——LLM 在 99% 时间正确不代表 1% 错误时不闯祸。三类规则分别检查：生产遥测误述、LLM 直接控制声明、未验证策略动作"

> [!IMPORTANT]
> 这周的人工标注是 [project_review_2026_05_22.md](project_review_2026_05_22.md) 里的 P0-2，**做完后简历就能多一句"含人工校准 24 条"**。这是这 8 周里 ROI 最高的单一动作之一。

---

## 🔴 第 8 周：面试串讲 + 整体优化（10h）

**目标**：3 个版本的项目讲述（30s / 3min / 5min）+ 5 个高频追问的标准答案。

### 必做的 5 件事

#### 1. 录三个版本的项目讲述并自评（3h）

按 [career_plan.md](career_plan.md) 第九节的建议：

| 版本 | 时长 | 内容骨架 |
|---|---|---|
| **电梯版** | 30s | 一句话讲清"什么场景 + 用了什么 + 量化结果"。例："我做了一个 RAG + Tool Agent 系统，面向 BEAR HVAC 仿真，工具路由 100%、证据覆盖 91%" |
| **标准版** | 3min | 场景痛点 → 架构选型（为什么 LangGraph 不直接 ReAct）→ 关键技术（hybrid 检索 + Safety Audit）→ 量化结果 |
| **深度版** | 5-8min | 上述 + 一个有反直觉的实验故事（HyDE drift）+ 一个"如果重做会改什么"的回答 |

录音用手机就行，自己听一遍——卡壳的地方就是没真懂的地方，回去再读对应模块。

#### 2. 准备 5 个高频追问的标准答案（2h）

来自 [career_plan.md](career_plan.md) 的清单：

1. **"你的检索用了什么方案？对比过哪些？"**
   - 答：keyword (TF-IDF变体) → dense (BGE-small-zh + FAISS) → hybrid → hybrid+rerank → rewrite → HyDE，11 组对比，dense 单路最好（0.692），HyDE 反而最差，所以生产用 rewrite

2. **"你的 Agent 是怎么做路由的？"**
   - 答：默认 rule-based 关键词（accuracy 0.64），LangGraph 的 intent 节点可插拔切到 DeepSeek/Ollama LLM 路由。rule-based 的设计是可复现 baseline 而非最优解

3. **"为什么不让 LLM 直接控制 HVAC？"**
   - 答：高风险决策不能依赖概率模型。控制动作来自 RL/扩散策略适配器，LLM 只做证据整合 + Safety Audit 三类规则审计

4. **"你的评测怎么保证可信？"**
   - 答：三层。deterministic proxy 自动跑、optional LLM judge、人工标注 24 条 ground truth；后者与 proxy 的 Pearson r=X 证明 proxy 可信

5. **"你的扩散模型/RL 在这里什么角色？"**
   - 答：通过 `dropt_adapter` 将论文 checkpoint 包装成工具，被 Agent 在 policy_recommendation 路由调度。这是"LLM 解释 + 策略工具执行"的分工范式

#### 3. 完成 [project_review_2026_05_22.md](project_review_2026_05_22.md) 的 P0 任务（4h）

把那份评估里 P0-1 ~ P0-4 全做完：截图、人工标注、DeepSeek intent、CI+lint+logging。这些都是已经反复强调的低成本高回报项。

#### 4. 简历定稿 v1（1h）

用 [project_review_2026_05_22.md](project_review_2026_05_22.md) 第三节里那段简历 paragraph 作为模板。

#### 5. 写一篇博客发出去（持续）

career_plan 推荐的选题里挑一个：
- 《为什么 deterministic HyDE 在中文 HVAC 场景反而掉点》（W4 草稿改）
- 《如何为垂直领域 RAG Agent 设计 100 条评测集》（W7 草稿改）
- 《LLM 不能直接控制 HVAC：一种 Safety Audit 的最小实现》

发知乎或掘金，简历里加一行"个人技术博客：[链接]"。

---

## 总自检清单（8 周末必须能勾完）

### 项目掌握度

- [ ] 闭着眼能画出从 user question → answer 的完整调用链
- [ ] 能在白板上手写 BM25 / 简化 FAISS / mini agent loop 三段代码
- [ ] 能讲清 11 组 baseline 每一组的指标和反直觉发现
- [ ] 能用 LangGraph 写一个全新场景（不抄项目代码）
- [ ] 能解释 Safety Audit 三类规则和"为什么不用 LLM"
- [ ] 能给一个新领域设计 30 条评测集 + 5 项指标的方案文档

### 项目状态（与 [project_review_2026_05_22.md](project_review_2026_05_22.md) P0 对齐）

- [ ] `docs/images/` 有 6 张以上截图，README 头部显示
- [ ] 24 条人工标注全部填完，experiment_report 不再 pending
- [ ] `intent_routing_comparison.json` 含 deepseek 或 ollama 真实数据
- [ ] `.github/workflows/ci.yml` 跑通 + README 有 CI badge
- [ ] `pyproject.toml` 加了 ruff / mypy 依赖
- [ ] `src/core/logging_config.py` 存在，3 个核心模块用上 logger

### 面试就绪度

- [ ] 30s / 3min / 5min 三个版本能不卡讲
- [ ] 5 个高频追问有 100 字左右的标准答案
- [ ] 至少发了 1 篇技术博客
- [ ] 简历 v1 定稿，每个数字都能溯源到 baseline_comparison.json

---

## 必读资源汇总（不要超过这个清单）

| 类别 | 资源 | 优先级 |
|---|---|---|
| 大模型基础 | 李宏毅 2024 GenAI 课程 第 1-5 讲（B站） | ⭐⭐⭐⭐⭐ |
| Transformer | 3Blue1Brown 的 Transformer 可视化（YouTube） | ⭐⭐⭐⭐⭐ |
| Agent 概念 | Anthropic "Building effective agents" | ⭐⭐⭐⭐⭐ |
| RAG 论文 | RAG 原论文（Lewis 2020）+ HyDE（Gao 2022）| ⭐⭐⭐⭐ |
| LangGraph | LangGraph 官方 quickstart + intro | ⭐⭐⭐⭐⭐ |
| BM25 | wikipedia BM25 词条 + 1 篇中文博客 | ⭐⭐⭐ |
| FAISS | FAISS Wiki 的 Index types 页面 | ⭐⭐⭐ |
| 中文 Embedding | BGE GitHub README + MTEB leaderboard | ⭐⭐⭐⭐ |
| 评测 | RAGAS GitHub README + 1 篇 LLM judge bias 博客 | ⭐⭐⭐⭐ |
| Function Calling | Anthropic Cookbook tool_use 章节 | ⭐⭐⭐⭐ |

> [!CAUTION]
> **不要再加资源了。** 资源越多越焦虑越不学。这个清单全过一遍是 30-40 小时，已经覆盖了 8 周里的"补理论"时间。其余时间全部花在读代码 + 自己写 + 跑实验上。

---

## 给你的几个真心提醒

> [!IMPORTANT]
> 1. **AI 代写代码不是问题，不读代码才是问题。** 8 周路线就是把"AI 代写"补成"自己拥有"——很多创业公司创始人、独立开发者都是这么走的。坦诚比假装重要。
>
> 2. **手写比读代码重要 10 倍。** 每周的"实操"环节（手写 BM25 / mini FAISS / mini agent / 新 LangGraph 场景）才是真正长肌肉的部分。读代码 1 小时学到的东西不如手写 30 分钟。
>
> 3. **不要中途加新功能。** 每周都会冒出"要不要顺便加个 X"的冲动——别加。8 周里**只允许做改进建议清单里的事**，新功能等 9 月以后。
>
> 4. **每周末发我一次进度。** 把当周的"自检产出"截图给我，我帮你看哪里不到位。这种 checkpoint 比闭门学三个月有效得多。
>
> 5. **不会的就直接说不会。** 面试官不是在找"什么都会的人"——他们在找"会什么、不会什么、不会的怎么补"路径清晰的人。AI 搭项目这件事本身可以讲（不要主动说，但被问到 "项目里 AI 帮了多少" 时可以坦诚说"架构和初版我用 AI 加速了，后续做的所有实验、调试、改进都是自己"）——比假装全是自己写的有面试价值得多。

---

## 一周一句话进度模板（每周末发我）

```
W{N} · {主题} · 进度 X%
✅ 做完了：...
🟡 卡住的：...
❓ 想确认的：...
下周想做：...
```

---

*最后更新：2026-05-22 · 与 [career_plan.md](career_plan.md) / [project_improvement_suggestions.md](project_improvement_suggestions.md) / [project_review_2026_05_22.md](project_review_2026_05_22.md) 配套使用*
