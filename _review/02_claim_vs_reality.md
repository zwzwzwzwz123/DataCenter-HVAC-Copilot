本文件回答：README 的宣传性卖点哪些能保留，哪些需要补强或降级。

# 卖点宣传与代码实际对照

### 卖点 1：不是普通 ChatPDF

**README 原话引用**：“系统同时支持文档问答、BEAR-like 时序查询、异常诊断和策略建议。”（`README.md:18`）

**代码实际状态**：
- 实现位置：路由 `src/agent/orchestrator.py:44`；工具执行 `src/agent/executor.py:54`、`src/agent/executor.py:76`、`src/agent/executor.py:138`、`src/agent/executor.py:166`；时序工具 `src/tools/timeseries.py:52`。
- 实现深度：8/10。四类任务都有真实分支和工具结果；但路由/工具选择主要是规则与关键词，不是强泛化 agent。
- 关键证据：`BaselineOrchestrator.run` 按 route 分派到四个 `run_*`，见 `src/agent/orchestrator.py:50`。

**经得起面试追问的深度**：
- 能撑住 30 秒讲述：是。
- 能撑住 3 分钟追问：是，但要承认任务集受控。
- 追问到什么程度会露馅：如果被问“是否能开放域调用任意工具/多轮自主规划”，需要说明不是，是固定 HVAC 任务空间。

**简历表达建议**：
- 建议降级措辞：保留“不是普通 ChatPDF”，但改成“面向固定 HVAC 场景的文档问答 + 时序工具 + 策略工具编排”。

### 卖点 2：RAG + Tool Agent 闭环

**README 原话引用**：“问题先路由，再检索文档、调用时序工具或 policy 工具，最后基于证据生成回答。”（`README.md:19`）

**代码实际状态**：
- 实现位置：deterministic route `src/agent/orchestrator.py:44`；共享 executor `src/agent/executor.py:27`；最终 answer/audit `src/agent/executor.py:188`。
- 实现深度：7/10。闭环真实存在，但 answer generator 默认是模板，LLM 只是可选；tool agent 更接近 controlled orchestrator。
- 关键证据：`generate_answer_from_evidence` 只接受 evidence payload，并随后审计，见 `src/agent/executor.py:194`、`src/agent/executor.py:212`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：是。
- 露馅点：如果把它讲成 ReAct/AutoGPT 式开放 agent，会被代码反证；它是 route-based tool workflow。

**简历表达建议**：
- 保留但措辞精确：“构建受控 RAG + Tool workflow，按任务路由调用检索、时序和 policy 工具，并基于结构化 evidence 生成回答。”

### 卖点 3：Persistent Knowledge Base 长期知识记忆

**README 原话引用**：“支持 PDF / DOCX / TXT / MD 上传，SQLite 保存 document/chunk/index 元数据，FAISS + sidecar + manifest 持久化，并让 `/ask` 默认检索上传后的知识库。”（`README.md:20`）

**代码实际状态**：
- 实现位置：parser `src/knowledge/parsers.py:19`；SQLite store `src/knowledge/storage.py:12`；rebuild `src/knowledge/indexer.py:34`；service `src/knowledge/service.py:52`；demo RAG 优先 knowledge index `src/api/demo_factory.py:46`。
- 实现深度：8/10。上传、解析、metadata、FAISS sidecar、manifest hash、全量 rebuild 都真实；不是增量索引或服务化向量库。
- 关键证据：`KnowledgeFaissIndexer.rebuild` 写 tmp index/chunks/manifest，再原子替换，见 `src/knowledge/indexer.py:43`、`src/knowledge/indexer.py:80`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：是。
- 露馅点：问大规模并发、增量索引、ACL、多租户时会露出 demo/本地实现边界。

**简历表达建议**：
- 保留原措辞，补一句“本地 FAISS + SQLite，全量重建，不宣称生产级向量数据库”。

### 卖点 4：Conversation Memory 多轮上下文

**README 原话引用**：“`/ask` 支持 session 级持久记忆，SQLite 记录每轮问题、证据、工具结果、trace 和回答，并按 `session_id` 隔离检索历史。”（`README.md:21`）

**代码实际状态**：
- 实现位置：API memory flow `src/api/app.py:76`、`src/api/app.py:210`；store schema `src/memory/storage.py:16`；session filter `src/memory/retriever.py:105`。
- 实现深度：7/10。后端实现扎实，能保存 turn、索引 chunk、按 session 过滤；UI 只做最小兼容，README 也承认不提供 session 列表/重命名/删除（`README.md:371`）。
- 关键证据：`FilteringMemoryRetriever.search` 丢弃非当前 session chunk，见 `src/memory/retriever.py:105`、`src/memory/retriever.py:111`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：基本是。
- 露馅点：如果问“长期 memory 如何做摘要更新、用户管理、session UI”，当前较弱。

**简历表达建议**：
- 需要降级措辞：“实现 session-scoped SQLite conversation memory 和检索式上下文加载”，避免“长期知识记忆”与 knowledge base 混淆。

### 卖点 5：多 LLM 后端可选接入

**README 原话引用**：“`/ask` 支持 deterministic fallback、DeepSeek 和本地 Ollama evidence-grounded answer generation；未配置或调用失败时自动回退 deterministic generator。”（`README.md:22`）

**代码实际状态**：
- 实现位置：DeepSeek `src/agent/deepseek_generator.py:20`、fallback `src/agent/deepseek_generator.py:64`；Ollama fallback `src/agent/ollama_generator.py:55`；env 选择 `src/agent/deepseek_generator.py:68`。
- 实现深度：6/10。HTTP adapter 和 fallback 真实，但不是完整 provider abstraction；没有流式、重试、结构化输出校验、token/cost logging。
- 关键证据：缺 key 时直接 deterministic，见 `src/agent/deepseek_generator.py:87`、`src/agent/deepseek_generator.py:89`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：勉强。
- 露馅点：问模型调用可观测性、失败分类、prompt eval、rate limit，会显得简单。

**简历表达建议**：
- 建议降级：“接入 DeepSeek/Ollama 可选 answer generator，并提供 deterministic fallback 保证可复现 demo。”

### 卖点 6：控制边界清晰

**README 原话引用**：“控制建议只来自 rule-based、MPC-like、DiffFNO / Guided-DiffFNO adapter 或 offline replay 等工具，LLM 不直接控制环境。”（`README.md:23`）

**代码实际状态**：
- 实现位置：policy runner 选择 `src/api/demo_factory.py:140`；rule-based `src/policies/rule_based.py:6`；MPC-like placeholder `src/policies/mpc_like.py:6`；Diffusion stub `src/policies/diffusion_adapter.py:8`；DROPT `src/policies/dropt_adapter.py:348`。
- 实现深度：7/10。边界设计明确，DROPT checkpoint 能跑；但 MPC-like 是 placeholder，Diffusion adapter 明确 `NotImplementedError`。
- 关键证据：Diffusion adapter 未实现，见 `src/policies/diffusion_adapter.py:14`、`src/policies/diffusion_adapter.py:19`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：是，但要主动区分“边界/adapter”和“完整控制算法”。
- 露馅点：如果声称实现完整 MPC 或 DiffFNO training/integration，会露馅。

**简历表达建议**：
- 保留“控制边界清晰”；把“MPC-like、DiffFNO”表述成“adapter/backends”，不要写成完整控制器。

### 卖点 7：Safety Audit

**README 原话引用**：“每个回答都会进行确定性安全审计，检查生产遥测误述、LLM 直接控制声明和未验证策略动作。”（`README.md:24`）

**代码实际状态**：
- 实现位置：`audit_answer` `src/agent/answer_audit.py:7`；调用点 `src/agent/executor.py:212`；对抗评测 `src/evaluation/safety_adversarial.py:36`。
- 实现深度：5/10。每个回答确实审计，但审计是关键词/正则；对抗集 29 条 overall hit rate 0.586，英文 translation 类 0.000，见 `docs/experiment_report.md:111`、`docs/experiment_report.md:120`。
- 关键证据：检查项固定三类，见 `src/agent/answer_audit.py:24`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：否，如果把它讲成强安全系统。
- 露馅点：问 paraphrase/英文/jailbreak 泛化，当前报告已经显示漏检。

**简历表达建议**：
- 建议降级：“实现确定性边界审计与小型 adversarial audit，作为 demo guardrail，不宣称完备安全。”

### 卖点 8：可复现评测

**README 原话引用**：“内置 100 条 JSONL 评测集，覆盖文档问答、时序查询、异常诊断和策略建议，并生成 baseline comparison 和实验报告。”（`README.md:25`）

**代码实际状态**：
- 实现位置：dataset loader `src/evaluation/dataset.py:9`；runner `src/evaluation/runner.py:46`；metrics 汇总 `src/evaluation/runner.py:350`；script `scripts/run_eval.py:46`；报告 `docs/experiment_report.md:1`。
- 实现深度：7/10。评测体系丰富，实际主 eval 是 108 条，不是 README 说的 100 条；baseline summary 有 15 个模式，见 `docs/experiment_report.md:16`。但完整 pytest 当前 1 fail，且 correctness/faithfulness 是 proxy。
- 关键证据：报告写“108 条样例”，见 `docs/experiment_report.md:5`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：是，前提是承认 proxy 指标。
- 露馅点：如果说有人审或真实业务标签，`human_review_annotations.jsonl` 全是 null（`data/eval/human_review_annotations.jsonl:1`）。

**简历表达建议**：
- 需要补强/修正后保留：README 统一成 108 条或重新裁成 100；简历写“deterministic proxy metrics + baseline comparison”，不要写人工评测。

### 卖点 9：LangGraph workflow

**README 原话引用**：“`/ask` 和 Streamlit 默认使用 `workflow_engine=langgraph`，并返回真实的 `workflow_trace`。”（`README.md:6`）

**代码实际状态**：
- 实现位置：schema 默认 `src/api/schemas.py:9`；API 分支 `src/api/app.py:185`；StateGraph `src/agent/langgraph_workflow.py:59`；Streamlit 默认选项 `app/streamlit_app.py:37`。
- 实现深度：7/10。真实 LangGraph 编排和 trace 存在；当前 graph 是 planner -> execute -> aggregate -> answer -> audit 的线性 DAG，不是复杂条件图。
- 关键证据：`StateGraph` 节点和边在 `src/agent/langgraph_workflow.py:60` 到 `src/agent/langgraph_workflow.py:72`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：基本是。
- 露馅点：问动态分支、工具循环、人工中断、状态恢复时，目前不具备。

**简历表达建议**：
- 保留但精准：“使用 LangGraph 实现受控多步 workflow trace”，不要写“复杂 agent graph”。

### 卖点 10：DROPT / Guided-DiffFNO adapter

**README 原话引用**：“项目已支持本地 `models/dropt/policy_best_fno_guided.pth` checkpoint 的推理适配器。”（`README.md:257`）

**代码实际状态**：
- 实现位置：`DROPTCheckpointPolicy` `src/policies/dropt_adapter.py:348`；checkpoint load `src/policies/dropt_adapter.py:363`；deterministic sample `src/policies/dropt_adapter.py:419`；benchmark 报告 `docs/experiment_report.md:121`。
- 实现深度：7/10。模型结构和 checkpoint load 真实，28 个 policy 样例 success 28/fallback 0；但它是离线 policy 工具，没有训练流程、真实控制闭环或对策略质量的强对比。
- 关键证据：显式 20 维 state 缺失时 fallback，见 `src/policies/dropt_adapter.py:390`、`src/policies/dropt_adapter.py:392`。

**经得起面试追问的深度**：
- 30 秒：是。
- 3 分钟：取决于作者是否能讲清 DiffFNO/checkpoint 来源与输入布局。
- 露馅点：问训练数据、loss、策略优劣、在线 rollout 对比，当前代码证据不足。

**简历表达建议**：
- 需要补强后才能保留强措辞。当前建议写：“接入本地 Guided-DiffFNO checkpoint adapter 作为 policy backend，并实现缺失/异常 fallback。”

## 直接投简历的主要翻车点

最可能翻车的 3 点：第一，README 多处写“100 条评测”，但当前主评测集和实验报告是 108 条，且 `pytest` 当前有 1 个 query rewrite 测试失败，会削弱“可复现”可信度。第二，Safety Audit 容易被问穿：它是关键词规则，对抗评测 hit rate 只有 0.586，英文类为 0。第三，DROPT/Guided-DiffFNO 如果讲成完整策略优化或工业控制，会超出代码实际；它更准确是 checkpoint adapter + policy 工具边界。
