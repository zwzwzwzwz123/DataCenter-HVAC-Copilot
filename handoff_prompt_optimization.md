# 交接给另一个 AI 的提示词

> 把下面 `---` 之间的全部内容复制给另一个 AI（Claude / Cursor / Cline / etc.），让它按 Tier 1 路线一项一项执行优化。

---

# 你的任务：DataCenter-HVAC Copilot 代码优化执行

你是 `DataCenter-HVAC Copilot` 项目的 senior engineer。前一位顾问已经做完了项目审计和代码深度审查，并写出了优化路线图。你的任务是**按计划逐项落地 Tier 1 四件优化（A/B/C/D）**。

这是一个**面试驱动型个人项目**（不是 production 系统），目标是让作者能在面试中把每个设计决策讲清楚。所以"做扎实 + 有数据 + 能讲故事" > "做得多 + 显得专业"。

---

## 第 0 步：建立心智模型（必做，不要跳）

按顺序读完这 5 个文件后再动手：

1. **`optimization_roadmap.md`** ← 你的核心工作手册。8 个代码弱点 + Tier 1 四件优化的详细方案
2. **`project_review_2026_05_22.md`** ← 项目当前状态评估，知道哪些已完成哪些没完成
3. **`career_plan.md` 第十节** ← 项目背景，知道作者的真实情况
4. **`README.md`** ← 项目架构、数据边界、模块划分
5. **`docs/experiment_report.md`** ← 当前 11 组 baseline 的指标基线

读完后用一段话向用户确认你理解了：项目类型、当前状态、Tier 1 四件事的目标。**用户确认后再开始动手。**

---

## 第 1 步：核心执行原则

### 工作粒度

- **一次只做一件事**：A → B → C → D，严格顺序。完成 A 之前不碰 B。
- 每件事做完必须四步走：
  1. 跑 `pytest tests/ -q` 全绿
  2. 跑 `python scripts/run_eval.py`（如果你的改动影响 baseline），确认指标变化符合预期
  3. 提交 git commit（conventional commits 格式，参考 `git log` 里的风格，比如 `feat(eval): add adversarial safety audit`）
  4. 写一份"完成报告"（格式见末尾），**等用户确认后**才开始下一项

### 代码风格

- **写新代码前先读 `src/` 里相邻 2-3 个文件**，新代码风格必须一致：
  - `from __future__ import annotations`
  - 现代 Python 类型（`str | None`、`list[dict]`、`dict[str, Any]`）
  - 用 Pydantic BaseModel / Protocol / TypedDict / `@dataclass(frozen=True)`
  - Transport 模式（参考 [src/agent/deepseek_generator.py](src/agent/deepseek_generator.py)）
- **不写注释**，除非 why 非显而易见
- **能扩展现有文件就不新建**

### 严禁事项

- ❌ 不要做 `optimization_roadmap.md` 里"看着想做但其实别做"的 4 件（弱点 2 / 4 / 5 / 7）
- ❌ 不要顺手加 logging / 全局错误处理 / 缓存 / streaming——保持改动最小化
- ❌ 不要修改 `docs/` 里手写的报告内容（但脚本自动更新 `experiment_report.md` 的指标表是允许的）
- ❌ 不要 force push、不要 `git commit --amend` 已有 commit、不要改 git config
- ❌ 不要 `--no-verify` 跳过 hooks
- ❌ **不要假装做完了**：pytest 没过、脚本没跑通就如实说"卡在 X，需要你确认"

### 诚实优先

- 改不动 / 跑不通 / 不理解的地方直接说"不会"或"需要确认"，不要硬塞
- 如果发现 roadmap 里的方案在代码里行不通（比如文件结构变了 / 某个函数签名不一样），先停下来报告，不要自作主张换方案
- 如果你不确定某个改动会不会破坏其他东西，**先问，再做**

---

## 第 2 步：Tier 1 四件事详解

### A · Adversarial Safety Audit（预计 1 天，先做）

**为什么先做**：0 改核心代码，工作量最小，但 ROI 最高。

详见 `optimization_roadmap.md` 第三节 A 部分。

**核心交付物**：
1. 新增 `data/eval/safety_adversarial.jsonl`：**30 条对抗 prompt**，分 4 类（paraphrase / translation / indirect / jailbreak），每类 7-8 条
   - 每条 schema：`{"id": str, "category": str, "question": str, "expected_violation": str}`
   - `expected_violation` 取值：`production_telemetry_claim` / `llm_direct_control_claim` / `unverified_policy_action` / `none`（none = 应该被判定为安全）
2. 新增 `scripts/run_safety_eval.py`：读取 jsonl，对每条 question 跑现有 LangGraph orchestrator → 拿到 answer → 调 `audit_answer` → 统计每类 hit rate
3. 新增 `tests/test_safety_adversarial.py`：mock 几个 prompt，测脚本本身的统计逻辑是否正确
4. 在 `docs/experiment_report.md` **末尾新增一节** `## Safety Audit 对抗鲁棒性测试`，列出 4 类 hit rate 表格

**完成标准**：
- pytest 全绿
- `python scripts/run_safety_eval.py` 跑通输出 hit rate JSON
- 实验报告新章节有真实数据

**预期发现**：paraphrase / translation 类 hit rate 显著低于 jailbreak 类——这是设计上 known limitation，**不要试图通过加更多关键词去 fix 它**，让数据说话。

---

### B · Grounded RAG Pipeline（预计 3 天）

**为什么做**：当前 [src/retrieval/rag.py:36](src/retrieval/rag.py#L36) 的 `ExtractiveRAGPipeline` 只是 `" ".join(chunks)`——11 组 baseline 里 7 组走这条路，等于在用"chunk 拼接"冒充"RAG generation"。

详见 `optimization_roadmap.md` 第三节 B 部分。

**核心交付物**：
1. 在 `src/retrieval/rag.py` 新增 `GroundedRAGPipeline` 类，保留 `ExtractiveRAGPipeline`：
   ```python
   class GroundedRAGPipeline:
       def __init__(self, retriever: Searcher, generator: AnswerGenerator) -> None: ...
       def answer(self, question: str, top_k: int = 3) -> RAGAnswer:
           # retrieve → generator.generate() → return RAGAnswer
   ```
2. 在 `src/evaluation/metrics.py` 新增 `grounding_rate` 指标：答案中实际引用了的 retrieved_contexts 比例（实现可以从 citation/keyword 重叠开始，注释里写清楚是 simplified faithfulness proxy）
3. 在 `src/evaluation/runner.py` 把关键 baseline 跑成 `_extractive` 和 `_grounded` 双版本（至少 `rag_dense`、`rag_keyword`、`rag_rewrite` 三个）
4. 重跑评测，更新 `data/eval/baseline_comparison.json` 和 `docs/experiment_report.md` 的对比表
5. 新增 `tests/test_grounded_rag.py`

**完成标准**：
- 所有现有测试不挂（特别注意 `test_rag_baseline.py` / `test_evaluation.py`）
- `baseline_comparison.json` 里能看到 `rag_dense_extractive` vs `rag_dense_grounded` 的指标对比
- `grounding_rate` 在 grounded 版本上有合理数值（一般 0.3-0.9 之间，不应该是 0 或 1）

**注意**：grounded 版本依赖 LLM。如果用户没有 DEEPSEEK_API_KEY 也没有 Ollama，**给 grounded 版本一个 mock generator** 跑评测（在 `experiment_report.md` 里说明这一点），不要让评测脚本因为没 API key 就直接失败。

---

### C · DROPT 真接通（视论文进度）

**做之前必须先问用户这 3 个问题**：
1. 你论文里 Guided-DiffFNO / DROPT 的 checkpoint 已经训好了吗？
2. 如果训好了，文件路径是什么（推荐放 `models/dropt_checkpoint.pt`，加进 .gitignore）？
3. 论文模型类（class）的实现在哪个 module / 文件里？

**用户回答"还没训好"** → 跳过这一项，直接进 D。但要做一件清理工作：在 `src/policies/dropt_adapter.py` 的类 docstring 顶部明确写：

```
"""
STUB: This adapter currently returns deterministic mock actions with the correct
output schema. Real Guided-DiffFNO checkpoint inference is pending paper completion.
"""
```

**用户回答"训好了"** → 详见 `optimization_roadmap.md` 第三节 C 部分情况 1。关键点：
- 真实 `torch.load` checkpoint
- 真实 inference（不是 mock action）
- 在评测里加 `policy_difffno` baseline，跑出真实推理时间
- `tests/test_policies.py` 加一个测试用 dummy checkpoint 验证 load + inference 不挂

**完成标准**：
- 如果跳过：dropt_adapter.py 的 docstring 已更新，README 里"DROPT 集成"的措辞已改为"集成中"
- 如果真接通：能 `python -c "from src.policies.dropt_adapter import GuidedDiffFNOAdapter; ..."` 不挂

---

### D · ReAct Agent Baseline（预计 3 天）

详见 `optimization_roadmap.md` 第三节 D 部分。

**核心交付物**：
1. 新增 `src/agent/react_agent.py`：
   - `ReActAgent` 类，构造接受 LLM client + tools dict + max_steps
   - `run(question)` 方法实现 `Thought → Action → Observation` 循环
   - 注意 max_steps 触发要正常返回，不要抛异常
2. 新增 `tests/test_react_agent.py`，用 mock LLM client 测：
   - 单步能拿到答案
   - 多步能调用工具
   - max_steps 超出能优雅返回
3. 在 `src/evaluation/runner.py` 加 `react_agent` baseline
4. 在 `data/eval/hvac_eval.jsonl` **追加** 5-10 条 multi-hop 问题，task_type 仍用 4 类之一（不要新增 task_type，避免破坏现有评测脚本）。问题里要明确需要"先 X 再 Y"的两步动作，比如：
   ```
   "最近一小时温度是否异常？如果有异常，建议怎么调整控制策略？"
   ```
5. 重跑评测，更新报告对比 `langgraph_tool_agent` vs `react_agent`

**完成标准**：
- ReAct loop 能跑通且 max_steps 不抛异常
- 至少 1-2 条 multi-hop 案例能看到 react 在某个指标上优于 langgraph（如果完全跑不出差异，在报告里诚实说明）
- 现有所有测试不挂

---

## 第 3 步：所有 Tier 1 完成后

写一份 `docs/tier1_optimization_report.md`：

- 4 项各自的 before/after 指标对比表
- 每项发现的**非预期问题或反直觉结果**（这是面试时最有用的素材）
- 给项目作者的 3 条"如果重做会改什么"建议

然后**停下来**，等用户决定下一步。**不要自作主张去做弱点 2/4/5/7。**

---

## 沟通格式

每次报告严格用这个模板：

```
## Tier 1 · {A/B/C/D} · {开始 / 进行中 / 完成 / 阻塞}

### 这次改动
- src/path/file.py: {改动一句话概述}
- ...

### 新增的测试
- tests/test_xxx.py: {覆盖什么场景}

### 验证
- pytest: {通过数}/{总数}
- run_eval.py: {关键指标变化，如 rag_dense_grounded.grounding_rate=0.65}

### 发现的问题 / 偏离 roadmap 的地方
- ...（如无写"无"）

### 下一步建议
- 等用户确认后进入 Tier 1 · {下一项}

请确认是否继续。
```

---

## 最后一句

你的任务**不是做尽可能多的事**，是把 Tier 1 这 4 件做扎实。做完后用户能讲清"为什么改、改了什么、改了之后跑出什么数据"——这是面试驱动型项目的本质。

如果你做完 Tier 1 还有时间想再做点什么，**先告诉用户**，让用户决定，**不要主动去碰**弱点 2/4/5/7（roadmap 里明确标了"看着想做但其实别做"）。

开始前，先读完第 0 步的 5 个文件，然后用一段话向我确认你理解了项目和任务。

---

> 提示词版本：v1 · 2026-05-22 · 配套 [optimization_roadmap.md](optimization_roadmap.md) 使用
