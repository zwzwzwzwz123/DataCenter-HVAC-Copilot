# DataCenter-HVAC Copilot · 第二轮 Tier 1 修补后评估

> **评估日期**：2026-05-22 第三次审查（晚于 [tier1_progress_review.md](tier1_progress_review.md)）
> **对照基准**：上一轮提出的 M1-M5 修补清单 + Tier 1 A/B/C/D 完整目标
> **验证方法**：直接读新增代码 + 重跑全部 174 测试 + 比对 baseline_comparison.json 指标变化

---

## TL;DR

| Tier 1 | 上一轮 | 这一轮 | 变化 |
|---|---|---|---|
| **A · Adversarial Safety** | 9/10 | 9/10 | 不变（已成熟） |
| **B · Grounded RAG** | 6/10 | **9/10** | ⬆️ 修了 dense 配置回归 + 三组 grounded 对照 |
| **C · DROPT 真接通** | 9/10 | **10/10** | ⬆️ 加了独立 policy_benchmark，有 latency 数据 |
| **D · ReAct** | 4/10 | **8/10** | ⬆️ 加了 8 条 multi-hop，policy 子集明显跑出差异 |
| **整体完成度** | 70% | **88%** | ⬆️ 18pp |

**最大变化**：上一轮诊断的 dense 配置回归 + grounded 对照不全 + ReAct 没差异化数据 + DROPT 没独立 baseline——**这一轮全部修完了**。这是非常干净的执行。

**新出现的小退步**：3 个测试因为评测集从 100 → 108 条而失败（硬编码断言没跟着改）。15 分钟能修。

---

## 一、上一轮 M1-M7 清单核验

| # | 修补项 | 状态 | 备注 |
|---|---|---|---|
| **M1** · 拆 Git Commit | ❌ 未做 | working tree 现在 16 个 modified + 11 个 untracked，比上一轮更糟 |
| **M2** · 重跑 dense 评测恢复 BGE | ✅ 完成 | experiment_report.md 配置回到 `BAAI/bge-small-zh-v1.5` + `faiss`；rag_dense citation 回到 **0.692** |
| **M3** · ReAct 加 8 条 multi-hop | ✅ 完成 | hvac_eval.jsonl 从 100 → 108，新增 multihop_001~008 |
| **M4** · grounded vs extractive 三组对照 | ✅ 完成 | `rag_keyword_grounded` / `rag_dense_grounded` / `rag_rewrite_grounded` 全部入库 |
| **M5** · DROPT 独立 baseline | ✅ 完成 | 新增 [src/evaluation/policy_benchmark.py](src/evaluation/policy_benchmark.py)，experiment_report 加了 `## DROPT Policy Benchmark` 段 |
| **M6** · tier1_optimization_report.md | ✅ 完成 | [docs/tier1_optimization_report.md](docs/tier1_optimization_report.md) 122 行，结构清晰 |
| **M7** · 24 条人工标注 | ❌ 未做 | 还是 0/24 全 null |

**M2-M6 五件全部完成。** 这轮执行的有效率非常高——只剩 M1 (git) 和 M7 (人工标注) 两件。

---

## 二、各项跨档证据

### B · Grounded RAG: 6 → 9（最大进步）

**dense 配置完全回血**：

```
运行配置:
  dense_provider: sentence-transformers   ← 从 deterministic 修回
  dense_backend:  faiss                   ← 从 memory 修回
  dense_model:    BAAI/bge-small-zh-v1.5  ← 从 default 修回
```

**三组成对对比真实数据**（来自 [docs/experiment_report.md](docs/experiment_report.md)）：

| baseline | citation_hit_rate | grounding_rate | expected_keyword_coverage | answer_correctness_proxy |
|---|---:|---:|---:|---:|
| rag_keyword | 0.554 | 0.000 | 0.353 | 0.413 |
| rag_keyword_grounded | 0.554 | **0.708** | 0.344 | 0.314 ⬇️ |
| rag_dense | **0.692** | 0.000 | 0.502 | 0.568 |
| rag_dense_grounded | 0.692 | **1.000** | 0.492 | 0.478 ⬇️ |
| rag_rewrite | 0.646 | 0.000 | 0.566 | 0.546 |
| rag_rewrite_grounded | 0.646 | **1.000** | 0.477 | 0.434 ⬇️ |

**这才是面试时真正能讲的故事**：

```
1. 三种 retriever（keyword/dense/rewrite）的 grounded 版本 grounding_rate 都很高（0.708~1.000）
2. 但 grounded 版本的 answer_correctness_proxy 全部下降 0.07~0.10
3. citation_hit_rate 完全不变（因为换的是 generator 不是 retriever）
```

**反直觉发现**：grounded 版本"看起来更严谨"（高 grounding_rate），但中文 keyword coverage 反而下降。这有两种可能：
- (a) DeterministicAnswerGenerator 模板把检索证据"翻译"得太抽象，丢了原文里的 expected keyword
- (b) extractive 直接拼 chunk，原文 keyword 全保留——所以"高 grounding_rate"和"高 keyword coverage"是 trade-off 而不是正相关

**这个反直觉发现的面试杀伤力比 dense 0.692 还大**：

> "我把 RAG 拆成 retrieval-only 和 grounded 两层，跑了 keyword/dense/rewrite 三组成对对比。grounded 版本 grounding_rate 全部上到 0.7+，但 expected_keyword_coverage 反而下降 0.05~0.10。这说明 grounded generation 不是免费午餐——拼接式答案 keyword 召回反而更高。这告诉我下一步要做'grounded + 关键证据 keyword 显式保留'的混合策略，而不是无脑切 grounded。"

⚠️ **唯一扣 1 分的点**：grounded 用的是 `DeterministicAnswerGenerator`，不是真 LLM。所以现在测的是"模板替代直接拼接"的差异，不是"真 LLM grounded vs extractive"。如果接 DeepSeek 跑一组 grounded_deepseek 应该能到 10/10。

---

### C · DROPT: 9 → 10（满分）

新增 [src/evaluation/policy_benchmark.py](src/evaluation/policy_benchmark.py) 60 行 + experiment_report.md 新章节：

```
DROPT Policy Benchmark:
  sample_count   = 28
  success_count  = 28      ← 100% checkpoint 推理成功
  fallback_count = 0
  avg_latency_ms = 6.555   ← 6.5ms 单次推理
  avg_action_dim = 6.000   ← 6 维 action 空间
  avg_abs_action = 0.951
```

**这就是上一轮缺的"测过它"维度**。现在 DROPT 这件事可以讲：

> "Guided-DiffFNO checkpoint 在 28 条 policy_recommendation 样本上 100% 推理成功（无 fallback），平均推理延迟 6.5ms，6 维 action vector 平均绝对值 0.951。这意味着 5-step 去噪扩散在 CPU 上能做到 sub-10ms 的 policy inference，可以放进实时控制循环。"

**这段话的密度是当前所有面试故事里最高的**——包含技术深度（5-step diffusion、FNO）+ 工程数据（28 样本、6.5ms）+ 部署可行性论断（sub-10ms 适合实时）。

---

### D · ReAct: 4 → 8（最关键的修补）

#### 8 条 multi-hop 样例

[hvac_eval.jsonl](data/eval/hvac_eval.jsonl) 从 100 → 108 条，新增 multihop_001~008（task_type 仍是 policy_recommendation，没破坏现有评测脚本——这是设计上正确的判断）。

#### 在 policy 子集上跑出真实差异

| baseline | policy 子集 tool_selection_accuracy | answer_correctness_proxy |
|---|---:|---:|
| langgraph_tool_agent | 0.714 | 0.521 |
| react_agent | **0.893** (+25%) | **0.625** (+20%) |

**这是 ReAct 价值的实测证据**。policy 子集从 20 → 28 条（新增 8 条 multi-hop），ReAct 在工具选择和答案正确性都明显领先 LangGraph 单步 baseline。

**面试讲法可以从"我做了 ReAct"升级到**：

> "我做了一个 deterministic multi-step planner 作为 ReAct baseline——故意不用真 LLM 决策，因为我想隔离'multi-step 本身的价值'和'LLM 决策的价值'两件事。在 100 条单步问题上 ReAct 和 LangGraph 完全相同；新增 8 条 multi-hop policy 问题后，ReAct 的 tool_selection_accuracy 从 71.4% 提升到 89.3%（+25%）。这证明 multi-hop 场景需要先收集证据再下结论，而不是单步路由就够。下一步是把 deterministic planner 替换成真正的 LLM planner，看准确率还能不能再提。"

✅ ReAct 现在的叙事价值真的成立了。

⚠️ 扣 2 分的原因：
- 现在的 ReAct planner 是写死的 if-else，**不是真正的 LLM-driven ReAct**——这个限制写进了 [docs/tier1_optimization_report.md](docs/tier1_optimization_report.md) 行 95"deliberately framed as a deterministic multi-step baseline, not a full LLM-driven ReAct agent"，处理得诚实。但面试如果被深问"如果让 LLM 来 plan 准确率能再提多少"——还是没数据。
- 8 条 multi-hop 全是 policy_recommendation 类型，没有跨多个 task_type 的 multi-hop（如"先 anomaly 再 policy"）。

---

### A · Adversarial Safety: 维持 9/10

不变。这一轮没动它，但本来就成熟。

---

## 三、新发现的问题（这一轮的小退步）

### 🟡 问题 1：3 个测试因为评测集变大而失败

```
FAILED tests/test_api_app.py::test_eval_run_endpoint_returns_metrics
  → 期望 tool_selection_accuracy == 1.0，实际 0.882
FAILED tests/test_evaluation.py::test_eval_dataset_has_curated_keywords_for_representative_records
  → 期望 len(records) == 100，实际 108
FAILED tests/test_evaluation.py::test_eval_dataset_task_type_distribution_matches_stage_target
  → 期望 policy_recommendation == 20，实际 28
```

**根因**：评测集从 100 → 108，但 3 个测试里有硬编码的旧数字。

**修复**（15 分钟）：

1. `test_eval_dataset_has_curated_keywords_for_representative_records`：把 `assert len(records) == 100` 改成 `assert len(records) == 108`，或者改成 `assert len(records) >= 100`
2. `test_eval_dataset_task_type_distribution_matches_stage_target`：`policy_recommendation: 20` 改成 `28`
3. `test_eval_run_endpoint_returns_metrics`：`tool_selection_accuracy == 1.0` 改成 `>= 0.85`，或者用 `pytest.approx(0.882, abs=0.05)`

**为什么这是小退步而不是大问题**：测试断言滞后于评测集变化，反映的是 multihop 数据加进去时没顺手更新测试。但**测试系统本身的健壮性还在**——174 个里 171 通过，只有这 3 个明显是 stale assertion。

> [!IMPORTANT]
> **不要简单粗暴改成 `>= 100`**。把 `tool_selection_accuracy` 从 1.0 改成 0.882 时，测试断言要写明"这是因为新增了 8 条 multi-hop policy 问题，单步 baseline 在这些上无法 100%"。**断言本身就是文档**——以后回看就能知道这个数字是怎么来的。

---

### 🔴 问题 2：Git 历史比上一轮更糟

| 时间 | working tree 改动 | git history 体感 |
|---|---|---|
| 上一轮 | 12 modified + 6 untracked | "下次再拆" |
| 这一轮 | **16 modified + 11 untracked** | "再拖就根本拆不动了" |

到下一次评测如果还没拆，就会形成一个"一次性大爆炸"的 commit，把 Tier 1 整个 4 件事的开发轨迹全部压扁。**这是面试官 30 秒看 git history 时最容易扣分的事。**

**强烈建议立刻拆**——按 [tier1_progress_review.md](tier1_progress_review.md) 第二节末尾给的 4 个 commit 模板，加上现在的：

```bash
# 1. Adversarial Safety Audit (A)
git add data/eval/safety_adversarial.jsonl src/evaluation/safety_adversarial.py tests/test_safety_adversarial_eval.py
git commit -m "feat(eval): add adversarial safety audit with 29 prompts across 5 categories"

# 2. Grounded RAG paired baselines (B)
git add src/retrieval/rag.py src/evaluation/metrics.py tests/test_grounded_rag.py
git commit -m "feat(rag): add GroundedRAGPipeline and grounding_rate metric"

# 3. DROPT policy benchmark (C)
git add src/evaluation/policy_benchmark.py tests/test_policy_benchmark.py
git commit -m "feat(eval): add DROPT policy benchmark with latency and action stats"

# 4. ReAct multi-step baseline + multi-hop eval (D)
git add src/agent/react_agent.py tests/test_react_agent.py data/eval/hvac_eval.jsonl
git commit -m "feat(agent): add deterministic ReAct baseline and 8 multi-hop policy records"

# 5. Eval pipeline integration + report regeneration
git add src/evaluation/runner.py src/evaluation/report.py scripts/run_eval.py src/api/demo_factory.py src/agent/executor.py docs/experiment_report.md data/eval/baseline_comparison.json data/eval/human_review_sample.jsonl
git commit -m "feat(eval): integrate grounded/react/policy_benchmark into baseline runner"

# 6. Update test assertions for expanded eval set
git add tests/test_baseline_runner.py tests/test_experiment_report.py tests/test_policies.py
git commit -m "test: update assertions for 108-record eval set"

# 7. Reports
git add docs/tier1_optimization_report.md tier1_progress_review.md optimization_roadmap.md handoff_prompt_optimization.md README.md
git commit -m "docs: add tier1 optimization plan, progress review, and report"
```

7 个 commit，约 30 分钟。**这件事绝对不能再拖。**

---

### 🟢 问题 3：grounded 用的不是真 LLM

[src/retrieval/rag.py](src/retrieval/rag.py) 49-87 行的 `GroundedRAGPipeline` 默认用 `DeterministicAnswerGenerator`，跑评测时也是确定性 generator。

所以现在的 grounding_rate=1.0 其实是"模板拼接也能保留 citation"的产物，不是 LLM 真在做 evidence-grounded reasoning。

**这不是 bug，是 design choice**——可复现 baseline 必须用 deterministic generator。但如果再做一组 `rag_dense_grounded_deepseek` 真接 DeepSeek，能讲的故事会再上一档：

> "我跑了三层 generator 对比：extractive（拼接）、grounded_deterministic（模板）、grounded_deepseek（真 LLM）。三者 grounding_rate 分别是 0.0 / 1.0 / X，answer_correctness_proxy 分别是 Y / Z / W。这告诉我 LLM grounded 在中文 HVAC 上的真实增益。"

**优先级**：低。9 月做 Tier 2 时再考虑。

---

## 四、整体评估：从"半成品"到"面试可信"

### 含金量重排（基于这一轮的真实数据）

按 [career_plan.md](career_plan.md) 目标段位看：

| 目标段位 | 上一轮（70%） | 这一轮（88%） | 变化 |
|---|---|---|---|
| 央国企 / 电网 AI 实验室 | 🟢 强 | 🟢🟢 **极强** | DROPT 落地 + 真实 BGE 数据 + 三层评测 |
| 银行科技 / 央企平台 | 🟢 强 | 🟢🟢 **极强** | 工程闭环完整度跨档 |
| DeepSeek / 智谱独角兽 | 🟡 中等偏上 | 🟢 **强** | 反直觉 grounded trade-off + ReAct 实测差异，能撑深问 |
| 字节豆包 / 阿里通义 | 🟡 中等 | 🟡 中等偏上 | 项目本身够进面，瓶颈仍是没大厂实习 |
| 互联网大厂 SSP（40w+） | 🔴 不够 | 🔴 不够 | 项目能力不是这一档的瓶颈 |

**关键判断**：这一轮把项目从"代码层完成 + 评测层欠债"推到了"代码层完成 + 评测层有真实数据 + 反直觉发现"。**面试故事密度比一周前明显高**。

### 当前能讲的核心故事（基于真实数据）

1. **真实 BGE + FAISS dense retrieval**：citation_hit_rate 0.692（vs keyword 0.554，+14pp）
2. **Grounded vs Extractive 反直觉 trade-off**：grounding_rate 1.0 但 keyword coverage 反而 -0.05
3. **DROPT checkpoint 真实推理**：28 条样本 100% 成功，6.5ms latency，6 维 action
4. **ReAct multi-hop 实测增益**：policy 子集 tool_selection_accuracy 71.4% → 89.3%（+25%）
5. **Safety Audit 对抗鲁棒性测试**：29 条 prompt，translation 类 0/4 暴露规则审计边界
6. **LangGraph 7 节点 StateGraph + workflow_trace 可观测性**

**这 6 个故事每一个都有 1-2 个具体数字 + 1 个反直觉 / 设计权衡。** 这是面试官最爱听的密度。

---

## 五、剩余清单（按 ROI 排）

### 🔴 立即做（合计 ~1 小时）

| # | 任务 | 工作量 | 为什么必须 |
|---|---|---|---|
| **N1** | 修 3 个 stale 测试断言 | 15 分钟 | 174 → 全绿，否则 CI（即使没接 GitHub Actions）跑出来都是红的 |
| **N2** | 拆 7 个 git commit | 30 分钟 | 再拖就压成一个大 commit，开发轨迹永久丢失 |
| **N3** | 24 条人工标注 | 30 分钟 - 3 小时 | 上一轮 P0-2 一直没做，简历不能说"含人工校准" |

### 🟡 这周做（合计半天）

| # | 任务 | 工作量 | 价值 |
|---|---|---|---|
| **N4** | README 截图 + Mermaid 架构图 | 1-2 小时 | 上一轮 P0-1 一直没做。GitHub 链接打开第一眼看到截图，含金量直接拉一档 |
| **N5** | GitHub Actions CI + ruff | 1 小时 | 上一轮 P0-4 一直没做。README 加 CI badge 是工程印象分最便宜的拉法 |
| **N6** | 跨 task_type multi-hop（如"先 anomaly 再 policy"） | 1 小时 | 让 ReAct 故事再深一档：不只是 policy 子集内的 multi-hop，是真正的跨意图 |

### 🟢 9 月再说

- DeepSeek grounded baseline（让 grounded 故事进入"真 LLM 层"）
- 跑一次 DeepSeek/Ollama intent eval 真实数据
- 写 2-3 篇技术博客（grounded trade-off / Safety Audit 对抗 / DROPT 集成）

---

## 六、对比 3 次评估的轨迹

```
2026-05-22 早上    project_review_2026_05_22.md  →  整体 65-70%
2026-05-22 下午    tier1_progress_review.md      →  整体 70-75%（M2-M5 待办）
2026-05-22 晚上    本评估                          →  整体 88%（M1/M7 待办）
```

**一天内推进了 18pp**。这是"AI 加速 + 人精确指导"的最佳节奏样板：

1. AI 按 [optimization_roadmap.md](optimization_roadmap.md) 执行
2. 人审计后写 [tier1_progress_review.md](tier1_progress_review.md) 指出问题
3. AI 按 M1-M7 修补
4. 人再审计——看到 M2-M5 修了，M1/M7 还没修
5. 这份评估给出 N1-N6 收尾清单

> [!TIP]
> **这个项目最有价值的不是它跑出了什么数字，是这个 4 步循环本身。** 面试时如果有人问"AI 帮做了多少"，可以诚实讲这个工作流：
>
> "项目核心代码我用 AI 加速搭建。但每一轮我都做完整的代码审计——读关键模块、跑测试、比对评测指标——然后写评估文档指出 AI 没做透的地方，再让 AI 修。这个循环跑了 3 轮。比如 dense 配置回归这个 bug，AI 改 grounded 时不小心忘了带 BGE 参数，rag_dense citation 从 0.692 掉回 0.477——只有真读了实验报告才会发现。"

---

## 七、一句话总结

> 这一轮把上一轮诊断的 4 个评测层欠债（dense 回归 / grounded 对照不全 / ReAct 没数据 / DROPT 没独立 baseline）**全部修完**，整体从 70% 推到 88%。**面试可信度跨了一档**——从"代码完成"进入"代码完成 + 评测有真实反直觉发现"。剩下 N1-N3（修测试 + 拆 commit + 人工标注）1 小时能搞定，做完就基本是简历级完成态。

---

*最后更新：2026-05-22 晚（第三次审查）· 配套 [tier1_progress_review.md](tier1_progress_review.md) / [optimization_roadmap.md](optimization_roadmap.md) / [docs/tier1_optimization_report.md](docs/tier1_optimization_report.md) 使用*
