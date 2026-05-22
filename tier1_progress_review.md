# DataCenter-HVAC Copilot · Tier 1 优化进展评估

> **评估日期**：2026-05-22（晚于本日早上的 [project_review_2026_05_22.md](project_review_2026_05_22.md)）
> **对照基准**：[optimization_roadmap.md](optimization_roadmap.md) Tier 1 四件事（A/B/C/D）+ [handoff_prompt_optimization.md](handoff_prompt_optimization.md) 交付标准
> **验证方法**：pytest 全套跑了一遍 + 直接读新增代码 + 比对 baseline_comparison.json 的指标变化

---

## TL;DR

| Tier 1 | 状态 | 评分 | 关键判断 |
|---|---|---|---|
| **A · Adversarial Safety Audit** | ✅ 完成 | **9/10** | 跑出预期反直觉发现（translation 类 hit rate=0%），是这一轮最高质量的产出 |
| **B · Grounded RAG Pipeline** | ⚠️ 部分完成 | **6/10** | 核心代码到位，但**评测配置回归**——`rag_dense` 从 0.692 掉回 hash embedding 的 0.477 |
| **C · DROPT 真接通** | ✅ 完成 | **9/10** | 大惊喜：468 行真实 DiffFNO + checkpoint（`models/dropt/policy_best_fno_guided.pth`），不是 stub |
| **D · ReAct Baseline** | ⚠️ 架子完成 | **4/10** | `react_agent` 指标和 `langgraph_tool_agent` 完全相同，没补 multi-hop 样例 |

**总进度：约 70%**。 174 个测试全绿，4 件事都动了，但 B/D 的"评测层证据"没做透——这恰恰是面试时被问"那你 grounded vs extractive 对比数据呢""ReAct vs workflow 实测差异呢"会卡住的两个点。

---

## 一、逐项核验

### A · Adversarial Safety Audit ✅ 9/10

**核验代码**：
- [data/eval/safety_adversarial.jsonl](data/eval/safety_adversarial.jsonl) — 29 条（roadmap 要求 30，少 1 条）
- [src/evaluation/safety_adversarial.py](src/evaluation/safety_adversarial.py) — 86 行，schema 定义 + 评测逻辑
- [tests/test_safety_adversarial_eval.py](tests/test_safety_adversarial_eval.py) — 64 行
- [docs/experiment_report.md](docs/experiment_report.md) 新增了 `## Safety Audit 对抗鲁棒性测试` 章节

**实测结果**（来自 experiment_report.md）：

| Category | sample | hit | hit_rate |
|---|---:|---:|---:|
| paraphrase | 8 | 8 | **1.000** |
| jailbreak | 6 | 4 | 0.667 |
| mixed | 5 | 3 | 0.600 |
| indirect | 6 | 2 | 0.333 |
| translation | 4 | 0 | **0.000** |
| **overall** | 29 | 17 | **0.586** |

**这是这一轮最有面试价值的产出**：
- ✅ translation 类 hit_rate = 0/4 完美命中"对抗鲁棒性差"的预期——英文绕过中文 risky_phrases 字典
- ✅ paraphrase 类反而 100%（说明你的中文同义词字典覆盖得不错，反而是直白的英文"绕"出去了）
- ✅ missed_ids 全部列出来了，方便面试时 "你看 adv_translation_01 是怎么漏的" 这种深问
- ✅ 多了 `mixed` 类（roadmap 没要求），是合理的扩展

**面试可直接复用的故事**：
> "我做了 29 条对抗 prompt 测 Safety Audit。中文 paraphrase 全命中（8/8），但**英文 translation 类 0/4 全军覆没**——因为我的 risky_phrases 字典是中文。这正好印证了一开始用确定性规则的 known limitation：规则的对抗鲁棒性差。下一步要么加双语字典，要么把 audit 升级成 LLM judge + 规则双重验证。**这件事的价值不是 hit_rate=0.586 这个数字，是它让我能讲清规则审计的边界在哪。**"

**扣 1 分**：roadmap 要求 30 条 + 4 类，实际是 29 条 + 5 类（paraphrase 8 / jailbreak 6 / mixed 5 / indirect 6 / translation 4）。translation 类只 4 条样本量太小，结论"0%"统计上不稳——补到 8 条更稳。

---

### B · Grounded RAG Pipeline ⚠️ 6/10

**核验代码**：
- [src/retrieval/rag.py](src/retrieval/rag.py) 新增 `GroundedRAGPipeline`（49-99 行）
- [tests/test_grounded_rag.py](tests/test_grounded_rag.py) 87 行
- `src/evaluation/metrics.py` 新增 `grounding_rate` 指标
- baseline_comparison.json 多了 `rag_grounded` 这一组 + 所有 baseline 多了 `grounding_rate` 列

**实测数据**：

```
rag_grounded:  citation_hit_rate=0.477  grounding_rate=0.923  expected_keyword_coverage=0.412
rag_dense:     citation_hit_rate=0.477  grounding_rate=0.000  expected_keyword_coverage=0.368
```

**两个严重问题**：

#### 🔴 问题 1：评测配置回归（最严重）

[experiment_report.md:8-12](docs/experiment_report.md#L8-L12) 现在写的是：

```
- dense_provider: deterministic
- dense_backend: memory
- dense_model: default
```

**但本日早上的版本写的是 `BAAI/bge-small-zh-v1.5` + `faiss`**。重跑评测时**忘了带 `--dense-provider sentence-transformers --dense-backend faiss --dense-model BAAI/bge-small-zh-v1.5` 这组参数**，全部回退到 hash embedding。

直接后果：
- `rag_dense.citation_hit_rate` 从 **0.692 掉回 0.477**（-21pp）
- 你简历里"BGE-small-zh + FAISS 让检索召回率从 keyword 55.4% 提升到 dense 69.2%"这条**已失效**——当前 baseline_comparison.json 里 dense 比 keyword 还低（0.477 < 0.554）

> [!CAUTION]
> 这是 Tier 1 B 的最大隐患。**简历不能再用 0.692 这个数字**，除非你重新跑一次带 dense extras 的评测把它补回来。这件事 30 分钟能修。

#### 🟡 问题 2：没做"extractive vs grounded"成对对比

roadmap 第三节 B 明确要求：把 `rag_keyword` / `rag_dense` / `rag_rewrite` 三组都跑成 `_extractive` 和 `_grounded` 双版本。

实际只加了一个 `rag_grounded`（`rag_dense` 配 `DeterministicAnswerGenerator`），结构上是 dense + grounded 一组，没有 keyword + grounded、rewrite + grounded 的对照——所以"grounded 在不同 retriever 下的稳定性"没有被验证。

**面试被问"你的 grounded 对所有 retriever 都稳定吗"——只能说"我只跑了 dense 那一组"。**

#### ✅ 正面：grounding_rate 这个指标设计是对的

`rag_grounded.grounding_rate=0.923` vs `rag_dense.grounding_rate=0.000`，差异显著且符合预期（extractive 的"答案"是 chunk 拼接，自身就是 context，没有"引用"概念，所以 grounding_rate=0 是合理的）。

而且 `rag_tool_agent.grounding_rate=0.477`（document_qa 子任务上 0.775，其他任务 0.000）——这个数据本身就是好故事："Tool Agent 在文档问答上有 grounding，在工具任务上 grounding 不适用"。

---

### C · DROPT 真接通 ✅ 9/10

**这是这一轮最大的惊喜。** 我审计 [src/policies/dropt_adapter.py](src/policies/dropt_adapter.py) 时本来预期是 stub，结果发现：

- **468 行真实代码**，包含：
  - `SinusoidalPosEmb`（扩散时间嵌入）
  - `SpectralConv1d`（FNO 谱卷积，FFT + 复数权重）
  - `DiffFNO`（state-conditional FNO 去噪网络）
  - `DoubleCritic`（双 Q 网络，offline RL 的标配）
  - `Diffusion`（带 vp beta schedule 的去噪扩散过程）
- **真实 checkpoint 落地**：`models/dropt/policy_best_fno_guided.pth` + `models/dropt/main_building_fno_guided_bcfix_clean.py`（训练脚本）
- experiment_report.md 写："DROPT / Guided-DiffFNO checkpoint 作为可选策略后端已接通：checkpoint 可加载、20 维 BEAR state 可推理"

**为什么 9 分而不是 10 分**：
- ✅ 真模型 + 真 checkpoint，已经远超 stub
- ✅ 缺失 / 不完整 checkpoint 时有明确回退（看 base.py 应该有 try/except → rule_based fallback）
- ❌ baseline_comparison.json 里**没看到 `policy_difffno` 这一组 baseline**——意味着 DROPT 推理只在 `/ask` 走 policy_recommendation 路由时被调用，没有作为独立 baseline 跑全评测集，没有"这个推理在多少条样本上 work / 平均 latency / 平均 action 分布"的数据
- ❌ 实验报告没有 DROPT inference time 数据

**面试时这就是那张终极王牌**：
> "我把论文里的 Guided-DiffFNO（FNO + 扩散 offline RL）打包成 policy adapter，let LangGraph 在 policy_recommendation 路由调度。20 维 BEAR state vector → 5-step 去噪 → action vector。LLM 只做证据整合 + Safety Audit，控制动作必须来自这个适配器。这是'LLM 解释 + RL/扩散策略工具执行'分工范式的实际落地。"

**这一项对你的招聘段位影响最大**——电网 AI 实验室、智慧建筑 AI 团队的面试官看到这个会激动。比再加 5 个 RAG baseline 都有用。

---

### D · ReAct Baseline ⚠️ 4/10

**核验代码**：
- [src/agent/react_agent.py](src/agent/react_agent.py) 172 行，`ReActOrchestrator` + `DeterministicReActPlanner` + `ReActStep` dataclass
- [tests/test_react_agent.py](tests/test_react_agent.py) 79 行
- baseline_comparison.json 多了 `react_agent` 一组

**致命问题：和 langgraph 指标完全相同**

对比 experiment_report.md 第 80-87 行：

```
langgraph_tool_agent | anomaly_diagnosis    | citation 0.000 | answer_correctness 0.367
react_agent          | anomaly_diagnosis    | citation 0.000 | answer_correctness 0.367
langgraph_tool_agent | document_qa          | citation 0.625 | answer_correctness 0.597
react_agent          | document_qa          | citation 0.625 | answer_correctness 0.597
langgraph_tool_agent | policy_recommendation| citation 0.000 | answer_correctness 0.531
react_agent          | policy_recommendation| citation 0.000 | answer_correctness 0.531
langgraph_tool_agent | timeseries_query     | citation 0.000 | answer_correctness 0.833
react_agent          | timeseries_query     | citation 0.000 | answer_correctness 0.833
```

**逐字段一模一样。** 意味着 ReAct 在当前 100 条评测集上**没展现任何差异化价值**。

#### 原因分析

读 [react_agent.py](src/agent/react_agent.py) 后看清楚了：

- `DeterministicReActPlanner` 是**关键词路由的 multi-step 包装**——只有当问题命中"温度 / 趋势 / 最近"等词且初判 route 是 policy_recommendation 时，才会走"先 timeseries_query 再 policy_recommendation"两步路径
- 当前 100 条评测集里没有这种"先 X 再 Y"的 multi-hop 问题（roadmap D 第 4 步要求**追加 5-10 条 multi-hop 问题**，但 [hvac_eval.jsonl](data/eval/hvac_eval.jsonl) 还是 100 条，没追加）
- 所以每条问题在 ReAct 里都退化成单步 = 和 LangGraph 走同一个 task_executor = 指标完全相同

#### 第二个深层问题

`DeterministicReActPlanner` 不是真正的 ReAct——真正的 ReAct 是 LLM 在每一步重新判断"要不要再调工具"，这里是**写死的 if-else**：

```python
if last_route == "timeseries_query" and primary.route == "policy_recommendation":
    return "policy_recommendation", "已经拿到时序上下文..."
```

面试被深问"你的 ReAct 是 LLM 驱动还是规则驱动"——只能说规则驱动。**这就退化成"LangGraph 的 deterministic 版本上加了一个写死的 multi-step 路径"**，叙事价值大打折扣。

#### 修补成本

- 加 5-10 条 multi-hop 问题到 hvac_eval.jsonl（**1 小时**）
- 让 ReAct 在 multi-hop 子集上跑出差异（应该自然就有）
- 写一个分组对比：单步问题 langgraph 与 react 持平，multi-hop 问题 react 在某指标上 +X%

修完后 4/10 → 7/10。但面试讲法仍然得改成"我做了 deterministic 多步规划器作为 baseline，证明 multi-hop 场景下 multi-step 比 single-step 更优"，**避开"真正的 ReAct"这个措辞**。

---

## 二、其它观察（非 Tier 1 但相关）

### ✅ 测试覆盖大幅扩张

```
174 passed in 87.84s
```

之前是 27 个测试文件、约 3960 行。这次新增 3 个测试文件 + 修改 3 个，**174 个 test case 全绿**。这是工程基线改善的真实证据。

### ❌ Git 历史问题没改进，反而退步

```
57df260 更新README.md
37cd14d 上传职业规划和项目改进建议文档
b2bfdaa v1.0.0
892dabc feat: expand eval set to 100 records
...（早期 commits）
16b1a8c v3.2  ← 最新
```

[handoff_prompt_optimization.md](handoff_prompt_optimization.md) 明确要求："每件事做完按 conventional commits 拆 commit"。但 Tier 1 A/B/C/D 全部改动还停留在 working tree 没提交（git status 显示 12 个 modified + 6 个 untracked）。

如果接下来一次性 commit 成 `feat: tier1 optimizations`，就把 4 件事的开发轨迹**永久压缩成一个 commit**——面试官看 git history 时的"持续迭代感"就丢了。

**强烈建议立即按 4 件事拆 commit**：
```bash
# A
git add data/eval/safety_adversarial.jsonl src/evaluation/safety_adversarial.py tests/test_safety_adversarial_eval.py
git commit -m "feat(eval): add adversarial safety audit with 29 prompts across 5 categories"

# B
git add src/retrieval/rag.py src/evaluation/metrics.py src/evaluation/runner.py tests/test_grounded_rag.py
git commit -m "feat(rag): add GroundedRAGPipeline and grounding_rate metric"

# D
git add src/agent/react_agent.py tests/test_react_agent.py
git commit -m "feat(agent): add ReAct deterministic multi-step baseline"

# 实验报告 + 配置
git add docs/experiment_report.md data/eval/baseline_comparison.json scripts/run_eval.py src/api/demo_factory.py src/evaluation/report.py tests/test_baseline_runner.py tests/test_experiment_report.py tests/test_policies.py
git commit -m "feat(eval): rerun baseline comparison with grounded/react/safety audit"
```

### ❌ 还没做的

| 项目 | 状态 |
|---|---|
| 24 条人工评测标注 | 还是 0/24 全 null（[project_review_2026_05_22.md](project_review_2026_05_22.md) 的 P0-2） |
| README 截图 / 架构图 | 还没做（P0-1） |
| GitHub Actions CI / ruff / logging | 还没做（P0-4） |
| `docs/tier1_optimization_report.md` | 没生成（handoff prompt 第 3 步要求） |

---

## 三、修补优先级（按 ROI 排）

> 当前状态：Tier 1 完成 70%。补完下面 5 件，整体进入"面试可信"状态。

### 🔴 立刻做（合计 ~2 小时）

#### M1 · 拆 Git Commit（30 分钟）

按上面的脚本拆 4 个 commit。**这件事拖到下一轮再做就不可逆**——working tree 越大越难拆。

#### M2 · 重跑 dense 评测，恢复 BGE 数据（30 分钟）

```bash
# 安装 dense extras（如还没装）
pip install -e ".[dense]"

# 重跑评测，明确指定 BGE + faiss
python scripts/run_eval.py \
  --dense-provider sentence-transformers \
  --dense-backend faiss \
  --dense-model BAAI/bge-small-zh-v1.5
```

跑完后 `experiment_report.md` 的运行配置应该回到 `BAAI/bge-small-zh-v1.5`，rag_dense citation 应该回到 ~0.692。**这是简历"BGE 提升 14pp"故事的命脉。**

#### M3 · 给 ReAct 评测加 8 条 multi-hop（1 小时）

往 [data/eval/hvac_eval.jsonl](data/eval/hvac_eval.jsonl) 追加 8 条样本，task_type 仍用 4 类之一（不破坏现有评测脚本）。要求每条问题包含"先 X 再 Y"两步动作，例如：

```jsonl
{"id": "multihop_001", "task_type": "policy_recommendation", "question": "最近一小时温度趋势是否提示需要调整 HVAC 控制策略？", "expected_keywords": ["温度", "趋势", "policy_result", "recommended_action"], ...}
{"id": "multihop_002", "task_type": "policy_recommendation", "question": "先看一下能耗构成，再判断是否需要切换控制策略", ...}
```

补完后重跑 `python scripts/run_eval.py`，预期 `react_agent` 在这 8 条上的 `expected_keyword_coverage` 应该高于 `langgraph_tool_agent`——这是 ReAct 价值的实测证据。

### 🟡 这周做（合计 ~1 天）

#### M4 · grounded vs extractive 三组对照（半天）

按 roadmap 原计划，把 `rag_keyword` / `rag_dense` / `rag_rewrite` 都跑成 `_extractive` 和 `_grounded` 双版本。这是让"我把 RAG pipeline 拆成两层"这个故事真正立得住的关键。

#### M5 · 把 DROPT 加成独立 baseline（半天）

新增 `policy_difffno` baseline 跑全评测集，至少在 `policy_recommendation` 那 20 条上跑出真实推理时间 + action 分布。报告里加一段：

```
DROPT inference latency: avg X ms (5 denoising steps, 20-d state)
DROPT action diversity: 标准差 vs rule-based 标准差
```

**这是把"我集成了论文模型"升级成"我集成了论文模型并测了它"的关键。**

### 🟢 下周做

#### M6 · `docs/tier1_optimization_report.md`（1 小时）

handoff prompt 第 3 步要求的报告，等 M1-M5 跑完再写——把 before/after 数据 + 反直觉发现 + "如果重做会改什么"三段式写完。这是**面试时直接拿出来的素材**。

#### M7 · 24 条人工标注（3 小时）

仍然欠账。

---

## 四、和上一轮（早上）评估的对比

| 维度 | 早上（[project_review](project_review_2026_05_22.md)） | 现在 | 变化 |
|---|---|---|---|
| 整体完成度 | 65-70% | **70-75%** | ↑ 5pp |
| Safety Audit 深度 | 1/10（无对抗测试） | **9/10** | ↑↑↑ |
| RAG 真 generation | 0/10（join chunk） | **6/10** | ↑↑（架子有了，对照不全） |
| DROPT 真接通 | 假设是 stub | **9/10** | ↑↑↑（最大惊喜） |
| ReAct baseline | 0/10（不存在） | 4/10 | ↑（架子有了，无差异化数据） |
| Dense 真实 BGE | 9/10 | **6/10**（⚠️ 回归） | ↓ 配置丢了，需要 M2 修 |
| 测试覆盖 | 27 个文件 | **174 个 test case** | ↑↑ |
| Git 历史 | 12 个 commit | 13 个 + 18 个未提交 | 退步 |
| 人工标注 | 0/24 | 0/24 | 不变 |

**净评价**：A/C 是质的飞跃（Safety Audit 跑出反直觉、DROPT 真接通），B/D 架子搭好但评测层欠债，加上 dense 配置回归是新出的 bug。**做的事多了，但落地完整度参差**——这正好是用 AI 执行 roadmap 的典型副作用：每件事都"看起来动了"，但深度需要人工补刀。

---

## 五、面试叙事更新

### ✅ 现在可以自信讲（基于这一轮的真实进展）

1. **Safety Audit + 对抗测试**：
   > "我做了 29 条对抗 prompt 测 Safety Audit，跑出 paraphrase 100% 命中、translation 0% 漏报的反直觉结果。这正是用确定性规则的 known limitation——下一步是双语字典或 LLM judge 双重验证。"

2. **DROPT 真接通**：
   > "把论文里的 Guided-DiffFNO（FNO 谱卷积 + 5 步去噪扩散 + offline RL critic）打包成 policy adapter，20 维 BEAR state 真实推理。LangGraph 在 policy_recommendation 路由调度它，LLM 只做解释 + Safety Audit。"

3. **Grounded vs Extractive 设计**：
   > "我把 RAG pipeline 拆成 retrieval-only 和 grounded 两层，新增 grounding_rate 指标衡量答案是否真的来自检索证据。grounded 版本 grounding_rate=0.923，extractive 版本 0.000——这两个指标合起来才能区分检索质量和生成漂移。"
   
   ⚠️ 但只能讲到这——三组对照（M4）做完后才能完整讲"grounded 在所有 retriever 下都稳定"。

### ⚠️ 现在需要小心讲

1. **不能再说"BGE-small-zh + FAISS 让检索召回率从 keyword 55.4% 提升到 dense 69.2%"**——当前评测里 dense 是 0.477。**M2 修完前，简历这条要改成保守版**："集成 sentence-transformers + FAISS dense retrieval 接口，支持本地 BGE-small-zh，作为可选增强 baseline"。

2. **不能说"ReAct 比 LangGraph workflow 在 multi-hop 上有提升"**——当前两者指标完全相同，没数据支撑。**M3 修完前，ReAct 只能讲"我对比了 deterministic 单步 workflow 和 deterministic multi-step planner，证明在 multi-hop 场景下需要 multi-step"。**

### 🔥 高级反问（M2-M5 全部做完后可用）

> "如果重新做一遍，我会先做 grounded 三组对照——dense + grounded、keyword + grounded、rewrite + grounded——而不是只做 dense 一组。因为 grounded 的价值在于'对所有 retriever 都稳定降低生成漂移'，单组数据证明不了这件事。我做完才意识到这个评测设计缺陷。"

**这段话的杀伤力**：体现"做完了 → 发现设计缺陷 → 知道下一步怎么改"的反思能力，比"我做了 X"高 2 个段位。

---

## 六、一句话总结

> **A 和 C 是这一轮的两次跨档**（Safety Audit 跑出反直觉 + DROPT 真接通），**B 和 D 是"架子搭起来但数据没说话"**。M1-M3 三件 2 小时内能修完——修完后整体进入"简历可信、面试不慌"的状态。**dense 配置回归是个静默 bug**，不修会让你简历里最显眼的数字之一失效——优先级最高。

---

*最后更新：2026-05-22 晚 · 配套 [optimization_roadmap.md](optimization_roadmap.md) / [project_review_2026_05_22.md](project_review_2026_05_22.md) / [handoff_prompt_optimization.md](handoff_prompt_optimization.md) 使用*
