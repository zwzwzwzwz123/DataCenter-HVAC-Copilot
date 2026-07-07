# DataCenter-HVAC Copilot —— 项目面试完全攻略

> 目标岗位：大模型算法工程师 / Agent 算法工程师
> 用途：简历项目面试准备 + 系统性梳理项目知识
> 一句话定位：面向数据中心 HVAC 运维的**检索增强工具型 Agent**——LLM 只负责「规划」和「解释」，绝不直接产生控制动作，每一次工具调用都由本地 runtime 校验执行。最新工作是把线上 planner 的路由决策能力用**人工 gold 做 SFT 蒸馏**进一个本地 1.5B 小模型。

---

## ⚠️ 全文最重要的术语纪律（先记这一条）

面试里凡涉及蒸馏，务必守住这条边界，说错会被直接扣分：

- 最终小模型是**纯 SFT，训练标签 100% 是人工手标 gold**（数据卡里 `teacher: hand_labeled`）。
- **DeepSeek 是评测时的对照基线**（线上另一条 planner 路线），**不是训练数据的来源**。
- 正确表述：「用人工 gold 给 planner 做 SFT」，**不要说**「蒸馏了 DeepSeek」——后者会被理解成拿 DeepSeek 的输出当标签。
- 「反超 DeepSeek」指的是**反超评测对照基线**，不是「学生超过教它的老师」——它的老师是人类标注，没有大模型教师。

记住这条，后面「危险问题」那一节的一大半就能稳稳接住。

---

## 一、项目价值包装

### 1.1 最值得讲的技术亮点（按面试性价比排序）

**亮点一：把「LLM 规划 / 工具执行」彻底解耦，LLM 永远在受控边界内。**
LLM（大语言模型）只做两件事——把问题拆成一段计划、对聚合好的证据做解释；它**不产生任何控制动作**，也不直接调用工具。真正的工具调用由本地 `AgentTaskExecutor` 校验后执行，控制建议只能来自 policy 工具。这套「LLM 提议、runtime 裁决」的边界让系统可解释、可回退、可测试，是整个项目的设计主心骨。面试时这是你区别于「调个 API 套个 prompt」的关键。

**亮点二：受控 route planner + 确定性 guard，任何非法计划都被拦截并优雅回退。**
Planner 只允许输出 4 类 route，计划长度 1–5 步，工具名 / `time_window` 都过白名单校验，`policy_recommendation` 必须是最后一步。非法 JSON、未知 route、超长计划、LLM 超时——全部回退到确定性 planner，并在 trace 里记录 `fallback_used`。这体现「不可信组件外面必须包一层确定性校验」的工程判断。

**亮点三（最新、最亮）：用人工 gold 给 planner 做 SFT 蒸馏，本地 1.5B 反超云端 DeepSeek 对照。**
把「问题 → 合法计划」这个结构化决策能力，用 700 条经守卫校验的人工 gold 监督微调进 Qwen2.5-1.5B（QLoRA）。四方评测里本地小模型达 **84% step_acc**，是规则基线（14%）的 6 倍，反超作为对照基线的云端 DeepSeek（63%）21 个点，延迟仅其 1/23、零 API 成本。而且训练 / 推理 / 验收复用同一套 prompt 构造与解析函数，零格式漂移——这是「蒸馏能真正接回系统」的前提。

**亮点四：bounded ReAct 循环——有边界的自主性，而不是开放式 agent。**
初始计划之后进入 ReAct（Reason+Act，边推理边行动）循环，但 controller 每轮只能从 5 个结构化动作里选（continue / insert / replace / stop_and_answer / stop_blocked）。每个动作都过本地校验：route/tool 白名单、5 步预算、非相邻重复工具拦截、input signature 去重、policy 必需步骤保护、policy deadline guard。这是对「ReAct agent 容易失控、死循环、重复调用」的直接工程回应。

**亮点五：检索命名成可验证 baseline，收益能归因到具体改动。**
`rag_hybrid`（BM25 词法）和 `hybrid_rrf`（BM25 + dense 用 RRF 融合）严格区分命名，可选 cross-encoder 做二阶段精排。评测表因此能把每一分收益归因到具体技术改动，而不是"换了个模型分数就上去了"说不清。

### 1.2 简历项目描述推荐写法（STAR 格式）

> STAR = Situation（背景）/ Task（任务）/ Action（做法）/ Result（结果）。简历上不必写出四个字母，但每一条都应暗含这四要素。

**版本 A（偏 Agent 系统，主线）**

- **S**：面向数据中心 HVAC 运维的问答场景，需要基于运维文档 + HVAC 仿真遥测回答问题，但 LLM 直接产控制动作有安全风险、且行为不可控。
- **T**：设计一个「LLM 只规划与解释、工具执行全程受控」的检索增强工具型 Agent，并保证可评测、可回退。
- **A**：实现受控 route planner（4 类 route、1–5 步、确定性 guard 校验）+ bounded ReAct 循环（5 类结构化动作、步预算、去重与 policy 保护）+ 跨 workflow 共享 executor + 确定性安全审计；检索侧做 BM25 / dense / RRF 融合 / cross-encoder 精排的可验证 baseline。
- **R**：Agent runtime 集上 required_step_recall 0.99、tool_sequence_accuracy 0.935、approval_block_success_rate 1.00、trace_completeness 1.00；检索 `hybrid_rrf_cross_encoder` 达 Recall@10 0.854 / MRR@10 0.797。

**版本 B（偏 SFT / 模型蒸馏，最新亮点，建议主推）**

- **S**：线上 route planner 走云端 DeepSeek，联网、有延迟（约 18s/条）、有 API 成本，且通用大模型没针对本项目 schema 微调，大量输出被守卫拒。
- **T**：把 planner 的结构化路由决策能力蒸馏进一个本地可跑的小模型，做到更快、零成本、且合法率更高。
- **A**：构造 700 条**经线上同一个 guard 校验**的人工 gold（12 工具全覆盖、多步 ≥2 占 56%），用 QLoRA（4-bit + LoRA）微调 Qwen2.5-1.5B；训练 prompt、标签序列化、验收解析全部复用线上函数杜绝格式漂移；用「推理层归一化兜底（A2）+ 补数据治本（A1）」两步解决时间窗格式短板。
- **R**：val 合法率 100% / route 精确匹配 97.14%；端到端四方评测 step_acc 达 84%，是规则基线的 6 倍，反超云端 DeepSeek 对照 21 个点，单条延迟 0.8s（对照的 1/23）、零 API 成本。

> 一句话简历版：「设计并落地受控工具型 Agent（LLM 规划+解释、工具执行全程 guard 校验、bounded ReAct、可回退），并用人工 gold 对 planner 做 QLoRA SFT 蒸馏，使本地 1.5B 在复合任务规划上达 84% step_acc、反超云端 DeepSeek 对照基线 21 个点、延迟降至 1/23。」

---

## 二、面试题库（含答案）

答案统一采用「先结论，后展开」结构。每题标注**考察意图**，帮你理解面试官真正想确认什么。

### 🟢 基础层（必答）—— 确认你真的做过这个项目

**Q1. 用两三句话讲清这个项目是做什么的。**
> 考察意图：能否一句话说清项目定位，是不是真的想清楚了系统边界。

**结论**：这是一个面向数据中心 HVAC 运维的检索增强工具型 Agent，基于运维文档和 HVAC 仿真遥测回答问题。

**展开**：系统接到问题后，先由受控 planner 把它拆成一段 1–5 步的工具计划，再进入 bounded ReAct 循环执行——工具分三类：文档检索（RAG）、时序分析、策略查询；执行完聚合证据、生成有据可依的解释，最后交由确定性安全审计检查。核心边界是：**LLM 只负责规划和解释，不直接产生控制动作**，每一次工具调用都由本地 runtime 校验和执行。数据源是 BEAR 这个开源 HVAC 仿真环境的 rollout 数据，明确不是真实生产遥测。

**Q2. 系统从收到问题到给出答案，完整链路是怎样的？**
> 考察意图：是否真正理解自己系统的数据流，而不是只知道几个名词。

**结论**：问题 → 受控 planner 出计划 → bounded ReAct 循环执行 → 共享 executor 调工具 → 证据聚合 → 答案生成 → 安全审计，全程写入可追踪的 trace。

**展开**（按 `/ask` 的实际流程）：
1. **规划**：`RoutePlanner` 把问题拆成受控计划（`PlanDecision`，含若干 `PlanStep`）。可以是确定性 planner（关键词路由），也可以是 LLM planner（DeepSeek），LLM 出错就回退确定性。
2. **执行**：进入 bounded ReAct 循环，controller 每轮从 5 个结构化动作里选一个决定下一步。
3. **工具调用**：由共享的 `AgentTaskExecutor` 真正执行——RAG 检索 / 时序工具 / policy 工具，输入都过 pydantic schema 校验。
4. **聚合**：把 citations、retrieved_contexts、tool_results、policy_result 聚合成 evidence bundle。
5. **生成**：answer generator（默认确定性、可选 DeepSeek/Ollama）**只解释聚合后的证据**。
6. **审计**：`audit_answer` 做确定性边界检查，标记「把仿真说成真实遥测」「LLM 直接控制」「策略动作未经 policy 工具验证」这类违规。
7. 返回 `workflow_trace`、`todos`、`runtime_trace`（hooks / approvals / recoveries）、`react_trace`。

**Q3. 你为什么要区分四类 route？分别是什么？**
> 考察意图：确认 planner 的输出空间是你自己设计的，能讲清每类的用途。

**结论**：四类 route 是 `document_qa`（文档问答）、`timeseries_query`（时序查询）、`anomaly_diagnosis`（异常诊断）、`policy_recommendation`（策略建议），对应四种任务意图，且是一个**受限的、可校验的输出空间**。

**展开**：把 planner 的输出限制在这 4 类枚举里，是为了让计划**可校验**——guard 能检查「route 是否合法」「这个 route 下能用哪些工具」「policy 是否放在最后」。如果让 LLM 自由发挥输出任意 route，就没法做确定性校验了。每类 route 对应一组白名单工具，比如 `timeseries_query` 下有 8 个工具（query_metric / compare_period / plot_metric_trend / compute_energy_breakdown / data_quality_check / zone_hotspot_rank / control_action_audit / cooling_efficiency_summary），`policy_recommendation` 只有 policy_runner。

**Q4. 「LLM 不直接产生控制动作」具体是怎么保证的？**
> 考察意图：这是项目的安全主心骨，考察你是否真的从机制上实现了，而不是嘴上说说。

**结论**：靠三层保证——planner 的输出 schema 里根本没有「控制动作」这个字段；控制建议只能来自 `policy_runner` 工具；最后还有确定性 audit 兜底检查回答里有没有偷偷出现未经验证的动作。

**展开**：
- **结构上**：`PlanStep` 的字段只有 route/reason/tool/metric_name/zone_id/time_window，没有任何「动作」出口。LLM 能做的只是「选择走哪个 route、用哪个工具」。
- **执行上**：真正的控制建议由 policy 工具产生（rule-based / MPC-like / offline replay / DROPT 等 adapter），LLM 只能读取 `policy_result` 去解释。
- **审计上**：`audit_answer` 用正则检查回答里的 `recommended_action=[...]`，若它和 `policy_result` 里 policy 工具实际返回的动作不一致，就标记 `unverified_policy_action` 违规。这样即使生成器幻觉出一个动作，也会被抓出来。

**Q5. planner 有哪几种？默认走哪个？LLM planner 挂了怎么办？**
> 考察意图：确认你理解「可选 LLM + 确定性回退」这套可用性设计。

**结论**：有确定性 planner（`DeterministicRoutePlanner`，关键词路由）和 LLM planner（`LLMRoutePlanner`，走 DeepSeek），由环境变量 `LANGGRAPH_PLANNER_PROVIDER` 决定；默认 `auto`——配了 `DEEPSEEK_API_KEY` 就用 DeepSeek，否则用确定性。LLM 挂了自动回退确定性，并在 trace 里标 `fallback_used=True`。

**展开**：这是典型的「不可信外部依赖 + 确定性兜底」设计。LLM planner 的 plan 方法里，只要出现非法 JSON、未知 route、超长计划、API 超时/异常，`except` 分支就调 `self.fallback.plan(...)` 拿确定性计划，并把失败原因拼进每一步的 reason。所以系统**在没有任何 API key 的情况下也能全程跑通**，这对 demo 和评测的可复现性很关键。

**Q6. 项目的评测是怎么做的？为什么要两个独立评测集？**
> 考察意图：确认你有「拿数据说话」的习惯，且理解不同能力要用不同评测集衡量。

**结论**：两个独立集分别衡量不同能力——一个 50 条手写子集（背后 7 篇公开 PDF、340 chunks）测**检索排序和回答质量**，另一个 50 条场景集测 **Agent runtime / guardrail 行为**（带难度分层、干扰项、注入的失败模式）。

**展开**：
- **检索 / 回答集**：配真实 embedding（BGE-small-zh + FAISS）、BGE cross-encoder reranker、DeepSeek answer generator。最佳检索配置 `hybrid_rrf_cross_encoder` 达 Citation/Context 0.781、Recall@10 0.854、MRR@10 0.797。
- **runtime 集**：required_step_recall 0.990、tool_sequence_accuracy 0.935、approval_block_success_rate 1.000、recovery_success_rate 0.833、trace_completeness 1.000。
- **为什么分开**：检索质量和 agent 编排是两种正交能力，混在一起测会互相污染归因。而且 runtime 集**刻意在 hard 难度保留失败信号**（比如 hard 场景 duplicate_guard 0.500、recovery 0.600），不是刷满分，是为了暴露真实短板。这些指标是**确定性 proxy，不是 LLM-judge 幻觉率**——这一点要主动说清，显得诚实。

**Q7.（针对 SFT 工作）你最近做的蒸馏，到底训练了什么、用什么数据训的？**
> 考察意图：这是简历新亮点，先确认你能把「做了什么」说准，尤其是训练数据来源这条红线。

**结论**：训练的是一个 **route planner**——把「用户问题 → 合法工具计划（JSON）」这个能力，用 **700 条人工手标 gold** 通过 QLoRA 监督微调进 Qwen2.5-1.5B-Instruct。训练标签 100% 是人工标注，**不是** DeepSeek 输出。

**展开**：线上 planner 可以走云端 DeepSeek，但它联网、慢（约 18s/条）、有成本。我把这个结构化决策能力 SFT 进一个本地 1.5B 小模型。数据是人工标的 `{question, steps}`，每一条都过线上同一个 `validate_plan_steps` guard 校验，700/700 全部合法。训练 630 / 验证 70，3 epoch，约 2 分钟，val 合法率 100%、route 精确匹配 97.14%。**（务必守住术语纪律：DeepSeek 是评测对照基线，不是训练教师，见开头那条。）**

### 🟡 进阶层（加分）—— 考察你对技术的理解深度

**Q8. QLoRA 里的 Q、LoRA 分别解决什么问题？为什么不用全参数微调？**
> 考察意图：SFT 的核心技术选型，能否讲清每个组件的动机。

**结论**：LoRA（Low-Rank Adaptation，低秩适配）解决「可训练参数太多、显存装不下、产物太大」；Q（4-bit 量化）解决「基座权重本身占显存」。两者叠加让单张 12–16G 消费级卡就能微调 1.5B。全参微调光是 Adam 优化器状态就要好几倍模型大小的显存，小卡根本装不下。

**展开**：
- **LoRA**：不改原权重矩阵 W，而在旁边加低秩旁路 BA（B 是 d×r、A 是 r×d），前向输出 = Wx + BAx，训练时冻结 W、只训 A/B。微调对权重的改动通常是低秩的，所以低秩逼近就够用。本项目 `r=16`、`alpha=32`（缩放因子 alpha/r=2 是常见缺省），注入注意力的 q/k/v/o 投影 + MLP 的 gate/up/down 投影——这是 Llama/Qwen 系 decoder 的标准注入点。产物只有几十 MB，可插拔。
- **Q（QLoRA 的量化）**：把冻结的基座从 16-bit 压成 4-bit（nf4 格式，专为神经网络权重分布设计），再用 double quant 对量化常数本身再量化一次。基座只是「被查询」不参与梯度，量化误差影响小；真正学习的 LoRA 旁路保持高精度。1.5B 的 16-bit 权重约 3GB，4-bit 后不到 1GB。

**Q9. completion-only loss 是什么？不用它会怎样？**
> 考察意图：是否理解 SFT 数据的 loss 计算细节，这是区分「跑过」和「懂」的分水岭。

**结论**：completion-only loss 指**只对 assistant 的答案部分算交叉熵，把 prompt（system+user）部分的 loss 掩掉**。不用它，模型会浪费容量去学「复述那段固定的系统提示」，而那段提示推理时本来就是给定的、不需要模型生成。

**展开**：一条训练样本 = prompt（工具 schema 说明 + 用户问题）+ completion（该输出的计划 JSON）。我们要模型学的只是「给定问题该产出什么计划」这个映射，所以用 `SFTConfig(completion_only_loss=True)` 把 prompt 部分 mask 掉。实现上值得一提的坑：旧版 TRL 用 `DataCollatorForCompletionOnlyLM` 靠字符串标记定位答案起点；TRL 1.7.1 把它内置成了 `completion_only_loss=True`，配合 prompt/completion 两字段数据格式自动掩码。我在服务器上就踩过这个 API 迁移（TRL 1.7.1 + transformers 5.13.0），把旧 API 全部迁移过来了。

**Q10. 你反复强调「训练格式 = 推理格式」，这具体指什么？做错会怎样？**
> 考察意图：这是蒸馏能真正上线的关键工程细节，很能体现落地能力。

**结论**：指训练数据的 prompt 构造、标签序列化、验收时的解析，**三处复用线上同一套函数**（`build_planner_messages` / `serialize_plan_steps` / `_decision_from_llm_payload`）。做错——比如训练用 A 格式、线上喂 B 格式——模型学到的分布和线上输入对不上，行为会漂移，蒸馏出来的模型接不回系统。

**展开**：Qwen 这类 instruct 模型要按它约定的 chat template（`<|im_start|>role...<|im_end|>`）拼接对话。我的做法是：训练数据的 prompt 直接调线上 planner 的 `build_planner_messages()` 生成，再由 SFTTrainer 用模型自己的 chat template 渲染；标签用 `serialize_plan_steps()` 生成成线上解析器认得的 JSON；训练后验收，用线上 `_decision_from_llm_payload()`（就是线上把 LLM 输出解析成 PlanDecision 的那个函数）去解析、校验。这样「训什么、线上跑什么、拿什么验收」由同一套代码保证一致。

**Q11. 你的 val 指标为什么用「合法率」和「精确匹配」，而不只看 loss？**
> 考察意图：是否理解「loss 低 ≠ 模型能用」，有没有端到端评测意识。

**结论**：loss 低只说明拟合得好，不代表输出真能被系统接受。我的验收指标直接复用线上代码：对 70 条 val 问题让模型**实际生成**计划，再用线上解析器 + guard 去判——`legal_rate=100%`（生成的计划都能被线上接受）、`exact_match=97.14%`（route 序列和人工标准答案完全一致）。

**展开**：这是「端到端、用生产同一套判据」的评测。它回答的是「这个小模型能不能真的替代线上 planner」，而不是「模型拟合得好不好」。100% 合法率不是自定义的宽松指标，是「输出可直接上线」的信号。判断有没有过拟合我看 eval_loss 是否单调下降不反弹——本项目 eval_loss 0.144→0.109→0.108 收敛，train/eval loss 没背离，所以没过拟合。

**Q12. A2 和 A1 两步优化分别是什么？为什么要分两步？**
> 考察意图：考察你解决问题的方法论——是否会区分「快速止血」和「根因治理」。

**结论**：A2 是**推理层兜底**（不用重训），A1 是**数据层治本**（补数据重训）。同一个问题（模型爱输出自然语言时间窗、被 guard 拒），先用 A2 快速止血，再用 A1 根治。step_acc 从 47% →（A2）68% →（A1）84%。

**展开**：
- **问题**：模型倾向输出 `past 7 days`、`last month`、`7d` 这类自然语言时间窗，但线上 guard 只认受限词表（`last_24_hours`、`last_168_hours` 等），不合规就被拒、回退规则基线。
- **A2（推理层，安全网）**：在解析层加 `_normalize_time_window`，把自然语言窗映射回合法词表（天/周/月折算成小时，因为 gold 只用小时——保证训练/推理一致）。无法可靠映射的值原样返回让 guard 照常拒绝，绝不强行猜。不用重训，蒸馏 47%→68%。
- **A1（数据层，治本）**：补 100 条经守卫校验的 gold——question 用自然语言窗、label 直接标成合法小时词表，让模型**自身**学会归一化；同时补 8 个 timeseries 工具的消歧样本 + 多步复合任务。重训后 68%→84%。
- **为什么分两步**：A2 是不需重训的快速止血（几分钟上线），A1 把能力固化进权重才是根治。这套「先兜底、再治本」的思路本身就是工程判断的体现。

**Q13. 讲讲 RRF 融合检索。为什么用 RRF 而不是直接把 BM25 和 dense 的分数加起来？**
> 考察意图：检索侧的核心技术点，考察你懂不懂「分数不可比」这个坑。

**结论**：RRF（Reciprocal Rank Fusion，倒数排名融合）用**排名**而非**分数**来融合多路召回——每个候选的融合分 = Σ 1/(k+rank)。直接加分数不行，因为 BM25 分数和 dense 余弦相似度**量纲完全不同、不可比**，强行归一化又会引入新的假设和噪声。

**展开**：BM25 分数可能是 0~几十的 tf-idf 累加，dense 是 -1~1 的余弦相似度，尺度差异巨大。RRF 只取「你在这一路里排第几」，天然免疫量纲问题，`k`（本项目 60）是个平滑常数防止 top-1 独大。实现上我把 `hybrid_rrf`（BM25+dense RRF）和 `rag_hybrid`（纯 BM25 词法）严格分开命名，就是为了让评测表能把收益归因到「加了 dense 融合」这个具体改动，而不是笼统说「换了检索器分数就上去了」。RRF 之上还可以挂 cross-encoder 做二阶段精排。

**Q14. bounded ReAct 的 5 个动作是什么？它和普通 ReAct agent 的区别在哪？**
> 考察意图：考察你对「受约束的自主性」的理解，这是当前 agent 领域的热点。

**结论**：5 个动作是 `continue_next_step`（执行下一步）、`insert_step`（插入新步）、`replace_next_step`（替换下一步）、`stop_and_answer`（停下作答）、`stop_blocked`（被拦截而停）。区别在于：普通 ReAct 让 LLM 自由决定下一步做什么，容易失控、死循环、重复调工具；bounded ReAct 把动作限制成这 5 个**结构化选项**，且每个动作都过本地校验。

**展开**：每一轮 controller 的决策都要通过：route/tool 白名单、5 步预算、非相邻重复工具拦截、input signature 去重（同样的工具+参数不会重复调）、policy 必需步骤保护、policy deadline guard。任何被批准或 policy 边界拦截的动作会触发 `stop_blocked`。controller 本身也分确定性和 LLM 两种，LLM 出错回退确定性。这套设计是对「开放式 agent 不可控」的直接回应——**自主性被约束在结构化动作、步数预算、任务义务和本地 guardrail 之内**。

**Q15. 为什么确定性 baseline、LangGraph workflow、bounded ReAct 要共享一个 executor？**
> 考察意图：考察架构解耦意识——编排策略和工具执行分离的价值。

**结论**：因为**编排策略**（怎么决定下一步）和**工具执行**（实际怎么调 RAG/时序/policy）是两个正交的关注点。共享同一个 `AgentTaskExecutor` 保证：换编排策略时工具行为不漂移，且确定性 baseline 能作为回归对照检测行为漂移。

**展开**：如果每种编排各写一套工具执行逻辑，那评测时就分不清「指标变化是编排改了还是工具执行改了」。共享 executor 后，工具执行、RAG 检索、policy 调用、answer audit 都走同一条路径，编排层只负责「决定执行哪些 step、什么顺序」。这样 workflow 可以自由迭代演进，而确定性 baseline 始终是那把「行为没漂移」的尺子。

### 🔴 深挖层（拉开差距）—— 考察技术判断力与举一反三

**Q16. 你说本地 1.5B 反超了云端 DeepSeek——这是不是意味着 1.5B 比 DeepSeek 更强？**
> 考察意图：这是最关键的「诚实性 + 理解深度」双重考察。答错（沾沾自喜说小模型更强）直接暴露你不懂评测口径。

**结论**：不是。小模型**不是更聪明**，是被专门训练成「只产出本系统能接受的合法计划」。反超的是**评测对照口径**，不是通用能力。

**展开**：反超的真实原因很实在——DeepSeek 是通用模型，没针对本项目 schema 微调，大量输出用了自然语言时间窗、越界工具，被线上 guard 拒后回退，step_acc 就低；而小模型用 700 条**经守卫校验**的 gold 专门训练，天生产出合规计划。所以这恰恰说明：**对一个 schema 明确、输出受限的窄任务，专门做 SFT 的小模型可以打败通用大模型**，这正是「为什么值得为窄任务做蒸馏而不是直接调云端大模型」的论据。如果换成开放域推理、创意写作，1.5B 绝不可能反超 DeepSeek。把边界讲清楚，比吹「我的小模型更强」有说服力得多。

**Q17. guard 会「按 route 去重」，这个细节在你构造 A1 数据时引发了什么坑？你怎么防的？**
> 考察意图：极深的细节题，考察你是不是真的动手标过数据、踩过坑。能答出来基本就证明项目是你做的。

**结论**：`_validate_steps` 里遇到同一个 route 的第二个 step 会**静默跳过**（`if step.route in seen: continue`）。这意味着如果我标了两个都是 `timeseries_query` 的步骤，第二个会被悄悄丢掉——label 就和问题不符了，但不会报错。我在 A1 的 builder 里加了硬校验：**「授权步数 == 校验后步数」**，任何触发去重的行直接报错拒收。

**展开**：这个坑很隐蔽，因为 guard 不会抛异常，只是返回一个更短的 plan。如果不察觉，训练数据里就混入了「问题要两步、标签只有一步」的脏样本，模型学到的路由就是错的。我的做法是在 append 前对每一行同时验证：① 过 `validate_plan_steps` 不抛错；② 校验前后步数一致（没被去重合并）；③ 过线上解析器零 fallback。100/100 全过才写入。**这也说明校验器既是训练数据的守门员，又是标注质检工具**——它在标注过程中抓出过我把 `last_7_days` 写成不支持格式的错误。

**Q18. 如果让你继续优化那 10 条残余 fallback，你会怎么做？为什么不是继续补数据？**
> 考察意图：考察你能不能诊断根因、区分「同类问题加数据」和「能力边界要改架构」。

**结论**：残余 10 条**全是最难的 3 步复合任务**，且**不是** time_window 或工具选择问题（那两类 A1 已基本解决）。它们分三类：多指标同图/比值绘图、绝对时段窗（10PM–6AM）、episode/多文档具体引用。这三类**不能靠继续补同类数据解决**，因为它们撞到了 schema 表达力的天花板。

**展开**：
- **多指标绘图**（`plot cooling_power and fan_power ratio`）：单步 schema 只有一个 `metric_name`，表达不了两个 metric 的组合——这是**要改 schema**（支持 metric 列表），动到线上 guard，得谨慎评估。
- **绝对时段窗**（`during the night 10PM–6AM`）：非「最近 N 小时」型，词表本就不支持——要在归一化里引入「时段」概念，也是能力扩展。
- **具体 id 引用**：模型对 `episode_001`、`doc_002` 这类具体 id 处理不稳。
- **我的判断**：接下来两条路——(a) **阶段 3 DPO**，正好用这 10 条真实错误当 `rejected` 负例，比人工编造真实；(b) 针对多指标/时段做 **schema 扩展（A3）**。补同类 gold 已经边际收益递减，因为问题不在「见得少」，在「表达不了」。

**Q19. 你的 planner 输出是 LLM 生成的 JSON，JSON 可能截断或非法。整条链路怎么保证不崩？**
> 考察意图：考察容错设计的完整性——不可信输出的多层防护。

**结论**：多层防护 + 全程可回退。解析失败、非法 route、越界工具、超长计划、LLM 超时——任何一环出问题都回退到确定性 planner，并在 trace 里标 `fallback_used=True`，绝不让异常冒泡到用户。

**展开**：具体分层——① `_parse_json_object` 先剥 ```` ```json ```` 代码块再 `json.loads`，失败进 except；② `_decision_from_llm_payload` 要求 `steps` 是 list，逐项过 `_step_from_llm_item`；③ `_validate_steps` 做 route/tool/time_window 白名单校验 + policy 末位校验；④ 任何 ValueError 被 `LLMRoutePlanner.plan` 的 try/except 捕获，调 `self.fallback.plan()` 回退，并把失败原因写进 step 的 reason。蒸馏 planner（`DistilledRoutePlanner`）同理，还额外在 `plan_batch` 里对单行失败做逐行回退。**关键理念：不可信组件（LLM）外面必须包确定性校验层，且失败要优雅降级而非崩溃。**

**Q20. 安全审计（answer audit）是怎么做的？它的局限是什么？你会怎么诚实地描述它？**
> 考察意图：考察你对「安全」的态度——是否夸大、是否理解边界检查器 ≠ 完整安全系统。

**结论**：它是一个**确定性的边界检查器**，用关键词/正则匹配三类高风险表述：把 BEAR 仿真数据说成「真实生产遥测」、声称「LLM 直接生成控制动作」、策略回答里出现 policy 工具没返回的动作。它**不是**完整安全系统，当前 adversarial hit rate 0.657，translation 类是 0.000（已知短板）。

**展开**：实现上 `audit_answer` 对每类先查「安全表述」白名单（比如「LLM 不直接生成控制动作」是安全的），命中就放行；再查风险短语。`unverified_policy_action` 会正则抽出回答里的 `recommended_action=[...]`，和 policy_result 里真实返回的动作比对，不一致就标违规。**诚实描述的关键**：我会主动说「这是边界检查器，不是安全护栏的全部；它能挡住明显的表述越界，但对翻译类改写、语义绕过无能为力，hit rate 0.657 就是如实的数字」。面试官更看重你**不粉饰局限**。

**Q21. QLoRA 里，4-bit 量化的是什么、训练的是什么？为什么这样能省显存又不太掉精度？**
> 考察意图：考察 QLoRA 的机制理解，是否只会喊名词。

**结论**：**冻结的基座权重被压成 4-bit（nf4）**，只有 LoRA 旁路（高精度的 A、B 两个低秩矩阵）参与训练。省显存是因为占大头的基座从 16-bit 变 4-bit；不太掉精度是因为基座只被「查询」不参与梯度更新，量化误差影响有限，真正学习的 LoRA 参数是高精度的。

**展开**：
- **LoRA**：不直接改原权重 W，而在旁边加低秩旁路 ΔW=BA（A 是 d×r、B 是 r×d，本项目 r=16）。前向 = Wx + BAx，训练只更新 A、B。原理是「微调对权重的改动通常是低秩的」，用低秩矩阵逼近 ΔW 效果接近全参微调，可训练参数降到百分之一以下，产物只有几十 MB 可插拔。
- **QLoRA 的量化**：`BitsAndBytesConfig(load_in_4bit, nf4, double_quant)`。nf4 是专为神经网络权重分布设计的 4-bit 格式；double_quant 再对量化常数本身量化一次，进一步省。
- **为什么单卡 12–16G 能跑 1.5B**：16-bit 权重约 3GB，4-bit 后不到 1GB，加激活和 LoRA，一张 16G 卡绰绰有余。
- **LoRA 加在哪**：`q/k/v/o_proj`（注意力四投影）+ `gate/up/down_proj`（MLP 三投影），是 Qwen/Llama 系 decoder 的标准注入点。

**Q22. completion-only loss 是什么？不用它会怎样？**
> 考察意图：考察 SFT 训练细节，是否理解「学什么不学什么」。

**结论**：只对 assistant 输出的计划 JSON 算交叉熵损失，把 prompt（system 工具说明 + user 问题）部分的 loss 掩掉（mask）。不用它的话，模型会浪费容量去「学着复述那段固定的系统提示」，但那段提示推理时是给定的、不需要模型生成，白学。

**展开**：一条样本 = prompt + completion。对整条算 loss，模型的一部分学习信号被固定不变的 system prompt 稀释了。掩掉 prompt 后，训练信号集中在「给定问题该产出什么计划」这个真正要学的映射上。实现变迁值得一提：旧版 TRL 用 `DataCollatorForCompletionOnlyLM` 靠字符串标记定位答案起点；TRL 1.7.1 内置成 `SFTConfig(completion_only_loss=True)`，配合 prompt/completion 两字段格式自动掩码——我在服务器上从旧 API 迁到新 API 时踩过这个坑（连带 `max_seq_length`→`max_length`、`tokenizer=`→`processing_class=`）。

**Q23. 你怎么保证 train/val 没有数据泄漏？验证集分数虚高怎么发现？**
> 考察意图：考察数据纪律，这是区分「认真做过」和「跑个脚本」的分水岭。

**结论**：划分时保证**问题不重复、train/val 无交叉**。曾经查出 1 条重复问题（gold_0479）并修掉。用 seed 固定划分（`build_gold_sft` 里 `--seed 13`、`--val-ratio 0.1`），可复现。

**展开**：如果同一个问题同时进了 train 和 val，验证集分数会虚高——模型是「背」出来的不是「学」出来的。发现手段：① 划分前对 question 做去重检查；② 看 val 指标是否高得不真实（如果 exact_match 突然 100% 反而要警惕泄漏）。本项目 val exact_match 97.14%（68/70），有 2 条不匹配反而说明没泄漏、指标可信。而且我还专门写了测试 `test_distill_gold.py`，硬断言「线上 gold 全部 100% 合法」，数字随 gold 扩充更新（现 700 条），防数据劣化。

**Q24. 假如这个 planner 要上线服务真实流量，你还差哪些工作？**
> 考察意图：考察工程落地的全局观——从「能评测」到「能上线」的差距。

**结论**：目前 `DistilledRoutePlanner` 实现了和线上一样的 `RoutePlanner` protocol，**技术上可插拔**，但还没接进线上 `/ask` 路径，且缺生产级的服务化、监控、回滚。差的主要是：模型服务化、线上 A/B、监控告警、以及 policy 边界的端到端安全验证。

**展开**：
- **服务化**：现在是 `model.generate` 本地推理，上线要用 vLLM/TGI 之类做批处理和并发，adapter 要么合并进基座要么用 LoRA 热加载。
- **灰度与回退**：线上应该 shadow / A-B——让蒸馏 planner 和 DeepSeek 并行跑一段，比对 step_acc 和 fallback 率，再逐步放量；随时能切回 DeepSeek 或确定性 planner。
- **监控**：线上 fallback 率、延迟 P99、非法计划率要有 dashboard，漂移了要告警。
- **持续数据回流**：线上真实 badcase 回流成新 gold 或 DPO 负例，形成迭代闭环。
- **诚实边界**：我会明确说「现在是离线评测验证了可行性（84% step_acc），但还没扛过生产流量」——不夸大成「已上线」。

---

## 三、追问模拟（完整对话链路）

真实面试是层层追问的。下面 3 条链路演示「问题 → 回答 → 追问 → 再回答」，重点是感受面试官如何顺着你的答案往深里挖。

### 链路一：从「反超 DeepSeek」一路挖到评测口径

**面试官**：你提到本地 1.5B 反超了云端 DeepSeek，具体是什么指标？

**你**：在 100 条多步复合任务的评测集上，step_acc（预测的 route 集合与期望完全一致的比例）小模型是 84%，DeepSeek 对照是 63%，高出 21 个点。同时小模型单条延迟 0.8s、DeepSeek 约 18s，且本地零 API 成本。

**面试官**（追问）：等一下，一个 1.5B 的模型 step_acc 反而比 DeepSeek 高，这不反直觉吗？是不是你的评测对 DeepSeek 不公平？

**你**：这个质疑很对，我得说清楚口径。这不是「1.5B 比 DeepSeek 聪明」，而是评测口径是「产出的计划能不能通过本系统的 guard」。DeepSeek 是通用模型，没针对我这个项目的 schema 微调，它大量输出自然语言时间窗（`past 7 days`）和越界工具，被 guard 拒了就回退、step_acc 就低。小模型是用 700 条**经这个 guard 校验过**的 gold 专门训练的，天生产出合规计划。所以严格说，这是「专门微调的窄任务模型，在该任务的特定口径下，胜过没微调的通用模型」——换成开放域任务，1.5B 不可能反超。

**面试官**（再追问）：那你怎么保证这个「口径」本身不是你为了让小模型好看而设计的？

**你**：因为这个 guard 不是我为评测新造的，它就是**线上系统真实使用的** `validate_plan_steps`——DeepSeek 走线上 planner 路径时也要过同一个 guard 才能执行。评测只是如实反映「哪条 planner 产出的计划更多能被线上直接接受」。而且 DeepSeek 和小模型都过 A2 的 time_window 归一化（共用同一个解析函数），起跑线一致。如果我想「作弊」，最简单的反而是不加 A2、让 DeepSeek 更惨——但我加了，因为要如实比。

---

### 链路二：从 SFT 挖到「为什么不直接用 DeepSeek」

**面试官**：你既然有 DeepSeek 可以用，为什么还要费劲蒸馏一个小模型？

**你**：三个原因：延迟、成本、可控性。DeepSeek 单条规划要约 18s，还要联网、按 token 付费；蒸馏后的本地模型 0.8s、零成本、可离线。而且更关键的是合规率——通用 DeepSeek 大量计划被 guard 拒，小模型专门训练后天生合规。对「路由规划」这种高频、schema 固定的窄任务，这笔账是划算的。

**面试官**（追问）：那你为什么用 SFT 而不是直接写规则？你不是已经有个 deterministic planner 吗？

**你**：因为规则 planner 的天花板很低——它在这个复合任务集上只有 14% step_acc。规则靠关键词匹配路由，遇到多步复合、口语化、指代、中英混杂就抓瞎。SFT 能学到「把一句话拆成 2-4 步、正确排序、policy 放最后」这种规则很难穷举的模式。所以我的定位是：规则 planner 做**兜底底线**（永远合法但不聪明），SFT 小模型做**主力**（又快又准），DeepSeek 做**对照基线**。

**面试官**（再追问）：SFT 学的是「模仿 gold」，那它会不会只是背下了训练集？遇到没见过的问法就崩？

**你**：这正是我用「端到端评测集」而非「训练集内验证」来验收的原因。那 100 条复合任务和训练 gold 是**不同的集合**，小模型在没见过的问法上拿到 84%，说明它学到的是泛化的路由能力不是死记。当然它有边界——残余 10 条 fallback 全是最难的 3 步复合任务（多指标绘图、绝对时段窗），那是 schema 表达力的天花板，不是「背没背下来」的问题。这些我留给阶段 3 的 DPO 和 schema 扩展。

---

### 链路三：从 bounded ReAct 挖到「去重和死循环」

**面试官**：你的 bounded ReAct 和普通 ReAct agent 区别在哪？

**你**：普通 ReAct 让 LLM 自由决定下一步，容易失控、死循环、重复调同一个工具。我的 bounded ReAct 把动作限制成 5 个结构化选项（continue/insert/replace/stop_and_answer/stop_blocked），每个动作都过本地校验：route/tool 白名单、5 步预算、重复工具拦截、input signature 去重、policy 保护。自主性被约束在结构化动作和步数预算内。

**面试官**（追问）：你说的「input signature 去重」具体怎么实现？为什么需要它？

**你**：每个执行过的 step 会算一个 signature——本质是 (route, tool, 关键输入参数如 metric_name/zone_id/time_window) 的元组。controller 想插入新 step 时，如果它的 signature 和已执行的某个 step 相同，就判为重复、拦截掉。需要它是因为：LLM controller 可能反复想「再查一次温度」，但参数一模一样的调用不会带来新证据，只会浪费步数预算、甚至死循环。去重保证每一步都在获取**新**证据。

**面试官**（再追问）：那如果两次调用工具相同、但参数确实需要不同（比如查两个不同 zone），你的去重会误杀吗？

**你**：不会，因为 signature 包含了关键输入参数——zone_a 和 zone_b 的 `zone_id` 不同，signature 就不同，两次调用都放行。去重拦的是「route+tool+参数全一样」的真重复，不是「同一个工具」。这也是为什么我用 signature 而不是简单地「同一个 tool 只能调一次」——后者会误杀合理的多 zone/多 metric 查询。另外去重只拦**非相邻**重复，避免和正常的连续同类操作冲突。

---

## 四、危险问题预警（可能被质疑/刁难，附化解话术）

这些问题的共同点是：面试官想看你**会不会诚实、能不能守住边界**。化解的核心不是硬辩，而是「主动承认边界 + 说清楚为什么这样是合理的」。

**危险 Q1：你这个「本地 1.5B 反超 DeepSeek」是不是在标题党？小模型怎么可能比大模型强？**
> 这是最高频的质疑，必须一次答稳。

化解话术：「这个说法我特意会讲清楚边界。不是小模型通用能力更强，而是在**本项目这个窄任务、用『产出线上可接受的合法计划』这个口径**下更强。原因很实在：DeepSeek 是通用模型，没针对我的 schema 微调，大量输出用了非法时间窗或越界工具，被守卫拒了、回退成基线；我的小模型用 700 条**经同一个守卫校验**的 gold 专门训练，天生就产出合规计划。所以准确说是『专门训练的小模型在特定口径下超过通用大模型』，而且这恰恰说明——为什么值得为窄任务做 SFT，而不是无脑调云端大模型。」

**危险 Q2：你是不是蒸馏了 DeepSeek？用它的输出当标签？**
> 一旦答错就露馅，这是术语陷阱。

化解话术：「没有。我的训练标签 100% 是人工手标 gold，数据卡里写的是 `teacher: hand_labeled`。DeepSeek 在我的项目里只是**评测时的对照基线**，是线上另一条 planner 路线，不是训练数据来源。代码里确实有个 `build_sft_data.py` 支持『用线上 planner 当 teacher 生成数据』的路径，但最终模型没走这条。所以准确说法是『用人工 gold 做 planner SFT』。」（这一条答对，面试官会认为你对项目细节极其清楚。）

**危险 Q3：700 条数据是不是太少了？这样训出来的模型能信吗？**

化解话术：「对通用能力肯定不够，但这是个 schema 极窄的结构化任务——输出只有 4 类 route、十几个工具、受限的时间窗词表，本质是『把自然语言映射到一个受约束的计划』。窄任务 + 高质量数据，700 条足够。而且我的验收不是看 loss，是用线上同一套 guard 判『生成的计划能不能被系统接受』——val 合法率 100%、端到端 step_acc 84%，这些是端到端指标，说明数据量对这个任务是够的。我也如实记录了残余的 10 条 fallback，都是最难的 3 步复合任务。」

**危险 Q4：你的评测指标都是「确定性 proxy」，不是真实的人工评审或 LLM judge，那可信吗？**

化解话术：「我很清楚这个边界，报告里也写明了。检索和工具选择这类指标（Recall、tool selection accuracy、step_acc）本来就是客观可判的，proxy 完全够用。回答质量类（correctness/faithfulness）我用的是代理指标，我明确标注了它『不等价于人工评审或 LLM judge』，还留了人工校准集的接口，未填写前报告只显示 `pending_human_review`。我不会把 proxy 说成人工评审——这正是我在项目里坚持的诚实边界。」

**危险 Q5：这个项目的数据是 BEAR 仿真，不是真实数据中心，那你这套东西有实际价值吗？**

化解话术：「数据确实是 BEAR 仿真/导出，我在系统里到处强调这一点——`data_source` 元数据会显式标注来源，安全审计还专门拦『把 BEAR 说成真实生产遥测』的表述。价值不在数据本身，而在这套**方法论和工程骨架**：LLM 受控边界、guard 校验、bounded ReAct、可回退、SFT 蒸馏——换成真实遥测，这套架构直接复用。仿真数据让我能在零风险下把工程做扎实。」

**危险 Q6：安全审计就是几个关键词匹配吧？这也算「安全系统」？**
> 面试官在试探你会不会夸大。

化解话术：「它就是个**确定性边界检查器**，我从不把它叫『完整的安全系统』。它做的是字符串/正则级的边界检查——标记『真实生产遥测』『LLM 直接控制』『未验证 policy action』这类高风险表述，当前 adversarial 命中率 0.657，translation 类是 0.000、是已知短板，我都如实记了。它的定位是『输出边界的最后一道确定性防线』，不替代人工审查。承认它的局限反而是我想强调的诚实。」

**危险 Q7（针对简历真实性）：这么多模块，真是你一个人做的？某个细节你能马上讲清吗？**

化解话术：不要慌，直接选一个你最熟的模块深挖给他看。比如：「我可以讲最细的——`_normalize_time_window` 这个函数。它接在 planner 解析时间窗那一步，先判断是不是已经是合法词表，是就直接过；不是就尝试三类映射：词组（last month→last_720_hours）、拼写数字（two→2）、相对窗正则（past 7 days→last_168_hours，天×24 折算成小时）。关键设计是**折算只到小时不到天**，因为 gold 只用小时，引入 days 会让模型见到没训过的格式。无法可靠映射的原样返回让 guard 拒绝，绝不强行猜。」——用一个能讲到函数级细节的例子，真实性质疑自然化解。

---

## 五、知识补充清单（回答上述问题的关联知识点）

每个知识点给简明解释（150 字以内），按主题分组。面试前扫一遍，确保每个术语你都能自己讲一遍。

### A. 微调与蒸馏

- **SFT（监督微调）**：用「输入→标准答案」配对数据做监督训练，本质是模仿学习。损失是标准交叉熵语言建模损失——把标准答案每个 token 的预测概率拉高。和 RLHF/DPO 那种偏好优化不同。

- **知识蒸馏（Distillation）**：让小模型（学生）模仿老师的决策，把能力压进小模型。老师可以是大模型，也可以是**人类专家**。本项目老师是人工 gold（人类专家），不是大模型——这点是术语关键。

- **LoRA（低秩适配）**：冻结原权重 W，旁边加两个小矩阵 A、B，用 BA 近似权重增量 ΔW，只训练 A、B。因为微调对权重的改动通常是低秩的。产物只有几十 MB，可插拔。本项目 r=16、alpha=32。

- **QLoRA（4-bit 量化 + LoRA）**：把冻结的基座权重压成 4-bit（nf4 格式）装进显存，只有 LoRA 旁路保持高精度参与训练。让单张 12–16G 消费级卡就能微调 1.5B。基座只被查询不算梯度，量化误差影响小。

- **PEFT（参数高效微调）**：Parameter-Efficient Fine-Tuning 的统称，只训练一小部分参数就达到接近全参微调的效果，LoRA 是代表。对应 HuggingFace 的 `peft` 库。

- **completion-only loss**：一条样本含 prompt（系统+用户提示）和 completion（该生成的答案），只对 completion 部分算 loss，掩掉 prompt。避免模型浪费容量去学复述固定提示。TRL 1.x 用 `SFTConfig(completion_only_loss=True)` 内置实现。

- **chat template**：instruct 模型约定的对话拼接格式（如 Qwen 的 `<|im_start|>role...<|im_end|>`）。训练和推理必须用同一模板，否则模型学到的格式和线上输入对不上，行为漂移。

- **交叉熵损失 / 过拟合判据**：交叉熵衡量预测 token 分布与真实 token 的差距。判断过拟合看 eval_loss 是否随训练反弹、train/eval loss 是否背离；本项目 eval_loss 单调降到 0.108 收敛，无过拟合。

- **DPO（直接偏好优化）**：Direct Preference Optimization，用「chosen 优于 rejected」的偏好对直接优化模型，不需训练奖励模型。是 SFT 之后的下一步——让模型在多个合法输出中偏向更优的那个。本项目规划中的阶段 3。

### B. 检索（RAG）

- **RAG（检索增强生成）**：先检索相关文档片段，再让模型基于检索到的证据生成答案，而非纯靠参数记忆。能引用来源、减少幻觉。本项目是 extractive RAG（抽取式）。

- **BM25**：经典词法检索算法，基于词频（TF）、逆文档频率（IDF）和文档长度归一化打分。对精确关键词匹配强，但不懂语义近义。本项目 k1=1.5、b=0.75。

- **Dense 检索**：把查询和文档都编码成向量，用向量相似度（如余弦）召回。懂语义近义，但对精确术语可能不如 BM25。本项目用 BGE-small-zh + FAISS。

- **RRF（倒数排名融合）**：Reciprocal Rank Fusion，融合多路召回时用**排名**而非分数：融合分 = Σ 1/(k+rank)。免疫不同检索器分数量纲不可比的问题，k（本项目 60）是平滑常数。

- **Cross-encoder 精排**：把 (查询, 文档) 拼一起进模型算相关性分，比双塔 dense 更准但更慢。用作二阶段：先用 BM25/dense 召回候选，再用 cross-encoder 对候选精排。本项目用 BGE-reranker。

- **FAISS**：Facebook 的向量相似度检索库，支持大规模向量的高效近邻搜索。本项目 dense 向量存 FAISS 索引，重建时先写临时文件再原子替换、校验 manifest hash 防半写。

- **检索评测指标**：Recall@K（前 K 个里召回了多少相关文档）、MRR@K（第一个相关文档排名倒数的均值，衡量排序质量）、Citation/Context 命中率。

### C. Agent 与编排

- **ReAct**：Reason + Act，让模型边推理边行动——生成想法、调工具、看观察结果、再想下一步。本项目做成 bounded（有界）版：动作限制成 5 个结构化选项且每步过校验。

- **Bounded / 受控 Agent**：把 agent 的自主性约束在结构化动作、步数预算、任务义务和本地 guardrail 内，而非开放式自由执行。解决开放 agent 失控、死循环、重复调用的问题。

- **LangGraph**：把 agent workflow 建模成有状态的图（节点+边）的编排框架。本项目用它做 planner→execute→aggregate→answer→audit 的 workflow，节点复用共享 executor。

- **Guardrail（护栏）**：包在不可信组件（LLM）外的确定性校验层。本项目 planner guard 校验 route/tool 白名单、≤5 步、时间窗格式、policy 必须最后一步；非法就回退。

- **工具调用 / Tool Use**：让 LLM 通过结构化接口调用外部函数（查数据、检索、跑策略）。本项目工具由 Pydantic 定义输入/输出 schema，执行前做 `validate_tool_input` 校验。

- **人在回路 / 审批边界（HITL）**：高风险工具（本项目 policy_runner，risk_level=control_boundary）执行前需审批，审批被拒则 `stop_blocked` 且不写入有效 policy_result。低风险工具自动放行。

### D. 训练工程

- **TRL**：HuggingFace 的 Transformer Reinforcement Learning 库，提供 SFTTrainer、DPOTrainer 等。本项目用 TRL 1.7.1 的 SFTTrainer + prompt/completion 数据格式。

- **梯度累积（grad accumulation）**：显存装不下大 batch 时，累积多个小 batch 的梯度再更新，等效放大 batch。本项目 batch 4 × 累积 4 = 等效 batch 16。

- **学习率调度（cosine + warmup）**：先小幅热身（warmup）避免初期不稳，再余弦退火让学习率平滑趋零。本项目 warmup 0.03、cosine，lr 从 2e-4 降到近 0。LoRA 学习率（2e-4）比全参微调（~2e-5）大一个量级。

- **nf4 / double quant**：nf4 是专为神经网络权重分布设计的 4-bit 量化格式；double quant 再对量化常数本身量化一次，进一步省内存。QLoRA 的 `BitsAndBytesConfig` 配置。

### E. 项目领域与边界

- **BEAR**：开源的建筑 HVAC 强化学习仿真环境（`chz056/BEAR`），提供 `BuildingEnvReal` 的 rollout。本项目数据源，是**仿真/导出数据不是真实生产遥测**——这是必须守住的表述边界。

- **数据契约（native/derived/synthetic）**：区分字段来源——native 直接来自 BEAR、derived 可重复计算、synthetic 需说明生成方式。`pue`/`humidity` 等不能默认当 BEAR 原生字段，防止编造数据。

- **确定性回退（deterministic fallback）**：LLM 不可用/超时/输出非法时，退回到基于规则的确定性组件（规则 planner、规则 policy、确定性 answer generator）。保证系统在无 API key 时也能全程跑通。

- **可复现评测口径**：`/eval/run` 和评测脚本默认关掉 env LLM、用 deterministic generator + rule-based policy，避免评测触发批量 API 调用、保持指标可复现。交互式 `/ask` 才启用真实 LLM。

---

## 附：面试前 5 分钟速记卡

- **一句话**：受控工具型 Agent，LLM 只规划+解释、工具执行全程 guard 校验；最新工作是用人工 gold 给 planner 做 QLoRA SFT 蒸馏。
- **术语红线**：teacher=hand_labeled，DeepSeek 是**评测对照**不是训练教师；说「用人工 gold 做 SFT」不说「蒸馏 DeepSeek」。
- **核心数字**：700 条 gold（多步≥2 占 56%）、train/val 630/70、val 合法率 100%/精确匹配 97.14%；step_acc 84%（规则基线 14% 的 6 倍，DeepSeek 对照 63%）；延迟 0.8s vs 18s（1/23）。
- **优化两步**：A2 推理层归一化兜底（47%→68%）、A1 补数据治本（68%→84%）。
- **4 类 route**：document_qa / timeseries_query / anomaly_diagnosis / policy_recommendation；计划 1–5 步、policy 必须最后。
- **12 个工具**：1 RAG + 8 timeseries + 2 anomaly + 1 policy（policy_runner 是唯一 control_boundary、需审批）。
- **bounded ReAct 5 动作**：continue / insert / replace / stop_and_answer / stop_blocked。
- **诚实边界**：proxy 指标不等于人工评审、安全审计只是边界检查器、BEAR 是仿真不是生产遥测、残余 10 条 fallback 如实记录。

