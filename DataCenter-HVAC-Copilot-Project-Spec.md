# DataCenter-HVAC Copilot 项目技术文档

> 这是一个面向后续 AI 执行的项目说明文档。目标不是做一个普通聊天机器人，而是做一个以 BEAR 多区域 HVAC 仿真为核心数据源、面向数据中心冷却优化类问题完成“检索、分析、诊断、建议、评测”的大模型应用系统。

## 1. 项目目标

本项目要实现一个 **DataCenter-HVAC Copilot**，用于辅助基于 **BEAR 仿真环境** 的多区域 HVAC / 冷却系统运维与能耗优化。BEAR 是主要实验环境，数据中心冷却是项目面向的应用叙事和拓展场景。系统需要能够：

1. 回答与数据中心 HVAC / 冷却系统相关的领域问题。
2. 检索并引用相关文档、论文、规范、设备说明。
3. 分析时序运行数据，如温度、湿度、功率、负载、PUE、告警记录。
4. 对异常能耗、温控波动、设备状态变化给出诊断结论。
5. 生成可执行的节能建议或控制建议。
6. 在统一评测集上验证效果，不只展示 demo。

## 2. 项目定位

这是一个 **B 路线** 项目：以 RAG + Agent + 工具调用 + 评测 为主体，核心实验环境明确使用 **BEAR 仿真轨迹**；必要时再接一个轻量优化器或仿真器工具，但不要求大模型直接学习控制策略。

不要把项目做成以下形式：

- 纯文档问答系统
- 只有聊天框的 ChatPDF
- 只展示界面但没有评测
- 只有单次 demo，没有工具链和可复现实验

项目必须体现三层能力：

1. **大模型应用层**：RAG、Agent、Tool Calling、引用控制、回答生成
2. **数据分析层**：时序数据查询、异常检测、趋势比较、指标汇总
3. **领域层**：数据中心冷却、HVAC、节能、温控约束、运维诊断

## 2.1 关键边界与防跑偏说明

后续 AI 执行本项目时必须遵守以下边界：

1. **不要把 BEAR 伪装成真实数据中心生产数据**。正确表述是：使用 BEAR 物理仿真环境生成 HVAC 运行轨迹，并将其作为数据中心冷却优化类问题的可控代理场景。
2. **不要让 LLM 直接决定控制动作并写回环境**。LLM / Agent 只负责任务路由、证据整合和解释生成；控制动作由规则控制器、MPC-like policy、DiffFNO / Guided-DiffFNO 等工具给出。
3. **不要先做复杂前端**。第一阶段优先完成数据、工具、RAG、Agent、评测闭环，Streamlit demo 足够。
4. **不要只做文档问答**。项目必须包含 BEAR 时序分析工具和至少一个 policy / optimizer 调用接口。
5. **不要只展示单个 demo case**。必须构造 eval.jsonl，并至少完成 LLM-only、RAG、RAG + Tool Agent 三组对比。
6. **不要把 Open WebUI Pipelines 当主项目**。它只能作为可选展示壳；核心逻辑必须保留在主仓库。

## 3. 最终交付物

最终要完成以下交付物：

1. 可运行的后端服务
2. 可运行的前端 demo
3. 领域文档知识库
4. 时序数据分析工具
5. Agent 工作流
6. 评测集与实验结果
7. README 和简历可用的项目总结

## 4. 推荐技术栈

### 后端

- Python
- FastAPI
- LangGraph
- Qdrant 或 FAISS
- pandas
- numpy
- scikit-learn

### 前端

- Streamlit 作为第一版
- 后续可扩展为 React + API

### 模型

- 开发期可用商用 API 或高质量开源模型
- 需要兼容本地开源模型接口，避免强绑定单一供应商

### 评测

- 自定义 eval.jsonl
- RAG 评测指标
- 工具调用准确率评测
- 人工标注的标准答案集

## 4.1 开源项目使用原则

本项目不建议 clone 一个大而全的现成应用作为主仓库地基。主仓库应当自行创建，核心贡献必须体现在：

1. BEAR 仿真轨迹数据处理
2. HVAC 时序分析工具
3. RAG 检索与引用控制
4. LangGraph Agent 工作流
5. diffusion / control policy 调用接口
6. eval.jsonl 评测集与 baseline 对比

可以参考或局部使用以下两个开源项目，但它们不能替代主项目：

### 推荐参考项目 1: Ragas

- GitHub: `https://github.com/explodinggradients/ragas`
- 文档: `https://docs.ragas.io/`
- 用途：作为 RAG 与问答系统评测参考。
- 推荐使用方式：参考它的 quickstart 目录结构、评测脚本组织方式和指标设计，而不是把整个项目作为主仓库。

可参考的目录思路：

```text
eval/
├── datasets/
│   └── hvac_eval.jsonl
├── experiments/
│   ├── rag_baseline.json
│   ├── rag_hybrid_rerank.json
│   └── agent_tool_use.json
└── logs/
    └── tool_calls.jsonl
```

评测时至少要覆盖：

- answer correctness
- faithfulness
- context recall
- citation hit rate
- tool selection accuracy
- tool execution success rate

### 推荐参考项目 2: Open WebUI Pipelines

- GitHub: `https://github.com/open-webui/pipelines`
- 用途：作为可选 UI / pipeline 接入参考。
- 推荐使用方式：仅在需要快速接入聊天界面或演示界面时参考，不建议作为主仓库。

正确关系是：

```text
datacenter-hvac-copilot/       # 主项目，必须自己实现核心逻辑
open-webui/pipelines/          # 可选展示壳或接入参考
```

如果使用 Open WebUI Pipelines，必须保证核心逻辑仍然位于主项目中，例如：

```text
src/agent/
src/retrieval/
src/tools/
src/policies/
src/evaluation/
```

Open WebUI Pipelines 只负责把用户输入转发给主项目 API，并展示返回结果，不负责承载核心算法。

## 4.2 推荐主项目目录结构

主项目建议使用以下结构。后续 AI 搭建项目时，应优先按此结构创建文件，避免把所有逻辑写进单个脚本。

```text
datacenter-hvac-copilot/
├── README.md
├── pyproject.toml
├── .env.example
├── docs/
│   ├── project_spec.md
│   ├── system_design.md
│   └── experiment_report.md
├── data/
│   ├── bear_raw/
│   ├── bear_processed/
│   ├── documents/
│   └── eval/
│       └── hvac_eval.jsonl
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── agent/
│   ├── tools/
│   ├── policies/
│   ├── evaluation/
│   └── api/
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── export_bear_data.py
│   ├── build_index.py
│   ├── run_eval.py
│   └── run_demo.py
└── tests/
```

核心责任划分：

- `src/ingestion/`：文档解析、清洗、chunk、元数据绑定
- `src/retrieval/`：embedding、向量检索、hybrid search、rerank、引用片段返回
- `src/tools/`：BEAR 时序数据查询、异常检测、能耗统计、图表数据生成
- `src/agent/`：LangGraph 工作流、router、tool selection、report generation
- `src/policies/`：规则控制器、MPC-like policy、DiffFNO / Guided-DiffFNO 调用适配器
- `src/evaluation/`：评测集读取、baseline 运行、指标计算、实验结果保存
- `src/api/`：FastAPI 接口，供 Streamlit 或 Open WebUI Pipelines 调用

## 5. 核心架构

系统分成 5 个子模块：

1. **Document Ingestion**
   - 负责读取 PDF、Markdown、网页内容
   - 做清洗、切分、向量化、索引

2. **Retrieval Engine**
   - 负责检索相关文档片段
   - 支持 dense retrieval、hybrid search、rerank

3. **Time-Series Tool Layer**
   - 负责读取和分析运行时序数据
   - 提供查询、趋势、异常检测、对比分析等工具

4. **Agent Orchestrator**
   - 负责判断用户意图
   - 决定调用 RAG、数据分析工具、优化器工具还是组合调用

5. **Evaluation & Demo Layer**
   - 负责自动评测
   - 负责展示回答、引用、工具调用日志、图表和建议

## 6. 典型用户问题

系统至少要支持以下问题类型：

### 6.1 文档问答

- 数据中心冷却系统为什么会产生高能耗？
- ASHRAE 对温控有什么要求？
- 送风温度升高会带来什么风险？

### 6.2 数据分析

- 最近 24 小时 PUE 为什么升高？
- 哪个机房温度波动最大？
- 冷机功率和 IT 负载是否同步变化？

### 6.3 异常诊断

- 某区域温度持续高于阈值的可能原因是什么？
- 为什么风机功率上升但温度没有下降？
- 是否存在过度制冷或控制震荡？

### 6.4 优化建议

- 如果目标是节能，应该如何调整设定温度？
- 当前策略是否应该降低风机负载？
- 哪些指标说明该策略有舒适度风险？

## 7. 数据准备方案

### 7.1 文档数据

准备一个面向数据中心 HVAC 的文档库，来源可以包括：

- HVAC / 冷却系统公开论文
- ASHRAE 相关公开材料
- 数据中心能耗和热管理资料
- 设备说明文档
- 你自己的研究论文

要求：

- 文档必须可追溯
- 每个文档要保留标题、来源、发布时间、类别
- 文档切分后保留原始出处信息，方便引用

### 7.2 时序数据

准备一个结构化时序数据集。BEAR 原始状态字段与项目字段需要做一次显式映射，不能直接假设 BEAR 原生包含数据中心字段。

推荐字段如下：

```text
timestamp
scenario_id
zone_id
zone_temperature
outdoor_temp
solar_irradiance
ground_temp
internal_load
humidity
it_load
cooling_power
fan_power
chiller_power
pue
control_action
reward
alarm_flag
comfort_violation
```

本项目的主要时序数据来源为 **BEAR 仿真环境导出的轨迹数据**。如果后续需要增强鲁棒性，可以在 BEAR 数据基础上补充少量 synthetic noise 或手工扰动。

字段使用原则：

1. `zone_temperature`、`outdoor_temp`、`solar_irradiance`、`ground_temp`、`internal_load`、`control_action`、`reward`、`comfort_violation` 应优先来自 BEAR 或由 BEAR 轨迹直接计算。
2. `cooling_power`、`fan_power`、`chiller_power` 如果 BEAR 没有拆分字段，可以先统一映射为 `hvac_energy` 或 `hvac_power`，不要编造设备级字段。
3. `pue` 不是 BEAR 原生建筑控制指标。如果没有可解释计算方式，应标记为 optional derived metric，不要作为核心评测指标。
4. `humidity`、`it_load` 如果 BEAR 原始环境没有提供，可以作为 optional synthetic feature；生成方式必须写进数据说明。
5. `alarm_flag` 可以由规则派生，例如温度越界、能耗突增、控制动作震荡、comfort violation 连续出现等。

可选增强：

1. 在 BEAR 轨迹上注入传感器噪声
2. 在 BEAR 轨迹上注入缺失值
3. 构造能耗突增、温度越界、控制震荡等异常窗口
4. 如有公开建筑能耗数据，可作为补充，不作为主数据源

要求：

- 时间粒度统一
- 缺失值处理明确
- 指标定义固定
- 所有指标都要可重复计算

### 7.3 评测集

必须构造 `eval.jsonl`，每一条样本至少包含：

- 问题
- 任务类型
- 标准答案
- 需要调用的工具
- 需要引用的文档
- 期望输出格式

建议至少 100 条，最好 150 到 200 条。

## 8. 任务拆解

### Task 1: 需求冻结

目标：明确系统只做什么，不做什么。

必须确定：

- 主要用户是谁
- 主要问题类型有哪些
- 哪些问题必须调用工具
- 哪些问题只需要文档检索
- 哪些问题需要图表

输出：

- 一页需求说明
- 任务类型列表
- 评测指标列表

### Task 2: 文档入库

目标：把所有知识文档变成可检索语料。

步骤：

1. 收集 PDF / Markdown / 网页资料
2. 提取文本并清洗
3. 统一切分 chunk
4. 为每个 chunk 绑定来源元数据
5. 建立向量索引

要求：

- chunk 不要太长
- 每个 chunk 保留 `source_id`、`title`、`section`、`page` 等信息
- 保证后续回答能引用来源

### Task 3: 基础 RAG

目标：实现一个最小可用的文档问答系统。

必须具备：

- 用户问题输入
- 检索 top-k 文档片段
- 模型基于片段回答
- 回答附带引用

最低要求：

- 能跑通
- 能输出引用
- 能处理常见领域问题

### Task 4: 检索增强

目标：提升文档召回质量。

推荐顺序：

1. dense retrieval
2. hybrid search
3. reranker
4. query rewrite

要求：

- 对比不同检索方案
- 保存每次检索的证据片段
- 能解释为什么检索结果更好

### Task 5: 时序分析工具

目标：让系统能读懂运行数据，而不是只读文档。

至少实现以下工具：

```python
query_metric(metric_name, start_time, end_time, zone_id=None)
compare_period(metric_name, period_a, period_b, zone_id=None)
detect_anomaly(metric_name, window_size, threshold, zone_id=None)
compute_energy_breakdown(start_time, end_time)
plot_metric_trend(metric_name, start_time, end_time, zone_id=None)
```

要求：

- 工具输出要标准化
- 工具要返回结构化数据和可视化数据
- 工具调用日志要保存

### Task 6: Agent 工作流

目标：让系统根据问题自动选择处理路径。

建议工作流：

1. 意图识别
2. 路由选择
3. 检索文档
4. 调用时序工具
5. 如有需要，调用优化器工具
6. 汇总证据
7. 生成最终回答

推荐使用 LangGraph，而不是把所有逻辑塞进一个 prompt。

### Task 7: 优化器接口

目标：把你的研究背景接入项目，使项目更有深度。

不要求一开始就把复杂控制模型完全线上化，但至少要实现一个可调用接口。这里的目标不是让 Agent 自己训练 diffusion 模型，而是让 Agent 在合适时机调用已经训练好的模型或策略接口，拿到推荐动作或策略评估结果。

```python
run_policy_baseline(state)
run_mpc_like_policy(state)
run_diffusion_policy(state)
simulate_policy(policy_name, horizon)
evaluate_energy_comfort_tradeoff(result_a, result_b)
```

如果现成模型不能直接部署，可以先接：

- 离线结果回放
- 规则控制器
- 简化 MPC
- 仿真器评估

### Task 7.1: Agent 与 diffusion 模型的关系

Agent 不是 diffusion 模型本身，二者分工如下：

- **Agent 负责决策流程编排**：判断用户问题属于文档问答、数据分析、异常诊断还是策略推荐；决定要不要查文档、查时序、调用优化器。
- **Diffusion 模型负责策略生成**：在控制/推荐类任务里，接收状态输入，输出候选控制动作或策略建议。
- **Agent 负责调用 diffusion 接口**：当问题需要策略层输出时，Agent 调用你训练好的 diffusion policy，读取其返回的动作、价值估计或对比结果，再组织成自然语言答案。

因此，正确的关系不是“Agent 去训练 diffusion 模型”，而是：

1. 先离线训练好 diffusion / control policy
2. 再把它封装成一个可调用工具
3. Agent 根据问题决定是否调用该工具
4. Agent 将工具结果与文档证据、时序分析结果一起整合输出

Diffusion / control policy 工具接口必须返回结构化结果，建议格式如下：

```json
{
  "policy_name": "guided_diffno",
  "input_state_id": "episode_001_step_024",
  "recommended_action": [-0.2, -0.1, -0.3, -0.1, -0.2, -0.1],
  "estimated_energy": 901.3,
  "estimated_comfort_violations": 0.886,
  "mean_action_change": 0.0402,
  "baseline": "diffusion_mlp",
  "notes": "Values may come from online simulation or offline replay."
}
```

如果真实 DiffFNO / Guided-DiffFNO 模型暂时无法部署，必须使用 `offline_replay` 模式：从已保存实验结果中读取策略表现，而不是让 LLM 伪造策略输出。

### Task 8: 评测体系

目标：让项目有可量化对比，而不是只有演示。

至少评测四类指标：

1. **检索质量**
   - context recall
   - citation hit rate
   - evidence coverage

2. **回答质量**
   - correctness
   - faithfulness
   - hallucination rate

3. **工具调用质量**
   - tool selection accuracy
   - tool execution success rate
   - tool result usage quality

4. **优化建议质量**
   - energy reduction estimate
   - comfort risk estimate
   - recommendation plausibility

### Task 9: Demo 页面

目标：给面试官一个能快速理解项目价值的展示界面。

页面应该展示：

- 用户输入
- Agent 路由过程
- 调用的工具
- 检索到的证据
- 图表
- 最终结论

第一版可以用 Streamlit，后续再考虑更完整的前端。

### Task 10: README 与简历表达

目标：把项目转成可投递、可面试、可讲述的形式。

README 必须包含：

- 项目背景
- 架构图
- 核心功能
- 数据说明
- 评测方式
- 实验结果
- Demo 截图
- 启动方式

简历描述必须体现：

- 你解决了什么问题
- 用了什么方法
- 数据和评测如何构建
- 结果提升是多少

## 9. 推荐实现顺序

建议按以下顺序推进：

1. 定义任务边界
2. 整理文档数据
3. 构建基础 RAG
4. 加检索增强
5. 做时序分析工具
6. 搭 Agent 工作流
7. 接优化器接口
8. 做评测集
9. 跑实验
10. 做 demo 和文档

不要一开始就做完整前端，也不要先做很复杂的优化算法。

## 10. 里程碑

### Milestone 1: 最小可用版本

标准：

- 能回答文档问题
- 能引用来源
- 能查时序数据
- 能输出结构化结果

### Milestone 2: 可展示版本

标准：

- 有 Agent 工作流
- 有工具调用日志
- 有图表
- 有 demo 页面

### Milestone 3: 可面试版本

标准：

- 有评测集
- 有 baseline 对比
- 有实验表格
- 有清晰技术报告

## 11. 技术细节要求

### 检索

- chunk 不能太大
- 要保留元数据
- 要支持 hybrid search
- 要有 reranker

### 生成

- 回答必须尽量引用检索证据
- 对不确定的问题要明确说明不确定
- 不允许无证据编造数据

### 工具调用

- 工具输入输出要结构化
- 工具失败要有 fallback
- 工具调用日志必须记录

### 评测

- 不能只看主观感觉
- 必须有 baseline
- 必须能复现
- 必须保存实验配置

## 12. 风险与约束

### 风险 1: 项目做成普通问答

解决：

- 必须引入时序工具
- 必须引入 Agent 路由
- 必须做评测集

### 风险 2: 数据不够真实

解决：

- 文档数据和 synthetic data 结合
- 先用可控数据跑通流程
- 后续逐步替换成真实数据

### 风险 3: Agent 不稳定

解决：

- 使用可控工作流
- 减少自由规划
- 对每类问题使用固定路径

### 风险 4: 没有可讲述的结果

解决：

- 做 baseline 对比
- 做实验表格
- 保存案例分析

## 13. 验收标准

项目完成时，至少要满足：

1. 能回答 4 类问题
2. 能调用 3 类以上工具
3. 有可检索知识库
4. 有时序数据分析能力
5. 有评测集和 baseline
6. 有 demo 页面
7. 有 README 和简历描述
8. 有明确的实验结果

## 14. 一句话项目定义

> 面向 **BEAR 仿真环境** 的 LLM Agent 平台，融合领域知识检索、时序数据分析与控制优化建议，实现故障诊断、能耗归因和策略推荐，并通过评测集验证其可靠性。
