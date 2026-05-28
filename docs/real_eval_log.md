# 真实数据评测实验记录

日期：2026-05-28  
分支：`real-data-eval`  
开工快照：`d03783a snapshot before real-data eval`

## 1. 实验目标

本轮优化的目标是把项目评测从“样例文档 + deterministic dense”推进到“真实公开文档 + BGE-small-zh + FAISS”的完整链路：

- 确认真 embedding 环境可用；
- 下载并上传 5-8 篇真实公开文档；
- 用项目上传接口构建知识库，而不是手工复制文件；
- 手写真实评测子集；
- 在真实子集上完整运行 BGE + FAISS eval；
- 整理 data card、实验记录和 README 结果。

## 2. 环境与备份

原计划使用 `conda activate hvac-copilot`，但本机不存在该 conda 环境；当前 base Python 可正常导入 `sentence_transformers` 和 `faiss`：

```bash
python -c "import sentence_transformers, faiss; print('ok')"
```

开工前在 `main` 上提交快照，然后切到实验分支：

```bash
git add -A
git commit -m "snapshot before real-data eval"
git checkout -b real-data-eval
```

## 3. 真实公开文档

临时下载目录：`tmp/real_eval_documents/`

| 文件名 | 来源 | 类型 | 用途 |
| --- | --- | --- | --- |
| `doe_best_practice_guide_data_center_design_2024.pdf` | DOE | 指南/白皮书 | 数据中心节能设计、冷却系统、空气管理、PUE |
| `ashrae_tc99_power_equipment_thermal_guidelines_2016.pdf` | ASHRAE TC 9.9 | 白皮书 | 热环境、电力设备热边界、A2/A3/A4 温度等级 |
| `uptime_annual_outage_analysis_2024_exec_summary.pdf` | Uptime Institute | 行业报告摘要 | outage 趋势、原因、成本和后果 |
| `google_ml_applications_data_center_optimization.pdf` | Google Research | 论文/白皮书 | PUE 优化、运行数据建模、机器学习辅助优化 |
| `ocp_cooling_efficiency_platform_power_telemetry_2024.pdf` | Open Compute Project | 白皮书 | 平台功耗遥测、IT 负载与冷却效率协同 |
| `ashrae_standard_55_2023_fact_sheet.pdf` | ASHRAE | 标准事实表 | 热舒适、室内环境边界、Standard 55 |
| `bear_physics_principled_building_environment_arxiv_2211_14744.pdf` | arXiv | 论文 | BEAR 仿真环境与 building control 基准 |

这些文档均为公开可访问 PDF，来源与用途已整理到 [docs/data_card.md](data_card.md)。

## 4. 知识库上传与验证

本轮没有手工把文件复制进 `data/knowledge/`，而是通过项目接口上传，验证完整链路：

```bash
POST http://127.0.0.1:8000/knowledge/documents/upload
multipart/form-data: file=<PDF>
```

上传时使用：

```powershell
$env:KNOWLEDGE_EMBEDDING_PROVIDER='sentence-transformers'
$env:KNOWLEDGE_EMBEDDING_MODEL='BAAI/bge-small-zh-v1.5'
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

上传后知识库状态：

```json
{
  "document_count": 7,
  "chunk_count": 340,
  "index": {
    "available": true,
    "chunk_count": 340,
    "embedding_provider": "sentence-transformers",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "error": ""
  },
  "refresh_dirty": false
}
```

文档 ID 记录：

| 文档 | document_id | chunks |
| --- | --- | ---: |
| ASHRAE Standard 55 fact sheet | `doc_30c3960a99ab4974bc5b3681a7782854` | 4 |
| ASHRAE TC9.9 thermal guidelines | `doc_1724dac7c67a4e1fae2eba99484501aa` | 140 |
| BEAR arXiv paper | `doc_38217fc244974e90b1a91885a8fc4530` | 29 |
| DOE data center design guide | `doc_6bdd844b98a84da38de714a4130ee36c` | 92 |
| Google optimization paper | `doc_dcf932b9b0f24bc48877146c9b4bfeda` | 28 |
| OCP telemetry white paper | `doc_261fc67d9ff24ec2bfe2c84abe093509` | 28 |
| Uptime outage analysis summary | `doc_4fe6918bc4c6409aac07524baee00f01` | 19 |

验证问题：

```text
ASHRAE 推荐的数据中心 IT 设备进风温度范围是多少？请引用刚上传的真实文档。
```

结果：`/ask` 返回 200，`citations` 和 `retrieved_contexts` 命中 `ashrae_tc99_power_equipment_thermal_guidelines_2016.pdf`，说明上传、解析、FAISS 索引和 orchestrator refresh 链路可用。

## 5. 解析质量观察

抽样检查 `data/knowledge/parsed/` 与 PDF 前几页文本后，结论如下：

- 大多数 PDF 能被 `pypdf` 抽取出可用正文，足以支撑第一轮 RAG 和 eval；
- ASHRAE TC9.9、DOE、OCP 含目录、封面、表格、页眉页脚，会带来少量 chunk 噪声；
- Google PDF 存在不可见空格/排版空格；
- BEAR 论文为双栏论文，可能有轻微行序混杂；
- Uptime 摘要含版权页眉，控制台输出需要 UTF-8。

后续提升优先级：去页眉页脚、跳过封面和免责声明、清理特殊空格、对表格页做更稳定的 chunk 标注。

## 6. 假 dense 与真 BGE 对照

旧 baseline 来自默认 deterministic dense，不代表真实 embedding。旧数字：

| Mode | 假 dense citation/context | expected keyword | correctness proxy |
| --- | ---: | ---: | ---: |
| `rag_dense` | 0.508 | 0.261 | 0.273 |
| `rag_hybrid` BM25 | 0.523 | 0.295 | 0.344 |
| `hybrid_rrf` | 0.569 | 0.292 | 0.350 |

随后使用 BGE-small-zh + FAISS 在 108 条合成/样例集上重跑。第一次运行被 `data/knowledge/` 的真实知识库污染，因为 108 条样例集的 gold document IDs 指向 `data/documents/` 的 demo 文档；真实知识库启用后，citation/context 与 gold 不再对齐。因此公平对照时显式隔离真实知识库：

```powershell
$env:KNOWLEDGE_BASE_DIR='tmp/empty_knowledge_for_demo_eval'
$env:KNOWLEDGE_EMBEDDING_PROVIDER='sentence-transformers'
$env:KNOWLEDGE_EMBEDDING_MODEL='BAAI/bge-small-zh-v1.5'
python scripts/run_eval.py `
  --output data/eval/real_bge_demo_docs/baseline_predictions.jsonl `
  --comparison-output data/eval/real_bge_demo_docs/baseline_comparison.json `
  --report-output data/eval/real_bge_demo_docs/experiment_report.md `
  --human-review-sample-output data/eval/real_bge_demo_docs/human_review_sample.jsonl `
  --human-review-annotations-output data/eval/real_bge_demo_docs/human_review_annotations.jsonl `
  --dense-provider sentence-transformers `
  --dense-backend faiss `
  --dense-model BAAI/bge-small-zh-v1.5
```

公平对照结果：

| Mode | 假 dense citation/context | 真 BGE citation/context | 变化 |
| --- | ---: | ---: | ---: |
| `rag_dense` | 0.508 | 0.708 | +0.200 |
| `rag_hybrid` BM25 | 0.523 | 0.523 | +0.000 |
| `hybrid_rrf` | 0.569 | 0.708 | +0.138 |

结论：同一 108 条样例集和同一 demo 文档语料上，BGE-small-zh + FAISS 明显提升 dense 与 RRF 的 citation/context recall；BM25 不依赖 embedding，因此保持不变。

## 7. 真实评测子集

新建真实子集：

```text
data/eval/real_eval.jsonl
```

初版样本共 24 条，后来根据“不能让系统轻松满分”的评审意见重构为 50 条。重构目标是增加真实难度梯度、相似文档干扰、多文档整合和文档外边界题，让检索 eval 能暴露上限和弱点。

| task_type | 数量 | 说明 |
| --- | ---: | --- |
| `document_qa` | 30 | 覆盖基础召回、语义整合、相似文档干扰、跨文档整合和文档外边界 |
| `timeseries_query` | 7 | 覆盖 BEAR rollout 行数/时间范围、温度统计、compare_period、plot_metric_trend、时间窗口边界 |
| `anomaly_diagnosis` | 6 | 覆盖异常候选、comfort violation、低温仿真/数据质量边界、检测器参数敏感性 |
| `policy_recommendation` | 7 | 覆盖 rule_based_policy、边界声明、多步 query_metric / detect_anomaly -> policy、文档约束下的策略解释 |

文档问答内部梯度：

| 类别 | 数量 | 目的 |
| --- | ---: | --- |
| 基础召回 | 6 | 保留少量明确文档事实题，作为基础能力下限 |
| 语义题 | 8 | 问法与文档措辞不完全一致，考查 semantic retrieval |
| 干扰题 | 7 | 多篇相似文档表面相关，但只有一篇或少数几篇是主证据 |
| 跨文档题 | 5 | 要求同时命中 2-3 篇文档，拉开单路检索和 RRF |
| 边界题 | 4 | 文档中没有直接答案，要求说明不能编造或不能越界 |

标注原则：

- gold answer 来自真实 PDF、当前 BEAR CSV 或项目工具输出；
- 数值类答案保留容差表达，避免过度依赖显示精度；
- 不把 BEAR rollout 表述成生产遥测；
- 对 `PUE`、`humidity`、`it_load` 等当前 CSV 不存在的字段明确要求“不编造”；
- 异常数量高的问题写成“当前检测器较敏感”的诚实边界，而不是生产事故结论。

格式验证：

```bash
python -c "from src.evaluation.dataset import load_eval_dataset; rs=load_eval_dataset('data/eval/real_eval.jsonl'); print('ok', len(rs))"
```

输出：

```text
ok 50
```

新增质量门槛测试：

```bash
pytest tests/test_real_eval_dataset.py -q
```

该测试约束真实子集必须为 50 条，任务分布为 30/7/6/7，至少包含 6 条干扰题、3 条边界题、5 条跨文档题、8 条多文档 required_documents 样本，并且所有记录可被 `EvalRecord` 加载。

## 8. 真实子集完整 BGE 评测

为了让 `run_baseline_comparison()` 在真实知识库上也能构造 `rag_dense`、`rag_hybrid`、`hybrid_rrf`，补了一个兼容修复：

- [src/api/demo_factory.py](../src/api/demo_factory.py)：让 `_LazyKnowledgeRetriever` 从 `data/knowledge/faiss/chunks.jsonl` 暴露 `.chunks`，并转换成 retrieval baseline 使用的 `DocumentChunk`；
- [tests/test_knowledge_api.py](../tests/test_knowledge_api.py)：新增回归测试，确认上传知识库后 retriever 暴露 chunks。

TDD 验证：

```bash
pytest tests/test_knowledge_api.py::test_uploaded_knowledge_retriever_exposes_chunks_for_eval_comparison -q
```

先失败于：

```text
AttributeError: '_LazyKnowledgeRetriever' object has no attribute 'chunks'
```

修复后通过：

```text
1 passed
```

完整真实子集 eval 第一次因 HuggingFace 可选配置文件网络请求超时失败。模型权重本地缓存可用，使用离线模式后正常运行：

```powershell
$env:KNOWLEDGE_EMBEDDING_PROVIDER='sentence-transformers'
$env:KNOWLEDGE_EMBEDDING_MODEL='BAAI/bge-small-zh-v1.5'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Remove-Item Env:KNOWLEDGE_BASE_DIR -ErrorAction SilentlyContinue
python scripts/run_eval.py `
  --eval-path data/eval/real_eval.jsonl `
  --output data/eval/real_eval_bge/baseline_predictions.jsonl `
  --comparison-output data/eval/real_eval_bge/baseline_comparison.json `
  --report-output data/eval/real_eval_bge/experiment_report.md `
  --human-review-sample-output data/eval/real_eval_bge/human_review_sample.jsonl `
  --human-review-annotations-output data/eval/real_eval_bge/human_review_annotations.jsonl `
  --dense-provider sentence-transformers `
  --dense-backend faiss `
  --dense-model BAAI/bge-small-zh-v1.5
```

输出产物：

- `data/eval/real_eval_bge/baseline_predictions.jsonl`
- `data/eval/real_eval_bge/baseline_comparison.json`
- `data/eval/real_eval_bge/experiment_report.md`
- `data/eval/real_eval_bge/human_review_sample.jsonl`
- `data/eval/real_eval_bge/human_review_annotations.jsonl`

## 9. 真实子集结果

真实公开文档 + 50 条真实手写子集，BGE-small-zh + FAISS：

| Mode | Citation / Context | Recall@10 | MRR@10 | nDCG@10 | Expected Keyword | Tool Select | Tool Success | Evidence | Correctness | Faithfulness | Hallucination Proxy | Grounding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rag_dense` | 0.875 | 0.932 | 0.719 | 0.753 | 0.451 | 0.000 | 0.000 | 1.000 | 0.210 | 0.210 | 0.000 | 0.000 |
| `rag_hybrid` BM25 | 0.812 | 0.885 | 0.922 | 0.885 | 0.484 | 0.000 | 0.000 | 0.760 | 0.222 | 0.222 | 0.000 | 0.000 |
| `hybrid_rrf` | 0.969 | 0.990 | 0.896 | 0.912 | 0.513 | 0.000 | 0.000 | 1.000 | 0.232 | 0.232 | 0.000 | 0.000 |
| `rag_tool_agent` | 0.562 | 0.667 | 0.651 | 0.622 | 0.643 | 0.850 | 1.000 | 1.000 | 0.703 | 0.690 | 0.042 | 0.938 |
| `langgraph_tool_agent` | 0.562 | 0.667 | 0.651 | 0.622 | 0.665 | 1.000 | 1.000 | 1.000 | 0.727 | 0.713 | 0.042 | 0.938 |
| `react_agent` | 0.562 | 0.667 | 0.651 | 0.622 | 0.648 | 0.900 | 1.000 | 1.000 | 0.713 | 0.700 | 0.042 | 0.938 |

关键观察：

- `hybrid_rrf` 在重构后真实文档子集上 Citation / Context 为 0.969、Recall@10 为 0.990，说明融合检索能覆盖绝大多数 required documents；
- `hybrid_rrf` 的 MRR@10 / nDCG@10 为 0.896 / 0.912，排序质量也高于纯 dense 的 0.719 / 0.753；
- BM25 在真实文档问答上仍较强，说明不少题目保留了术语线索；dense 单路在相似文档干扰和多文档命中上明显不足；
- 真实子集里的时序、异常和策略题需要结构化工具证据，纯 RAG correctness proxy 明显偏低；
- `langgraph_tool_agent` 的 tool selection、tool success、evidence coverage 均为 1.000，整体 correctness proxy 为 0.727。

## 10. 合成 108 vs 真实 50

| Mode | 合成 citation/context | 真实 citation/context | 合成 tool select | 真实 tool select | 合成 correctness | 真实 correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rag_dense` | 0.800 | 0.875 | 0.000 | 0.000 | 0.582 | 0.210 |
| `rag_hybrid` BM25 | 0.646 | 0.812 | 0.000 | 0.000 | 0.413 | 0.222 |
| `hybrid_rrf` | 0.815 | 0.969 | 0.000 | 0.000 | 0.601 | 0.232 |
| `rag_tool_agent` | 0.338 | 0.562 | 0.882 | 0.850 | 0.541 | 0.703 |
| `langgraph_tool_agent` | 0.338 | 0.562 | 0.882 | 1.000 | 0.541 | 0.727 |
| `react_agent` | 0.338 | 0.562 | 0.956 | 0.900 | 0.582 | 0.713 |

阶段结论：

> 重构后的 50 条真实子集更适合作为 README 和简历里的主推证据：它验证了真实上传文档、BGE + FAISS 知识库、RRF 融合检索和工具 agent 的完整链路，同时通过相似文档干扰、跨文档整合和文档外边界题暴露系统上限。合成 108 条仍用于规模化回归；真实 50 条用于证明系统能处理真实公开文档和真实工具链任务，并且能诚实呈现弱点。

## 11. 用户后续如何上传文档

方式一：Streamlit 页面上传。

```bash
streamlit run app/streamlit_app.py
```

打开页面后进入 `Knowledge Base` tab，选择 PDF / DOCX / TXT / MD 文件，点击 `Index document`。

方式二：API 上传。

```bash
curl -X POST http://127.0.0.1:8000/knowledge/documents/upload \
  -F "file=@你的文档.pdf"
```

PowerShell：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/knowledge/documents/upload `
  -Method Post `
  -Form @{ file = Get-Item "你的文档.pdf" }
```

上传后检查：

```bash
curl http://127.0.0.1:8000/knowledge/status
curl http://127.0.0.1:8000/knowledge/documents
```

注意：不要手工把文件复制进 `data/knowledge/`。正确链路是“上传接口 -> 解析 -> SQLite metadata/chunks -> FAISS 重建 -> orchestrator refresh”。
