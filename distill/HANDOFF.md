# Handoff — Planner 蒸馏 阶段 1（数据）+ 阶段 2（SFT）+ 阶段 4（评测）+ A2/A1 优化

**日期**：2026-07-04（末次更新：2026-07-05，A1 补数据 + 重训 + 四方评测完成后同步）
**范围**：`distillation_plan.md` 的阶段 1（精标数据）、阶段 2（SFT 训练脚本 + 实际训练）、阶段 4（接回评测，四方对比），外加 SFT 操作手册 + **A2（time_window 归一化，推理层）+ A1（补 100 条 gold，治本，已重训重评）**。阶段 3（DPO）**未开始**，A1 已完成故可启动。

**当前 git 状态**：A1 数据已提交（`54bf21d` 增加 timewindow 训练数据：gold 600→700、train/val 630/70、`build_a1_additions.py`、测试）。**工作区还有未提交改动**：
- `distill/HANDOFF.md`（本文件，A1 后同步）
- `docs/distillation_report.md`（A1 四方报告）
- `distill/checkpoints/sft-qwen1.5b/sft_train_card.json`（新模型 train card：train 630/val 70）
- 未跟踪：`distill/checkpoints/sft-qwen1.5b-600base/`（旧版 600 条模型备份，gitignore 会挡大文件，仅其 train_card 可进版本库）
- ⚠️ 直推 main 会被 auto 模式拦截；接手 `git add -A && git commit` 后，push 可能需要用户本地 `! git push`。

相关 commit：
- `36532e1` 重构 planner（抽出 `build_planner_messages` / `serialize_plan_steps`）
- `83aa39d` 阶段 1+2：600 条精标数据 + SFT 脚本 + 测试
- `4594405` / `0b07d5d` SFT 操作手册 + 技术方案 + gitignore 大权重
- `6b2783e` 适配服务器训练配置（TRL 1.7.1 API 迁移，见 §3.2）
- `69b97ec` 阶段4：评测脚本就绪 + deterministic 基线
- `70b927d` 阶段4加速：批量 GPU 推理 + 合并 adapter + `--fast`
- `0fa7e18` A2：time_window 归一化 + 阶段4三方评测报告
- `54bf21d` **A1：补 100 条 gold（自然语言时间窗直接标合法词表 + 工具消歧 + 多步复合）**

**训练结果**：
- 旧版（600 条，A2）：val legality=100% / exact_match=96.67%（n=60）。
- **新版（700 条，A1）：val legality=100% / exact_match=97.14%（n=70）**，约 2 分钟。产物 `distill/checkpoints/sft-qwen1.5b/`（LoRA adapter，gitignore 排除大文件，仅 `sft_train_card.json` 进版本库）。旧版留档 `distill/checkpoints/sft-qwen1.5b-600base/`。

**阶段4评测结果（最终，A1 补数据后）**：四方对比闭环完成——

| planner | step_acc | order_acc | req_recall | policy_final | 非fallback | 延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| deterministic 基线 | 14.0% | 14.0% | 56.3% | 72.7% | 100/100 | ~0s |
| 蒸馏 1.5B (A2, 600条) | 68.0% | 68.0% | 85.5% | 93.5% | 74/100 | 0.8s |
| **蒸馏 1.5B (A1, 700条)** | **84.0%** | **84.0%** | **93.0%** | **96.1%** | **90/100** | 0.8s |
| DeepSeek 云端 planner（对照） | 63.0% | 63.0% | 85.7% | 93.5% | 72/100 | 18s |

> ⚠️ **术语澄清（重要，勿写错）**：最终模型是**纯 SFT，训练标签 100% 人工手标 gold**（数据卡 `teacher: hand_labeled`）。DeepSeek 在本项目里是**阶段4 评测的对照基线**（线上另一条 planner 路线），**不是**训练数据的来源。`build_sft_data.py` 虽支持"用线上 planner / DeepSeek 当 teacher 生成 SFT 数据"的路径，但**最终模型没用它**。因此对外表述应说"把 planner 路由能力 SFT 进小模型"，**不要说"蒸馏了 DeepSeek"**——那会被理解成用 DeepSeek 输出当标签。下文"教师"一词若出现，均指评测对照，非训练教师。

**核心结论**：A1 补 100 条 gold 后重训，蒸馏 1.5B 达到 **84% step_acc**，是规则基线（14%）的 6 倍，**反超作为对照的云端 DeepSeek planner**（63%）21 个点，本地运行、延迟仅其 1/23、零 API 成本。反超的原因是小模型用 700 条经守卫校验的 gold 专门训练、天生产出合规计划，而未针对本项目 schema 微调的 DeepSeek 大量输出被守卫拒——不是小模型"更聪明"。A1 相对 A2（68%）+16 个点，逐条看修复了旧版 26 条 fallback 中的 20 条（新引入 4 条、两版都失败 6 条），净减 16 条回退（26→10）。

**A2 优化（推理层，不用重训）**：解析层加 `_normalize_time_window`，把自然语言时间窗（`past 7 days` / `last month` / `7d`）映射回守卫词表。蒸馏 47%→68%，DeepSeek 对照 21%→63%。详见 §9。

**A1 优化（补数据，治本）**：补 100 条经守卫校验的 gold（自然语言时间窗直接标成合法小时词表 + 8 工具消歧 + 多步复合），把归一化/工具选择能力固化进模型权重。gold 600→700，train/val 630/70，val exact_match 96.67%→97.14%。残余 10 条 fallback 全是最难的 3 步复合任务（多指标同图、绝对时段 10PM-6AM、episode/多文档引用），属阶段3 DPO 目标。详见 §10。

**下一步**：**提交未推送改动 → 阶段3 DPO**（A1 已完成，治本数据缺陷已补）。

---

## 1. 现在处于什么位置

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 1 数据构造 | ✅ 完成（`83aa39d` 600 条 + `54bf21d` A1 补至 700 条） | `distill/data/gold_labeled.jsonl`（700 条）+ train/val（630/70）+ data card |
| 2 SFT 脚本 | ✅ 完成（`83aa39d`，TRL 适配 `6b2783e`） | `distill/train_sft.py` |
| 2 SFT 训练 | ✅ 已跑通两版（旧 600 exact 96.67% / 新 700 exact 97.14%） | `distill/checkpoints/sft-qwen1.5b/`（新）+ `-600base/`（旧备份）（大文件 gitignore） |
| — 操作手册 / 技术方案 | ✅ 完成（`4594405` / `0b07d5d`） | `SFT_训练操作手册.pdf` / `SFT_技术方案.md` |
| 4 接回评测 | ✅ **完成**：四方对比闭环，蒸馏 A1 84% 反超 DeepSeek 对照 63% | `distilled_planner.py` / `eval_planners.py` / `docs/distillation_report.md` |
| — A2 time_window 归一化 | ✅ 完成（推理层，不用重训） | `_normalize_time_window`（`src/agent/planner.py`）+ 6 单测 |
| — A1 补数据（治本） | ✅ **完成**（`54bf21d`，已重训重评，step_acc 68%→84%） | `build_a1_additions.py` + 100 条 gold |
| — 报告/HANDOFF/train_card 提交 | ⚠️ **待提交**：本文件 + 报告 + 新 train_card | — |
| 3 DPO | ⛔ 未开始（A1 已完成，可启动，见 §6/§10） | — |

---

## 2. 各阶段做了什么

### 2.1 精标数据：300 → 600 → 700 条
- `distill/data/gold_labeled.jsonl` 从 300 扩到 600（阶段1），A1 再补 100 条到 **700 条**手标 `{question, steps}`。
- **全部 700/700 通过 `validate_plan_steps` 校验**（线上同一个 guard）。
- 步数分布（700 条）：1步 308 (44%) / 2步 187 (27%) / 3步 170 (24%) / 4步 35 (5%)；**多步（≥2）占 56%**。
- 覆盖：**12 个工具全覆盖**（含 `rag_retrieval`）；14 个 metric_name **全部是 `bear_rollout.csv` 的真实列**（零脏数据）；中英文混合、口语化、抗干扰、多轮指代。
- 运行 `python -m distill.build_gold_sft` 生成 `gold_sft_train.jsonl`（630）/ `gold_sft_val.jsonl`（70）/ `gold_sft_data_card.json`。

### 2.2 SFT 训练脚本 `distill/train_sft.py`
- 基座默认 **Qwen2.5-1.5B-Instruct**，`--model` 可配置（换 3B/8B 无需改代码）。
- **QLoRA 4-bit**（单卡 12–16G 可跑），`--no-quantize` 切全精度 LoRA。
- **数据用 TRL prompt/completion 格式**：SFTTrainer 自动套模型 chat template，训练格式 = 推理格式。
- **completion-only loss**：`SFTConfig(completion_only_loss=True)` 只对 assistant 输出算 loss（TRL ≥1.x 内置，取代旧的 `DataCollatorForCompletionOnlyLM`，见 §3.2）。
- **验收指标复用线上代码**：训练后用 `_decision_from_llm_payload`（线上 planner 解析 LLM 输出的同一函数）测 val 集**合法率**和 **route 精确匹配率**，写进 `sft_train_card.json`。
- `pyproject.toml` 新增 `[train]` extra 记录训练依赖。

### 2.3 测试
- `tests/test_distill_gold.py`，含硬断言"线上 gold 全部 100% 合法"（数字随 gold 扩充更新：现为 **700 条**），防数据劣化。
- 蒸馏 + route planner 全部相关测试通过（33 个）。

### 2.4 A1 补数据（`54bf21d`，治本优化）
- 针对阶段4 残余 fallback 的两个根因（time_window 无法归一化 + tool 选错），用 `distill/build_a1_additions.py` 补了 **100 条**经守卫校验的 gold：
  - **时间窗归一化样本**：question 用自然语言窗（`过去一周` / `last month` / `past 3 days`），label 直接标成合法小时词表（`last_168_hours` 等，天/周/月折算成小时），让模型**自身**学会归一化，不再只靠 A2 推理层兜底。
  - **工具消歧样本**：把 `timeseries_query` 下 8 个工具的边界讲清（单值→`query_metric`、两期→`compare_period`、画图→`plot_metric_trend`、能耗→`compute_energy_breakdown`、热点→`zone_hotspot_rank`、效率→`cooling_efficiency_summary`、审计→`control_action_audit`、质量→`data_quality_check`）。
  - **多步复合样本**：因评测集全是复合任务，补了大量 2-4 步、含自然语言窗的复合任务，在真实失败条件下演练归一化。
- builder 内置双重校验：每行过 `validate_plan_steps`，且**授权步数 == 校验后步数**（守卫按 route 去重，两个同 route 步骤会静默丢工具——此检查防止 label 与问题不符）。100/100 通过守卫、通过线上解析器零 fallback、无重复 ID/问题。
- 重训后 step_acc 68%→84%，详见头部与 §10。

---

## 3. ⚠️ 接手前必读的风险与前提

### 3.1 distill 脚本依赖的 planner 函数（已提交，风险已消除）
`distill/build_gold_sft.py` 和 `distill/build_sft_data.py` 依赖：
- `src.agent.planner.build_planner_messages`
- `src.agent.planner.serialize_plan_steps`

**这两个函数已在 commit `36532e1`（"重构 planner：抽出可复用的消息构建与计划序列化函数"）提交进 `src/agent/planner.py`**（现位于第 338、363 行）。阶段 1+2 的全部产物（gold 数据、脚本、测试）也已在 `83aa39d` 提交，git 现为干净稳定基线。**原先"函数只在未提交工作区"的风险已不存在。**

**如需自检**（无害）：`grep -n "def build_planner_messages\|def serialize_plan_steps" src/agent/planner.py` 应各命中一次。

### 3.2 训练脚本的 TRL 版本适配（已解决，记录备查）
服务器实装 **TRL 1.7.1 + transformers 5.13.0**，与脚本初版假设的旧 API 不同，已在 `6b2783e` 完成迁移。若日后换环境重现类似报错，对照下表改：

| 旧 API（已弃用） | 新 API（TRL 1.7.1） |
| --- | --- |
| `from trl import DataCollatorForCompletionOnlyLM` | 删除；改用 `SFTConfig(completion_only_loss=True)` |
| 数据集拼成单一 `text` 字段 | `{prompt: [消息...], completion: [{role:assistant,...}]}` 两字段 |
| `SFTConfig(max_seq_length=...)` | `SFTConfig(max_length=...)` |
| `SFTConfig(dataset_text_field="text")` | 删除（prompt/completion 格式不需要） |
| `SFTTrainer(tokenizer=...)` | `SFTTrainer(processing_class=...)` |

排错方法（可复用）：`python -c "from trl import SFTConfig; import inspect; print(sorted(inspect.signature(SFTConfig.__init__).parameters))"` 可列出当前版本支持的全部参数名，据此对齐。

### 3.3 预先存在的测试失败（与本次工作无关）
`tests/test_knowledge_indexer.py`(7) 和 `tests/test_memory_retriever.py`(1) 失败，原因是**未安装 `sentence-transformers`（`[dense]` extra）**，报错信息里已明示。**不是本次改动引入的**，跑 `pip install -e '.[dense]'` 或忽略即可。

---

## 4. 已验证清单

**数据与脚本（本地）**：
- ✅ 700 条 gold 全部过 `validate_plan_steps`（含 A1 补的 100 条）
- ✅ ID 唯一、问题唯一、train/val（630/70）无 ID 泄漏、train+val 覆盖全部
- ✅ 全部 700 条 completion 可被线上 `_decision_from_llm_payload` 解析（含 markdown 围栏兼容）；A1 的 100 条额外验证零 fallback
- ✅ A1 的 100 条无 route 塌缩（授权步数==校验后步数）、无 policy 错位、time_window 全合法、metric 全是真实列
- ✅ 14 个 metric_name 全是真实数据列
- ✅ `train_sft.py` 可 import、lint 通过、依赖缺失时给可操作提示、`load_records` 正常
- ✅ 蒸馏 + route planner 测试全绿（33 个）

**训练（GPU，2026-07-05）**：
- ✅ 旧版（600）：train_loss 0.60→0.058，eval_loss 0.144→0.108（无过拟合反弹），val legality 100% / exact_match 96.67%（n=60）
- ✅ **新版（700，A1）：val legality 100% / exact_match 97.14%（n=70）**，约 2 分钟
- ✅ TRL 1.7.1 + transformers 5.13.0 下 API 兼容（`6b2783e` 迁移后）

**评测（四方，2026-07-05）**：
- ✅ deterministic 14% / 蒸馏A2 68% / **蒸馏A1 84%** / DeepSeek 对照 63%（step_acc）
- ✅ 新旧 adapter 用同一套 `eval_planners score` 对比，可比

**产物留存**：
- ✅ 新版 `distill/checkpoints/sft-qwen1.5b/`（adapter 37M + tokenizer + train_card）
- ✅ 旧版备份 `distill/checkpoints/sft-qwen1.5b-600base/`
- ✅ 精简备份包 `distill/checkpoints/sft-qwen1.5b-a1.tar.gz`（30M，已剔除训练 checkpoint/optimizer 冗余）

---

## 5. 阶段 2 复现命令（已跑通）

在 AutoDL GPU（基础镜像 PyTorch + CUDA）上，实际成功的流程：

```bash
# 1. 拉代码（GitHub 慢时先 source /etc/network_turbo）
cd ~/autodl-tmp && git clone <repo> && cd DataCenter-HVAC-Copilot

# 2. 装训练依赖
pip install -e '.[train]'

# 3. 模型下载走国内镜像；注意不要同时开学术加速代理（二者冲突会卡 0）
unset http_proxy https_proxy
export HF_ENDPOINT=https://hf-mirror.com

# 4. 跑 SFT（约 2 分钟）
python -m distill.train_sft \
    --train distill/data/gold_sft_train.jsonl \
    --val   distill/data/gold_sft_val.jsonl \
    --output distill/checkpoints/sft-qwen1.5b
```

结束会打印 `[eval] val legality=100.00% exact_match=97.14% (n=70)`（700 条数据版；旧 600 条版是 96.67% / n=60），并写 `sft_train_card.json`。

**踩过的坑（已记录）**：
1. TRL API 版本不匹配 → 见 §3.2。
2. 模型下载速度为 0 → 学术加速代理（`http_proxy`）与国内镜像 `hf-mirror` 冲突；用镜像时先 `unset http_proxy https_proxy`。
3. **xet 下载 401 Unauthorized** → 新版 `huggingface_hub` 默认用 xet 后端，绕过 `hf-mirror` 直连 `cas-server.xethub.hf.co` 导致 401。解决：`export HF_HUB_DISABLE_XET=1`（仍不行就 `pip uninstall -y hf_xet hf-xet`，或先用 `huggingface-cli download` 预拉到本地再 `--model <本地目录>` 训练）。

**下一步**：产物下载留存后关机停计费。是否继续阶段 3/4 见 §6。

---

## 6. 阶段 3（DPO）预备（可启动，A1 已完成）

计划见 `distillation_plan.md` 阶段 3。SFT（含 A1 重训）已在云端跑通，可启动 DPO：
- DPO 以 SFT 模型为起点（policy）+ SFT 冻结副本为 reference——起点用新版 `distill/checkpoints/sft-qwen1.5b/`（A1，84% step_acc）。
- DPO 偏好数据的 `rejected` 最好用 **SFT A1 版模型的真实错误**构造（那 10 条 fallback + exact_match 未命中样本），比人工编造的负例更真实。A1 已把数据缺陷补掉，DPO 不再跟同一缺陷较劲。
- 待写脚本：`distill/build_dpo_data.py`（用项目 guard 信号构造偏好对）+ `distill/train_dpo.py`（TRL `DPOTrainer`）。

---

## 7. 关键文件索引

| 文件 | 作用 |
| --- | --- |
| `distill/data/gold_labeled.jsonl` | 700 条精标源数据（阶段1 600 + A1 100，可继续扩充） |
| `distill/augment_questions.py` | 问题扩充（生成 `data/questions.jsonl`） |
| `distill/build_sft_data.py` | 备选：用线上 planner/DeepSeek 当 teacher 生成 SFT 数据（`data/sft_*.jsonl`）——**最终模型未用此路径**，用的是手标 gold |
| `distill/build_gold_sft.py` | gold → train/val + data card（依赖 §3.1 函数） |
| `distill/build_a1_additions.py` | **A1：把 100 条新样本过守卫校验后增量追加进 gold（幂等 + 双重校验）** |
| `distill/train_sft.py` | 阶段 2 SFT 训练 |
| `distill/distilled_planner.py` | 阶段 4：蒸馏模型的 RoutePlanner 封装 |
| `distill/eval_planners.py` | 阶段 4：predict/score 评测入口（`--fast` GPU 批量 / `--concurrency` API 并发） |
| `distill/make_manual_pdf.py` | 生成 SFT 训练操作手册 PDF |
| `distill/SFT_训练操作手册.pdf` / `_v2.pdf` | 训练操作手册（面向操作，v2 为最新版） |
| `distill/SFT_技术方案.md` | SFT 技术方案手册（面向理解，讲原理与设计取舍） |
| `tests/test_distill_gold.py` | 保护 gold 数据 100% 合法（硬断言 700 条） |
| `tests/test_distill_augment.py` | 问题扩充测试 |
| `tests/test_distill_sft_data.py` | SFT 数据构造测试 |
| `distillation_plan.md` | 五阶段总计划 |
| `pyproject.toml` | 新增 `[train]` extra |

---

## 8. 阶段 4（接回评测）——✅ 已完成（四方对比闭环）

目标：把蒸馏 SFT 模型接进现有评测，与 `deterministic`（规则基线）和 `env`（DeepSeek 云端 planner，作**评测对照**、非训练教师）做**四方对比**（deterministic / 蒸馏A2旧 / 蒸馏A1新 / DeepSeek对照），产出对比表 + `docs/distillation_report.md`。这是整条蒸馏链的闭环，**已完成**（结果见头部表格）。以下步骤留作复现/重跑参考。

### 8.1 脚本与产物
- **`distill/distilled_planner.py`**：`DistilledRoutePlanner`，实现 `RoutePlanner` 协议。加载基座 + LoRA adapter，复用线上同一套 `build_planner_messages`（构造 prompt）和 `_decision_from_llm_payload`（解析+校验），失败回退 deterministic。重依赖惰性加载，无 GPU 也能 import。
- **`distill/eval_planners.py`**：评测入口，`predict` / `score` 两段解耦。`predict` 跑一个 planner 产出 `predictions.jsonl`；`score` 读一或多个 predictions 出指标对比表（纯 Python，复用 `src/evaluation/metrics.py`）。
- **现有预测文件**（`distill/data/`，均 gitignore）：`eval_pred_deterministic.jsonl`（基线）/ `eval_pred_distilledA2.jsonl`（旧600）/ `eval_pred_distilled_a1.jsonl`（新700）/ `eval_pred_teacher.jsonl`（DeepSeek）。A2 前的对照快照 `eval_pred_distilled.jsonl` / `eval_pred_teacher_preA2.jsonl` / `eval_pred_teacher_t20.jsonl` 仅留档。

### 8.2 复现命令（在 GPU 机器上）

**环境**：GPU 机器（AutoDL，参照 `SFT_训练操作手册.pdf`）+ DeepSeek API key（仅重跑教师那列时需要）。模型下载见 §5（`unset http_proxy` + `HF_ENDPOINT=hf-mirror` + `HF_HUB_DISABLE_XET=1`）。LoRA adapter：`distill/checkpoints/sft-qwen1.5b/` 未进 git，需上传本地已训好的目录，或在服务器重训一次。

```bash
# 蒸馏 A1（新版，GPU 批量解码，~1 分钟）
python -m distill.eval_planners predict --planner distilled \
    --adapter distill/checkpoints/sft-qwen1.5b \
    --out distill/data/eval_pred_distilled_a1.jsonl \
    --fast --batch-size 16 --max-new-tokens 160

# DeepSeek 教师（需 .env 配好，真实调 API，线程池并发压缩墙钟时间）
python -m distill.eval_planners predict --planner env \
    --out distill/data/eval_pred_teacher.jsonl --concurrency 8

# deterministic 基线（已有，重跑也可）
python -m distill.eval_planners predict --planner deterministic \
    --out distill/data/eval_pred_deterministic.jsonl
```

**四方对比 + 出报告**：
```bash
python -m distill.eval_planners score \
    distill/data/eval_pred_deterministic.jsonl \
    distill/data/eval_pred_distilledA2.jsonl \
    distill/data/eval_pred_distilled_a1.jsonl \
    distill/data/eval_pred_teacher.jsonl \
    --report docs/distillation_report.md
```

### 8.3 已完成的收尾
- ✅ `docs/distillation_report.md` 已更新为 A1 四方对比领先、A2 降为附录留档。
- ✅ 残余局限如实记录（10 条 fallback 全是最难 3 步复合任务，见 §10），不粉饰。
- ⏳ 关键数字回写 README / 简历（可参照 `SFT_技术方案.md` 措辞）：**蒸馏 1.5B 达 84% step_acc，6 倍于规则基线，反超作为对照的 DeepSeek 云端 planner 21 个点，本地延迟仅 1/23**。（措辞用"SFT/反超对照"，勿写"蒸馏了 DeepSeek"，见头部术语澄清）

### 8.4 已知注意点
- `env` planner 由 `build_route_planner_from_env()` 构造，依赖 `.env` 的 `LANGGRAPH_PLANNER_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`；配错会静默退回 deterministic（表现为 `env` 那列 = deterministic）。跑完看 `predictions` 里的 `planner` 字段确认是否真的走了 `llm:deepseek:...`。
- `predict` 对每条不传 `task_type`，让 planner 从问题自行路由（compound 任务本来也没单一 task_type）。
- 所有 predictions 文件都基于同一 `--eval-path`（默认 `data/eval/compound_task_eval.jsonl`，100 条），score 时才可比。
- **加速开关**：蒸馏用 `--fast --batch-size 16 --max-new-tokens 160`（批量 GPU 解码 + 合并 adapter）；教师/API 用 `--concurrency 8`（线程池并发，100 条从 ~40 分钟压到 ~3 分钟）。**蒸馏推理需权重齐全的 GPU 机器**（本地缓存曾没下全导致奇慢）。
- **新旧 adapter 对比**：想跑新旧蒸馏对照，把 `--adapter` 分别指向 `distill/checkpoints/sft-qwen1.5b`（新700）和 `sft-qwen1.5b-600base`（旧600），输出到不同文件再一起 score。

---

## 9. A2：time_window 归一化（已完成，推理层优化）

> 本节"教师"均指评测对照的 DeepSeek 云端 planner（非训练教师，见头部术语澄清）。

**问题**：阶段4 第一版评测发现，模型（DeepSeek 对照 + 蒸馏）大量输出自然语言时间窗（`"past 7 days"`、`"last month"`、`"7d"`、`"now-12h to now"`），而线上守卫 `validate_plan_steps` 只认受限词表（`full_demo_range` / `last_24_hours` / `latest` 等，见 `TIME_WINDOW_PATTERN`），导致计划被拒后回退 deterministic。DeepSeek 对照 100 条里 83 条、蒸馏 34 条栽在这上面。

**方案（A2，纯推理层，不用重训）**：在 `src/agent/planner.py` 加 `_normalize_time_window`，接在 `_step_from_llm_item` 解析时间窗那步（第 407 行附近）。两条线共用这个解析函数，同时受益。
- 相对时间折算成小时/分钟（`days×24`、`weeks×168`、`months×720`）——**关键**：gold 数据只用 hours，折算成小时才与训练格式一致，不引入模型没见过的 `days`。
- 覆盖：`past N days/hours/minutes/weeks/months`、词组（`last month`/`today`/`all data`）、紧凑写法（`7d`/`last_24h`）、区间（`now-12h to now`）、英文数字（`last two weeks`）。
- **无法可靠映射的值（字典形式、episode_id 错填等）原样返回**，让守卫照常拒绝——只救能确信的，绝不强行猜。是纯增量改动。
- 6 个单测（`tests/test_route_planner.py`）：回归、边界、负例、端到端。全绿。

**效果**（详见 `docs/distillation_report.md` 的 A2 前后对照表）：

| | 蒸馏 step_acc | 教师 step_acc | 蒸馏非fallback | 教师非fallback |
| --- | --- | --- | --- | --- |
| A2 前 | 47.0% | 21.0% | 51/100 | 13/100 |
| A2 后 | **68.0%** | **63.0%** | **74/100** | **72/100** |

> 注：教师 A2 前后是两次独立 API 调用（有随机性 + 超时条数不同），提升主体是 A2 但非 100% 纯归一化；蒸馏两版是同一 adapter + 确定性解码，差异可完全归因于 A2。

**A2 版预测文件**（本地，均 gitignore）：`eval_pred_distilledA2.jsonl`（服务器重跑后传回）、`eval_pred_teacher.jsonl`（本地并发重跑）。A2 前的对照快照留存：`eval_pred_distilled.jsonl`、`eval_pred_teacher_preA2.jsonl`、`eval_pred_teacher_t20.jsonl`（20s 超时废版，仅留档）。

---

## 10. 下一步（接手从这里选）

**A1 已完成**（`54bf21d`，补 100 条 gold + 重训 + 重评，step_acc 68%→84%）。蒸馏 A1 版残余 **10 条 fallback**，逐条核查后**全部是最难的 3 步复合任务**，集中在 A1 未专门覆盖的形态：
- **多指标同图/比值绘图**（`plot cooling_power and fan_power ratio`、`zone_temperature and setpoint together`）——单步 schema 只有一个 `metric_name`，表达不了两个 metric 的组合；
- **绝对时段窗**（`during the night 10 PM-6 AM`）——非"最近 N 小时"型，词表本就不支持；
- **episode/zone 具体引用 + 多文档检索**（`episode_001 zone_a`、`doc_002` / `doc_006`）——模型对具体 id 处理仍不稳。

这些**不是** time_window 归一化或基础 tool 选择问题（那两类 A1 已基本解决），是更深的多步复合表达能力。据此排下一步：

**收尾项（先做）**
- **提交未推送改动**：HANDOFF + 报告 + 新 train_card。直推 main 会被 auto 模式拦，push 可能需用户本地 `! git push`。
- 关键数字回写 README / 简历：**蒸馏 1.5B 达 84% step_acc，6 倍于规则基线，反超作为对照的 DeepSeek 云端 planner 21 个点，本地运行延迟仅 1/23**（可参照 `SFT_技术方案.md` 措辞；勿写"蒸馏了 DeepSeek"）。

**阶段3 DPO（推荐的下一个大步骤）**
- 见 §6。现在正好有蒸馏 A1 版的**真实错误**（那 10 条 fallback + exact_match 未命中的样本）可作 `rejected` 负例，比人工编造更真实。A1 已把数据缺陷补掉，DPO 不再是跟同一个缺陷较劲。
- 待写：`distill/build_dpo_data.py`（用项目 guard 信号构造偏好对）+ `distill/train_dpo.py`（TRL `DPOTrainer`）。

**可选：进一步扩数据 / 改 schema（治残余 10 条）**
- 多指标绘图需扩展 step schema（支持多 metric 或 metric 列表）——属**改 schema**，动到线上守卫，需谨慎评估。
- 绝对时段（10PM-6AM）需在词表/归一化里支持"时段"概念——同样是能力扩展，非小改。
- 这两类可作为独立的 A3 任务，或并入 DPO 的负例针对性优化。

**减少 JSON 截断（快速优化，若重现）**
- 若长计划被 `max_new_tokens` 截断，可在 `distilled_planner.py` 调大上限或优化停止条件。A1 版这类已很少（残余 10 条主因是上述表达力问题，非截断）。

