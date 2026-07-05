# Handoff — Planner 蒸馏 阶段 1（数据）+ 阶段 2（SFT 脚本）

**日期**：2026-07-04（末次更新：2026-07-05，SFT 在 GPU 上训练成功后同步）
**范围**：`distillation_plan.md` 的阶段 1（精标数据扩充）与阶段 2（SFT 训练脚本 + 实际训练），外加 SFT 训练操作手册的编写。阶段 3（DPO）**未开始**，应用户要求暂停。

**当前 git 状态**：工作区干净，全部产物已提交。相关 commit：
- `36532e1` 重构 planner（抽出 `build_planner_messages` / `serialize_plan_steps`）
- `83aa39d` 阶段 1+2：600 条精标数据 + SFT 脚本 + 测试
- `4594405` 生成 SFT 操作手册（PDF + 生成脚本）
- `6b2783e` 适配服务器训练配置（TRL 1.7.1 API 迁移，见 §3.2）

**训练结果（2026-07-05，AutoDL GPU）**：SFT 已成功跑通，**val legality = 100%，exact_match = 96.67%（n=60）**，eval_loss 0.144→0.108 平稳收敛无过拟合，用时约 2 分钟。产物 `distill/checkpoints/sft-qwen1.5b/`（LoRA adapter + train_card）在服务器上，需下载留存。

---

## 1. 现在处于什么位置

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 1 数据构造 | ✅ 完成（已提交 `83aa39d`） | `distill/data/gold_labeled.jsonl`（600 条精标）+ train/val + data card |
| 2 SFT 脚本 | ✅ 完成（`83aa39d`，TRL 适配 `6b2783e`） | `distill/train_sft.py` |
| 2 SFT 训练 | ✅ **已跑通**（AutoDL，legality 100% / exact 96.67%） | 服务器 `distill/checkpoints/sft-qwen1.5b/`（需下载） |
| — 训练操作手册 | ✅ 完成（已提交 `4594405`） | `distill/SFT_训练操作手册.pdf` / `_v2.pdf` + `make_manual_pdf.py` |
| — SFT 技术方案手册 | ✅ 完成 | `distill/SFT_技术方案.md` |
| 3 DPO | ⛔ 未开始（暂停） | — |
| 4 接回评测 | ⛔ 未开始 | — |

---

## 2. 本次会话做了什么

### 2.1 精标数据：300 → 600 条
- `distill/data/gold_labeled.jsonl` 从 300 扩到 **600 条**手标 `{question, steps}`。
- **全部 600/600 通过 `validate_plan_steps` 校验**（线上同一个 guard）。
- 步数分布（应用户"多步为主"的要求调整后）：
  - 1步 258 (43%) / 2步 173 (28%) / 3步 135 (22%) / 4步 34 (5%)
  - **多步（≥2）占比从原来的 29% 提升到 57%**。
- 覆盖：11 个工具全覆盖；14 个 metric_name **全部是 `bear_rollout.csv` 的真实列**（零脏数据）；中英文混合、口语化、抗干扰、多轮指代。
- 运行 `python -m distill.build_gold_sft` 生成 `gold_sft_train.jsonl`（540）/ `gold_sft_val.jsonl`（60）/ `gold_sft_data_card.json`。

### 2.2 SFT 训练脚本 `distill/train_sft.py`
- 基座默认 **Qwen2.5-1.5B-Instruct**，`--model` 可配置（换 3B/8B 无需改代码）。
- **QLoRA 4-bit**（单卡 12–16G 可跑），`--no-quantize` 切全精度 LoRA。
- **数据用 TRL prompt/completion 格式**：SFTTrainer 自动套模型 chat template，训练格式 = 推理格式。
- **completion-only loss**：`SFTConfig(completion_only_loss=True)` 只对 assistant 输出算 loss（TRL ≥1.x 内置，取代旧的 `DataCollatorForCompletionOnlyLM`，见 §3.2）。
- **验收指标复用线上代码**：训练后用 `_decision_from_llm_payload`（线上 planner 解析 LLM 输出的同一函数）测 val 集**合法率**和 **route 精确匹配率**，写进 `sft_train_card.json`。
- `pyproject.toml` 新增 `[train]` extra 记录训练依赖。

### 2.3 测试
- 新增 `tests/test_distill_gold.py`，含硬断言"线上 600 条永远 100% 合法"，防数据劣化。
- 蒸馏 + route planner 全部相关测试通过（28 个）。

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
- ✅ 600 条 gold 全部过 `validate_plan_steps`
- ✅ ID 唯一、问题唯一（修掉了 1 条重复 gold_0479）、train/val 无 ID 泄漏、train+val 覆盖全部
- ✅ 全部 600 条 completion 可被线上 `_decision_from_llm_payload` 解析（含 markdown 围栏兼容）
- ✅ 14 个 metric_name 全是真实数据列
- ✅ `train_sft.py` 可 import、lint 通过、依赖缺失时给可操作提示、`load_records` 正常
- ✅ 蒸馏 + route planner 测试全绿

**训练（AutoDL GPU，2026-07-05）**：
- ✅ SFT 训练收敛：train_loss 0.60→0.058，eval_loss 0.144→0.108（无过拟合反弹），约 2 分钟
- ✅ **val legality 100%**（60/60 生成计划全部通过线上 guard）——远高于未微调基座
- ✅ **exact_match 96.67%**（58/60 route 序列与 gold 完全一致）
- ✅ TRL 1.7.1 + transformers 5.13.0 下 API 兼容（`6b2783e` 迁移后）

**仍待办**：
- ⏳ 把服务器上的 `distill/checkpoints/sft-qwen1.5b/` 下载留存（LoRA adapter + train_card）

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

结束会打印 `[eval] val legality=100.00% exact_match=96.67% (n=60)`，并写 `sft_train_card.json`。

**踩过的两个坑（已记录）**：
1. TRL API 版本不匹配 → 见 §3.2。
2. 模型下载速度为 0 → 学术加速代理（`http_proxy`）与国内镜像 `hf-mirror` 冲突；用镜像时先 `unset http_proxy https_proxy`。

**下一步**：产物下载留存后关机停计费。是否继续阶段 3/4 见 §6。

---

## 6. 阶段 3（DPO）预备（暂停中，未来做）

计划见 `distillation_plan.md` 阶段 3。**建议先把 SFT 在云端跑通再启动 DPO**，因为：
- DPO 以 SFT 模型为起点（policy）+ SFT 冻结副本为 reference。
- DPO 偏好数据的 `rejected` 最好用 **SFT 初版模型的真实错误**构造，比人工编造的负例更真实。
- 待写脚本：`distill/build_dpo_data.py`（用项目 guard 信号构造偏好对）+ `distill/train_dpo.py`（TRL `DPOTrainer`）。

---

## 7. 关键文件索引

| 文件 | 作用 |
| --- | --- |
| `distill/data/gold_labeled.jsonl` | 600 条精标源数据（可继续扩充） |
| `distill/augment_questions.py` | 问题扩充（生成 `data/questions.jsonl`） |
| `distill/build_sft_data.py` | 教师产出 → SFT 数据（`data/sft_*.jsonl`） |
| `distill/build_gold_sft.py` | gold → train/val + data card（依赖 §3.1 函数） |
| `distill/train_sft.py` | 阶段 2 SFT 训练 |
| `distill/make_manual_pdf.py` | 生成 SFT 训练操作手册 PDF |
| `distill/SFT_训练操作手册.pdf` / `_v2.pdf` | 训练操作手册（面向操作，v2 为最新版） |
| `distill/SFT_技术方案.md` | SFT 技术方案手册（面向理解，讲原理与设计取舍） |
| `tests/test_distill_gold.py` | 保护 gold 数据 100% 合法 |
| `tests/test_distill_augment.py` | 问题扩充测试 |
| `tests/test_distill_sft_data.py` | SFT 数据构造测试 |
| `distillation_plan.md` | 五阶段总计划 |
| `pyproject.toml` | 新增 `[train]` extra |
