# Handoff — Planner 蒸馏 阶段 1（数据）+ 阶段 2（SFT 脚本）

**日期**：2026-07-04（末次更新：手册提交后同步）
**范围**：`distillation_plan.md` 的阶段 1（精标数据扩充）与阶段 2（SFT 训练脚本），外加 SFT 训练操作手册的编写。阶段 3（DPO）**未开始**，应用户要求暂停。

**当前 git 状态**：工作区干净，全部产物已提交。相关 commit：
- `36532e1` 重构 planner（抽出 `build_planner_messages` / `serialize_plan_steps`）
- `83aa39d` 阶段 1+2：600 条精标数据 + SFT 脚本 + 测试
- `4594405` 生成 SFT 操作手册（PDF + 生成脚本）

---

## 1. 现在处于什么位置

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 1 数据构造 | ✅ 完成（已提交 `83aa39d`） | `distill/data/gold_labeled.jsonl`（600 条精标）+ train/val + data card |
| 2 SFT 脚本 | ✅ 脚本完成，**未在 GPU 上实跑** | `distill/train_sft.py` |
| 2 SFT 训练 | ⏳ 待办 | 需租 GPU 执行 |
| — 训练操作手册 | ✅ 完成（已提交 `4594405`） | `distill/SFT_训练操作手册.pdf` / `_v2.pdf` + `make_manual_pdf.py` |
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
- **chat template 对齐**：用 `tokenizer.apply_chat_template`，训练格式 = 推理格式。
- **completion-only loss**：TRL `DataCollatorForCompletionOnlyLM`，只对 assistant 输出算 loss。
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

### 3.2 训练脚本尚未在真实 GPU 上跑过
`train_sft.py` 在**无 GPU 的本地**做了充分静态验证（见 §4），但 TRL API 版本差异可能需微调。已知需注意：
- TRL 版本演进较快，`SFTConfig` 的 `max_seq_length` / `dataset_text_field` 参数名在不同版本可能不同；若报参数错误，对照所装 TRL 版本文档调整。
- `DataCollatorForCompletionOnlyLM` 的 `response_template` 目前设为 Qwen 的 `<|im_start|>assistant\n`；换非 Qwen 基座时**必须**改这个 marker。

### 3.3 预先存在的测试失败（与本次工作无关）
`tests/test_knowledge_indexer.py`(7) 和 `tests/test_memory_retriever.py`(1) 失败，原因是**未安装 `sentence-transformers`（`[dense]` extra）**，报错信息里已明示。**不是本次改动引入的**，跑 `pip install -e '.[dense]'` 或忽略即可。

---

## 4. 已验证 / 未验证清单

**已验证（本地无 GPU）**：
- ✅ 600 条 gold 全部过 `validate_plan_steps`
- ✅ ID 唯一、问题唯一（修掉了 1 条重复 gold_0479）、train/val 无 ID 泄漏、train+val 覆盖全部
- ✅ 全部 600 条 completion 可被线上 `_decision_from_llm_payload` 解析（含 markdown 围栏兼容）
- ✅ 14 个 metric_name 全是真实数据列
- ✅ `train_sft.py` 可 import、lint 通过、依赖缺失时给可操作提示、`load_records` 正常
- ✅ 蒸馏 + route planner 测试全绿

**未验证（需 GPU）**：
- ⏳ SFT 实际训练收敛、val loss 曲线
- ⏳ 训练后 val 合法率是否显著高于未微调基座（阶段 2 验收标准）
- ⏳ TRL/peft 具体版本下 API 兼容性

---

## 5. 下一步怎么做（阶段 2 收尾）

在租用的 GPU 机器上（AutoDL/Colab，16–24G）：

```bash
# 1. 确认 §3.1 的 planner 函数在工作区存在
grep -n "def build_planner_messages\|def serialize_plan_steps" src/agent/planner.py

# 2. 装训练依赖
pip install -e '.[train]'

# 3. 跑 SFT
python -m distill.train_sft \
    --train distill/data/gold_sft_train.jsonl \
    --val   distill/data/gold_sft_val.jsonl \
    --output distill/checkpoints/sft-qwen1.5b

# 4. 看 distill/checkpoints/sft-qwen1.5b/sft_train_card.json 里的
#    legal_rate / exact_match_rate，对照阶段2验收标准
```

**验收判据**：val `legal_rate` 应显著高于未微调基座（基座通常输出格式混乱、合法率低）。若 1.5B 的 legal_rate 已接近教师且 route 匹配率高，阶段 2 达标；若某子任务明显偏弱，再考虑 `--model` 换 3B 重跑（这应由 eval 数字驱动，不要一开始就上大模型——理由见会话中关于模型选型的讨论）。

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
| `distill/SFT_训练操作手册.pdf` / `_v2.pdf` | 训练操作手册（v2 为最新版） |
| `tests/test_distill_gold.py` | 保护 gold 数据 100% 合法 |
| `tests/test_distill_augment.py` | 问题扩充测试 |
| `tests/test_distill_sft_data.py` | SFT 数据构造测试 |
| `distillation_plan.md` | 五阶段总计划 |
| `pyproject.toml` | 新增 `[train]` extra |
