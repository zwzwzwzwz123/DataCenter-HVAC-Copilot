# Handoff — Planner 蒸馏 阶段 1（数据）+ 阶段 2（SFT）+ 阶段 4（评测）+ A2 优化

**日期**：2026-07-04（末次更新：2026-07-05，阶段4评测跑通 + A2 归一化优化完成后同步）
**范围**：`distillation_plan.md` 的阶段 1（精标数据）、阶段 2（SFT 训练脚本 + 实际训练）、阶段 4（接回评测，三方对比），外加 SFT 操作手册 + **A2（time_window 归一化，推理层优化）**。阶段 3（DPO）**未开始**，应用户要求暂停,待 A1/A2 优化后再启动。

**当前 git 状态**：`0fa7e18`（A2 归一化 + 阶段4报告）已 push 到远程。**工作区还有未提交改动**：
- `distill/eval_planners.py`（`--concurrency` 并发预测，已本地改好，commit 曾被 auto 模式拦下，**待重新提交**）
- `docs/distillation_report.md`（A2 后三方报告，**待提交**）
- ⚠️ 之前尝试的并发 commit 因 auto 模式拦截主分支推送未成功，接手需 `git add distill/eval_planners.py docs/distillation_report.md && git commit && git push`。

相关 commit：
- `36532e1` 重构 planner（抽出 `build_planner_messages` / `serialize_plan_steps`）
- `83aa39d` 阶段 1+2：600 条精标数据 + SFT 脚本 + 测试
- `4594405` / `0b07d5d` SFT 操作手册 + 技术方案 + gitignore 大权重
- `6b2783e` 适配服务器训练配置（TRL 1.7.1 API 迁移，见 §3.2）
- `69b97ec` 阶段4：评测脚本就绪 + deterministic 基线
- `70b927d` 阶段4加速：批量 GPU 推理 + 合并 adapter + `--fast`
- `0fa7e18` **A2：time_window 归一化 + 阶段4三方评测报告**

**训练结果（2026-07-05，AutoDL GPU）**：SFT val legality=100% / exact_match=96.67%（n=60），约 2 分钟。产物 `distill/checkpoints/sft-qwen1.5b/`（LoRA adapter，gitignore 排除大文件，仅 `sft_train_card.json` 进版本库）。

**阶段4评测结果（最终，A2 归一化后）**：三方对比闭环完成——

| planner | step_acc | order_acc | req_recall | policy_final | 非fallback | 延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| deterministic 基线 | 14.0% | 14.0% | 56.3% | 72.7% | 100/100 | ~0s |
| **蒸馏 1.5B (A2)** | **68.0%** | **68.0%** | 85.5% | 93.5% | 74/100 | 0.8s |
| DeepSeek 教师 (A2) | 63.0% | 63.0% | 85.7% | 93.5% | 72/100 | 18s |

**核心结论**：蒸馏 1.5B 达到 **68% step_acc**，是规则基线（14%）的近 5 倍，**追平云端 DeepSeek 教师**（63%），且本地运行、延迟仅教师 1/23、零 API 成本。

**A2 优化（推理层，不用重训）**：解析层加 `_normalize_time_window`，把模型爱输出的自然语言时间窗（`past 7 days` / `last month` / `7d`）映射回守卫词表。蒸馏 step_acc 47%→68%，教师 21%→63%。详见 §9。

**下一步**：见 §10。建议顺序：**提交未推送改动 → A1（补 gold 数据）→ 阶段3 DPO**。

---

## 1. 现在处于什么位置

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 1 数据构造 | ✅ 完成（`83aa39d`） | `distill/data/gold_labeled.jsonl`（600 条）+ train/val + data card |
| 2 SFT 脚本 | ✅ 完成（`83aa39d`，TRL 适配 `6b2783e`） | `distill/train_sft.py` |
| 2 SFT 训练 | ✅ 已跑通（legality 100% / exact 96.67%） | `distill/checkpoints/sft-qwen1.5b/`（大文件 gitignore） |
| — 操作手册 / 技术方案 | ✅ 完成（`4594405` / `0b07d5d`） | `SFT_训练操作手册.pdf` / `SFT_技术方案.md` |
| 4 接回评测 | ✅ **完成**：三方对比闭环，蒸馏 68% 追平教师 63% | `distilled_planner.py` / `eval_planners.py` / `docs/distillation_report.md` |
| — A2 time_window 归一化 | ✅ **完成**（推理层，不用重训） | `_normalize_time_window`（`src/agent/planner.py`）+ 6 单测 |
| — A2/并发 提交 | ⚠️ **待提交**：`eval_planners.py`（并发）+ 报告 | — |
| A1 补数据 | ⛔ 未开始（建议下一步，见 §10） | — |
| 3 DPO | ⛔ 未开始（A1 后再做） | — |

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
| `distill/distilled_planner.py` | 阶段 4：蒸馏模型的 RoutePlanner 封装 |
| `distill/eval_planners.py` | 阶段 4：predict/score 评测入口 |
| `distill/make_manual_pdf.py` | 生成 SFT 训练操作手册 PDF |
| `distill/SFT_训练操作手册.pdf` / `_v2.pdf` | 训练操作手册（面向操作，v2 为最新版） |
| `distill/SFT_技术方案.md` | SFT 技术方案手册（面向理解，讲原理与设计取舍） |
| `tests/test_distill_gold.py` | 保护 gold 数据 100% 合法 |
| `tests/test_distill_augment.py` | 问题扩充测试 |
| `tests/test_distill_sft_data.py` | SFT 数据构造测试 |
| `distillation_plan.md` | 五阶段总计划 |
| `pyproject.toml` | 新增 `[train]` extra |

---

## 8. 阶段 4（接回评测）——进行中，接手请从这里开始

目标：把蒸馏 SFT 模型接进现有评测，与 `deterministic`（规则基线）和 `env`（DeepSeek 云端教师）做**三方对比**，产出对比表 + `docs/distillation_report.md`。这是整条蒸馏链的闭环。

### 8.1 已完成（脚本就绪 + 本地已验证）
- **`distill/distilled_planner.py`**：`DistilledRoutePlanner`，实现 `RoutePlanner` 协议。加载基座 + LoRA adapter，复用线上同一套 `build_planner_messages`（构造 prompt）和 `_decision_from_llm_payload`（解析+校验），失败回退 deterministic。重依赖惰性加载，无 GPU 也能 import。
- **`distill/eval_planners.py`**：评测入口，`predict` / `score` 两段解耦。`predict` 跑一个 planner 产出 `predictions.jsonl`；`score` 读一或多个 predictions 出指标对比表（纯 Python，复用 `src/evaluation/metrics.py`）。
- **deterministic 基线已跑出**（100 条 `data/eval/compound_task_eval.jsonl`）：step_acc **14.0%** / order 14.0% / req_recall 56.3% / policy_final 72.7% / 100 条全非 fallback。产物 `distill/data/eval_pred_deterministic.jsonl`。
  - 解读：规则 planner 靠关键词匹配单路由，多步复合任务拆解弱，只有 14%。**这是给蒸馏模型立的低基线对照**——蒸馏 1.5B 若大幅超过 14%，即证明 SFT 学到了规则学不到的多步规划能力。

### 8.2 方案 B：三方对比完整步骤（接手直接照做）

**环境**：需要一台 GPU 机器（AutoDL，参照 `SFT_训练操作手册.pdf` 租卡/装依赖），且要有 **DeepSeek API key**（教师那一列用）。

**步骤 1 — 拉代码 + 装依赖 + 放好 LoRA adapter**
```bash
cd ~/autodl-tmp && git clone <repo> && cd DataCenter-HVAC-Copilot
pip install -e '.[train]'
# LoRA adapter：本地 distill/checkpoints/sft-qwen1.5b/ 未进 git（大文件）。
# 需从本地上传该目录到服务器同路径，或重新训练一次（train_sft，约2分钟）。
```
> ⚠️ `distill/checkpoints/` 已被 gitignore（只保留 `sft_train_card.json`）。git clone **不会**带下 LoRA 权重，必须单独上传本地已下载的 `adapter_model.safetensors` 等文件到 `distill/checkpoints/sft-qwen1.5b/`；或直接在服务器重跑 `python -m distill.train_sft` 重新生成。

**步骤 2 — 配置 DeepSeek 教师**（否则 `env` 那列会退回 deterministic，白跑）
```bash
cp .env.example .env
# 编辑 .env：填 DEEPSEEK_API_KEY=sk-...；把 LANGGRAPH_PLANNER_PROVIDER 设为 deepseek
```

**步骤 3 — 模型下载走镜像，别开学术加速代理**（见 §5 踩坑）
```bash
unset http_proxy https_proxy
export HF_ENDPOINT=https://hf-mirror.com
```

**步骤 4 — 三个 planner 各跑一次 predict**
```bash
# 蒸馏模型（需 GPU + adapter）
python -m distill.eval_planners predict --planner distilled \
    --adapter distill/checkpoints/sft-qwen1.5b \
    --out distill/data/eval_pred_distilled.jsonl

# DeepSeek 教师（需 .env 配好，会真实调 API，100 条有少量费用）
python -m distill.eval_planners predict --planner env \
    --out distill/data/eval_pred_teacher.jsonl

# deterministic 基线（已有 eval_pred_deterministic.jsonl；重跑也可）
python -m distill.eval_planners predict --planner deterministic \
    --out distill/data/eval_pred_deterministic.jsonl
```

**步骤 5 — 三方对比 + 出报告**
```bash
python -m distill.eval_planners score \
    distill/data/eval_pred_deterministic.jsonl \
    distill/data/eval_pred_distilled.jsonl \
    distill/data/eval_pred_teacher.jsonl \
    --report docs/distillation_report.md
```
终端会打印对比表，并写出 `docs/distillation_report.md`（含指标说明）。

### 8.3 完成后（4.4 收尾）
- 核对 `docs/distillation_report.md` 的对比表，确认核心结论：**蒸馏 1.5B 的 step_acc/order_acc 显著高于 deterministic，且接近 DeepSeek 教师，但本地运行、延迟/成本低**。
- 若蒸馏模型某指标明显弱于教师，如实写进报告「局限」，不要粉饰。
- 把关键数字回写 README / 简历（可参照 `SFT_技术方案.md` 的措辞）。

### 8.4 已知注意点
- `env` planner 由 `build_route_planner_from_env()` 构造，依赖 `.env` 的 `LANGGRAPH_PLANNER_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`；配错会静默退回 deterministic（表现为 `env` 那列 = deterministic）。跑完看 `predictions` 里的 `planner` 字段确认是否真的走了 `llm:deepseek:...`。
- `predict` 对每条不传 `task_type`，让 planner 从问题自行路由（compound 任务本来也没单一 task_type）。
- 三个 predictions 文件都基于同一 `--eval-path`（默认 `data/eval/compound_task_eval.jsonl`），score 时才可比。
- **加速开关**（`70b927d` 加）：蒸馏用 `--fast --batch-size 16 --max-new-tokens 160`（批量 GPU 解码 + 合并 adapter，快数倍）；教师/API 用 `--concurrency 8`（线程池并发，100 条从 ~40 分钟压到 ~3 分钟）。**注意本地这台机器 `model.safetensors` 曾没下全导致奇慢，蒸馏推理请在权重齐全的服务器上跑。**

---

## 9. A2：time_window 归一化（已完成，推理层优化）

**问题**：阶段4 第一版评测发现，模型（教师 + 蒸馏）大量输出自然语言时间窗（`"past 7 days"`、`"last month"`、`"7d"`、`"now-12h to now"`），而线上守卫 `validate_plan_steps` 只认受限词表（`full_demo_range` / `last_24_hours` / `latest` 等，见 `TIME_WINDOW_PATTERN`），导致计划被拒后回退 deterministic。教师 100 条里 83 条、蒸馏 34 条栽在这上面。

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

蒸馏 A2 版残余 26 条 fallback：**10 条 JSON 截断/格式错**、**8 条 tool 非法**、**8 条 time_window 无法归一化**（字典形式等真正无法映射的）。据此排下一步优先级：

**A1 — 补 gold 数据（治本，推荐先做）**
- 给 `distill/data/gold_labeled.jsonl` 补样本：question 用自然语言时间窗，steps 里 `time_window` 标成合法词表，让模型**自身**学会归一化（不再只靠 A2 推理层兜底）；同时补 tool 选择更清晰的样本，缓解那 8 条 tool 非法。
- 补完 `python -m distill.build_gold_sft` 重新生成 train/val，服务器重训（~2 分钟），重跑评测看提升。
- A1 是 A2 的"治本"版：A2 是安全网，A1 让能力进模型。

**阶段3 DPO（A1 之后做）**
- 见 §6。现在正好有蒸馏 A2 版的**真实错误**（那 26 条 fallback）可作 `rejected` 负例，比人工编造更真实。但建议先做 A1 把数据缺陷补掉，否则 DPO 在跟同一个缺陷较劲。
- 待写：`distill/build_dpo_data.py` + `distill/train_dpo.py`（TRL `DPOTrainer`）。

**收尾项（随时可做）**
- **提交未推送改动**：`eval_planners.py`（并发）+ `docs/distillation_report.md`（并发 commit 曾被 auto 模式拦下）。
- 关键数字回写 README / 简历：**蒸馏 1.5B 达 68% step_acc，近 5 倍于规则基线，追平 DeepSeek 教师，本地运行延迟仅 1/23**（可参照 `SFT_技术方案.md` 措辞）。

**减少 JSON 截断（快速优化）**
- 那 10 条多是 `max_new_tokens` 截断长计划。可在 `distilled_planner.py` 调大上限，或优化停止条件。

