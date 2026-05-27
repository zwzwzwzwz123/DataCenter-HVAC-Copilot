本文件回答：代码里有哪些 README 没充分放大、但对简历 reviewer 有价值的真实亮点。

# 被低估的亮点

### 亮点 1：FAISS 索引用 manifest + backup 做原子替换

**位置**：`src/knowledge/indexer.py:43`

**是什么**：知识库不是简单写一个 `index.faiss`。rebuild 时先写带 uuid 的 tmp 文件，再写 sidecar 和 manifest；替换前备份旧文件，异常时恢复旧 index/chunks/manifest。

```python
tmp_index = self.index_dir / f"index.faiss.{token}.tmp"
tmp_chunks = self.index_dir / f"chunks.jsonl.{token}.tmp"
tmp_manifest = self.index_dir / f"manifest.json.{token}.tmp"
...
_atomic_replace(tmp_index, self.index_path)
_atomic_replace(tmp_chunks, self.chunks_path)
_atomic_replace(tmp_manifest, self.manifest_path)
```

**为什么值钱**：这体现了“本地 demo 也考虑数据一致性”的工程意识。对简历 reviewer 来说，比单纯“用了 FAISS”更有辨识度。

**目前 README 怎么说的**：提到“临时文件原子替换”和 manifest，但放在长段落里，容易被忽略（`README.md:179`）。

**建议怎么放大**：简历可写“实现 SQLite source-of-truth + FAISS derived index，使用 sidecar/manifest/hash 校验和原子替换避免半写索引污染 citation。”

### 亮点 2：Memory 失败不阻断主回答，且状态分层暴露

**位置**：`src/api/app.py:76`

**是什么**：`/ask` 尝试加载 memory，如果 SQLite 或 retrieval 失败，会写入 `memory_status` 和 workflow trace，但仍继续走当前轮回答。保存 turn、indexing、trace persistence 也是分开上报。

```python
except sqlite3.Error as exc:
    memory_status.update({
        "storage": {"available": False, "error": str(exc)},
        "retrieval": {"available": False, "error": "memory storage unavailable"},
    })
...
memory_status["indexing"] = {"saved": False, "error": str(exc)}
```

**为什么值钱**：这比“加了 memory”更成熟。它展示了降级策略、可观测状态和主路径鲁棒性，面试里很好讲。

**目前 README 怎么说的**：有提到状态分开上报和 trace persistence，但没有强调“memory 失败不阻断 fresh evidence 回答”（`README.md:239`）。

**建议怎么放大**：在 README 架构图旁加一句：“memory 是辅助上下文层，失败时降级为 stateless `/ask`，并在 response 中暴露 storage/retrieval/indexing 状态。”

### 亮点 3：LangGraph 与 deterministic baseline 共享同一个 executor

**位置**：`src/agent/langgraph_workflow.py:35`

**是什么**：LangGraph 没有复制一套工具执行逻辑，而是复用 `baseline.task_executor`。`AgentTaskExecutor` 负责所有 `collect_*_evidence` 和 answer/audit。

```python
self.task_executor = task_executor or baseline.task_executor
...
return self.task_executor.collect_timeseries_query_evidence(question, step.reason, step)
```

**为什么值钱**：这能解释为什么 `langgraph_tool_agent` 与 `rag_tool_agent` 指标一致：workflow 变化不污染底层工具行为。对 reviewer 来说，这是“可对照实验”意识，而不是随便接 LangGraph。

**目前 README 怎么说的**：写了共享组件，但没有作为设计亮点展开（`README.md:101`）。

**建议怎么放大**：面试里可说：“我把 orchestration 和 tool execution 分层，LangGraph 只改变编排与 trace，baseline 可作为回归对照。”

### 亮点 4：LLM planner 输出被 schema/规则夹住，失败可回退

**位置**：`src/agent/planner.py:406`

**是什么**：LLM planner 不是自由文本直接执行。它被限制为最多 3 步、固定 route、固定 tool、支持的 time_window，且 policy 必须最后；任何异常回退 deterministic planner。

```python
if len(steps) > MAX_PLAN_STEPS:
    raise ValueError("plan must contain at most 3 steps")
...
if step.tool not in ALLOWED_STEP_TOOLS[step.route]:
    raise ValueError(...)
...
if "policy_recommendation" in routes and routes[-1] != "policy_recommendation":
    raise ValueError("policy_recommendation must be the final step")
```

**为什么值钱**：这是大模型应用岗会关心的“受控 LLM 输出”和“失败语义”。它比单纯说“用了 LangGraph + DeepSeek planner”更专业。

**目前 README 怎么说的**：有描述 planner 只返回受控 steps 和非法 fallback（`README.md:8`），但没有把约束细节列成亮点。

**建议怎么放大**：简历可写“设计 constrained LLM route planner：固定 route/tool schema、max 3 steps、policy final-step constraint、invalid-output deterministic fallback。”

### 亮点 5：评测报告明确区分 proxy、人审和 LLM judge

**位置**：`src/evaluation/report.py:337`

**是什么**：报告不会把 deterministic proxy 或 LLM judge 伪装成人工评审。人审模板为空时显示 `pending_human_review`；LLM judge 默认关闭。

```python
if labeled_count == 0:
    return {
        "labeled_count": 0,
        "status": "pending_human_review",
    }
```

**为什么值钱**：这体现了评测诚实度。项目虽未完美，但知道哪些指标只是 proxy，这对简历作品很重要。

**目前 README 怎么说的**：写了“不要把 deterministic proxy 或 LLM judge 说成人工评测”（`README.md:551`），但亮点表达偏防御。

**建议怎么放大**：在简历/README 中把它转为正向表述：“建立三层评测口径：deterministic metrics、quality proxy、optional LLM/human calibration，并在报告中显式标注 pending 状态。”

### 亮点 6：DROPT adapter 有 deterministic sampling 和明确 fallback

**位置**：`src/policies/dropt_adapter.py:419`

**是什么**：adapter 不只是文件存在。它加载 checkpoint、抽取 20 维 BEAR state，基于 state hash 设 seed 做 deterministic sample；缺 checkpoint 或 state 不完整时回退 rule-based。

```python
seed = _stable_seed(input_state_id, state_vector)
with torch.no_grad():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        action_tensor = self._bundle._actor.sample(state_tensor)
```

**为什么值钱**：对“仿真控制 + LLM 解释边界”这个项目主题很贴合。它能说明 policy backend 可替换且输出可复现。

**目前 README 怎么说的**：写了默认 DROPT / Guided-DiffFNO 策略后端（`README.md:257`）。

**建议怎么放大**：可以放大为“离线策略工具接入”而不是“控制器”。⚠️ 需要确认作者能否讲清 checkpoint 来源、20 维 state layout、为什么用 deterministic sampling。
