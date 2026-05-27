# Resume Readiness Plan

本计划用于把当前项目推进到简历可展示状态。目标不是堆功能，而是让项目可运行、可展示、可解释、可验证。

## 当前结论

当前项目已经具备核心闭环：RAG、Tool Agent、时序工具、policy adapter、Safety Audit、FastAPI、Streamlit、108 条评测集和 baseline comparison。

当前不把人工评测作为阻塞项。默认展示口径是 deterministic metrics + quality proxy；LLM judge 可作为 LLM-as-Judge 增强，不能写成人工评审。Human Calibration 保持可选增强。

## 第一优先级

1. **截图和架构图**
   - 新建 `docs/images/`。
   - 截取 Copilot tab、三类 walkthrough、评测摘要 tab。
   - 在 README 首屏加入 Mermaid 架构图和截图。

2. **真实 embedding / FAISS 指标**
   - 使用 `pip install -e ".[dev,dense]"` 安装可选依赖。
   - 启用 sentence-transformers + FAISS 真实 dense retrieval。
   - 对比 keyword / dense / hybrid / hybrid+rerank。
   - 更新 `docs/experiment_report.md` 和 README 指标。

3. **LangGraph workflow**
   - 保留 deterministic router 作为 baseline。
   - 新增 LangGraph state graph：intent -> retrieval/tool -> evidence aggregation -> answer -> audit。
   - 对比 deterministic routing 与 LangGraph routing 的案例或指标。

## 已处理事项

- 人工评测从必须项调整为可选增强项。
- README 增加评测口径说明，明确 proxy、LLM-as-Judge 和 Human Calibration 的边界。
- 增加 Dockerfile 和 docker-compose，本地可用 `docker compose up --build` 启动 API + Streamlit。
- Streamlit 默认 API 地址支持 `HVAC_COPILOT_API_BASE_URL`，方便容器间调用。
- 使用你的 DROPT 源码仓库中的完整 BEAR 环境生成了 14 天逐小时、6 zone 的 `data/bear_processed/bear_rollout.csv`；demo 现在优先显示 `processed_csv`。

## 面试表述边界

当前可以写：

> 构建 DataCenter-HVAC Copilot：基于 BEAR HVAC 仿真轨迹，设计 RAG + Tool Agent + Evaluation 系统，支持文档问答、时序查询、异常诊断和策略建议；实现 evidence-grounded answer generation、Safety Audit、时序工具、policy adapter 边界、FastAPI/Streamlit demo 和 108 条评测集，并通过多 baseline comparison 验证检索、工具调用、证据覆盖和回答质量代理指标。

当前不要写：

- 基于 LangGraph 的 Agent 工作流。
- 已完成真实 FAISS/BGE 语义检索指标。
- 已完成人工评测。
- 使用真实数据中心生产遥测。
- LLM 直接控制 HVAC。
