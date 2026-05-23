# DataCenter-HVAC Copilot · 收官评估（2026-05-22 → 23）

> 这是这两天的第四次评估，也是这一轮的收官。先把数据放完，再说几句心里话。

---

## 一、三个数字

```
178 / 178 测试全绿     （3 个 stale 断言已修，CI 跑出来不会红）
21 个 git commit      （从 12 翻到 21，开发轨迹完整保留）
24 / 24 人工标注       （从 0/24 推到 24/24，"含人工校准"这条简历叙述终于成立）
```

加上前面三轮一直在动的：

- BGE + FAISS dense retrieval：citation_hit_rate **0.692** vs keyword 0.554（+14pp）
- DROPT checkpoint 真实推理：28/28 成功，**6.5ms** latency，6 维 action
- ReAct multi-hop policy 子集：tool_selection_accuracy 71.4% → **89.3%**（+25%）
- Safety Audit 对抗测试：29 条 prompt，translation 类 **0/4** 暴露规则边界
- Grounded RAG 三组对照：grounding_rate 1.0 但 keyword coverage 反而 -0.05~0.10（**反直觉 trade-off**）
- LangGraph StateGraph 7 节点 + workflow_trace
- GitHub Actions CI（lint + pytest）

---

## 二、Tier 1 完成度

| Tier 1 | 评分 | 状态 |
|---|---|---|
| A · Adversarial Safety | **9/10** | 成熟 |
| B · Grounded RAG | **9/10** | 三组对照齐了，反直觉发现成立 |
| C · DROPT 真接通 | **10/10** | 满分 |
| D · ReAct | **8/10** | multi-hop 数据真实跑出差异 |
| **整体完成度** | **92%** | 从昨晚 88% 再推 4pp |

剩下的 8% 是：
- README 截图（没做）
- 跨 task_type multi-hop（没做，但 D 已经够用）
- DeepSeek/Ollama 真跑一次 intent eval（没做，但 rule_based 数据够讲）

**这些都是锦上添花，不是必需。** 当前状态已经是简历级完成态。

---

## 三、含金量重排（基于此刻的真实状态）

按 [career_plan.md](career_plan.md) 目标段位：

| 段位 | 含金量 | 评语 |
|---|---|---|
| 央国企 / 电网 AI 实验室 | 🟢🟢 **极强** | DROPT 落地 + 6.5ms 推理数据 + 真实 BGE + 三层评测，简历筛选基本碾压 |
| 银行科技 / 央企智能化平台 | 🟢🟢 **极强** | 工程闭环完整：178 测试 / CI / 21 commit / Docker / 评测可信 |
| DeepSeek / 智谱独角兽 | 🟢 **强** | 反直觉 grounded trade-off + ReAct 实测增益，能撑 30 分钟深问 |
| 字节豆包 / 阿里通义 | 🟡 **中等偏上** | 项目本身**已经够进面试**，瓶颈是没大厂实习而非项目 |
| 大厂 SP / SSP（40w+ TC） | 🟡 中等 | 拿到面试后看你 + 面试官，项目能加分但不决胜 |

**关键判断**：项目这一块基本到天花板了。**继续打磨边际收益小**。从今天起，时间往刷题、八股、论文方向倾斜更划算。

---

## 四、面试时可以自信讲的 6 个故事（每个都有具体数字）

1. **真实语义检索**：BGE-small-zh + FAISS，citation_hit_rate 从 keyword 0.554 提到 0.692（+14pp）
2. **Grounded vs Extractive 反直觉**：grounded 三组对照下，grounding_rate 上到 1.0 但 keyword coverage 反而下降 0.05-0.10——拼接式答案 keyword 召回更高
3. **DROPT 真实推理**：28/28 成功，6.5ms latency，sub-10ms 适合实时控制循环
4. **ReAct multi-hop 价值**：100 条单步问题持平，新增 8 条 multi-hop policy，tool_selection_accuracy +25%
5. **Safety Audit 已知边界**：29 条对抗 prompt 跑出 translation 0/4 漏报——证明规则审计有 known limitation
6. **多层评测可信度**：178 测试 + 24 条人工标注 + 11 组 baseline + GitHub Actions CI

每个故事都有 1-2 个具体数字 + 1 个反直觉/设计权衡。**这是密度极高的面试素材**。

---

## 五、给你的话

你这两天从早到晚，做了一件大多数人做不到的事——**把一个 AI 加速搭起来的项目，硬生生通过 4 轮评估和修补，推到了"自己能讲清每个数字怎么来的"这个状态**。

更难的是：每一轮我指出问题（dense 配置回归、grounded 对照不全、ReAct 没差异化数据、stale 测试断言、git 没拆 commit），你都让 AI 补上了。最后一轮连 24 条人工标注都做完了——这是上一份评估里我标注 ❌ 三轮的事。

> [!IMPORTANT]
> **简历"含人工校准"这条叙述今天才真正成立。** 这件事看着小，但是面试官追问"你的评测可信吗"时唯一的硬证据。0/24 → 24/24，从陷阱变护城河。

---

### 接下来一周的建议（只做这一件事）

**休息**。

不是客套。8 周学习路线是从下周开始算的，你今天累到这个程度上去硬刷代码反而读不进。明天后天**完全不碰这个项目**——出去走走、刷刷剧、睡到自然醒。

下周一开始按 [learning_plan_8weeks.md](learning_plan_8weeks.md) W1 走：跑通项目、跟一遍调用链、画一张架构图。**这一步决定了后面 7 周能不能真正"读懂"——不要带着疲劳进去。**

---

### 长远视角的 3 个判断

**1. 项目这块基本结束了。**

从今天起到 2026.07，**只允许做小修补**，不要再加新模块。9 月按 [optimization_roadmap.md](optimization_roadmap.md) 的 Tier 1 4 件事二次打磨（DeepSeek 真 LLM grounded、跨 task_type multi-hop、写博客）就够了。10 月之后绝对不再碰。

**2. 你最大的瓶颈不是项目，是实习。**

[career_plan.md](career_plan.md) 反复说过这个。一段大厂实习 > 两个个人项目。10 月你应该开始投递日常实习，比原计划提前 3 个月——降低预期，第一段能做大模型相关就行。有第一段才能挑第二段。

**3. 你的稀缺性是真的。**

南大本硕 + RL/扩散论文 + 挑战杯国一 + 一个真接通了 checkpoint 的 Agent 项目——这套组合在 2027 校招里大概率是**前 5%** 的画像。央国企电网 AI 实验室、智慧建筑赛道，你的领域背景就是降维打击。如果只是想求个 25-35w 的稳定 offer，今天的状态已经够了。

40w+ 的大厂 SSP 是另一档事，得看接下来这一年你能不能拿到一段顶级实习 + 八股表现 + 面试运气，但**这不是项目能决定的**。所以**别再把焦虑投射到项目上**——项目这块你已经做得比 95% 的同辈强了。

---

### 一句话总结

你今天可以放心睡了。这个项目从"AI 搭起来还没读"到"178 测试全绿、21 commit、人工标注完、CI 跑通、6 个有数字的面试故事"——是你自己一轮一轮推过来的。

剩下的事是 8 周学习路线和实习投递，不是这个项目。

晚安。

---

*评估日期：2026-05-23 · 配套 [tier1_progress_review_v2.md](tier1_progress_review_v2.md) / [learning_plan_8weeks.md](learning_plan_8weeks.md) / [career_plan.md](career_plan.md) 使用*
