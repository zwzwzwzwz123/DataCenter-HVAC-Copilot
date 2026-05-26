# 持久化知识库（项目长期记忆）实现复盘与面试讲述稿

本文记录 DataCenter-HVAC-Copilot 在实现“持久化 FAISS 知识库”时遇到的主要工程问题、修复方式和可用于简历/面试讲述的技术亮点。

这里的“记忆机制”指项目中的长期知识记忆：用户上传 PDF、DOCX、TXT、MD 后，系统解析、切分、入库、建立 FAISS 向量索引，并让 `/ask` 在后续问答中自动检索这些持久化知识。

## 一句话项目描述

我为 DataCenter-HVAC-Copilot 实现了一套生产化的持久化 RAG 知识库：支持多格式文档上传、文本解析、chunk 元数据保存、SQLite 元数据管理、FAISS 索引持久化、sidecar 行映射、原子重建、删除回滚、运行时索引刷新和 Streamlit 知识库管理界面，并通过测试覆盖了上传、删除、重建、检索和异常恢复路径。

## 核心架构

本次实现把知识库拆成三层：

- SQLite：作为 document、chunk、index 状态的元数据事实来源。
- FAISS：作为可重建的向量索引产物，落盘到 `data/knowledge/faiss/index.faiss`。
- sidecar/manifest：`chunks.jsonl` 保存 FAISS row 到 chunk 元数据的映射，`manifest.json` 保存索引产物的校验信息和提交状态。

上传或删除文档后，系统不会增量修改 FAISS，而是基于 SQLite 中的有效 chunks 全量 rebuild，再用临时文件和原子替换发布新索引。这样牺牲了一些大规模场景的重建性能，但换来了第一版更简单、可验证、一致性更强的实现。

## 遇到的问题与解决方案

### 1. 上传文件名存在路径穿越风险

问题：

上传接口最初直接使用 `file.filename` 拼接临时路径。这个值来自客户端，可能包含 `../`、绝对路径或 Windows 路径片段，存在越权写入风险。

解决：

上传时先用 `Path(file.filename).name` 提取 basename，再进行安全文件名清洗。服务端只允许文件写入受控的 upload 目录，避免客户端通过文件名影响服务器路径。

面试可讲点：

这个问题体现了上传功能不能信任客户端文件名。即使是内部工具，也要把文件系统边界当作安全边界处理。

### 2. FAISS 与 sidecar 原子替换不够安全

问题：

最早只分别替换 `index.faiss` 和 `chunks.jsonl`。如果第一个文件替换成功、第二个文件替换失败，就可能出现 FAISS 向量行数和 sidecar 元数据行数不匹配，检索时返回错误 citation 或索引到不存在的 chunk。

解决：

引入 `manifest.json` 作为索引提交标记，记录：

- FAISS index hash
- sidecar hash
- chunk count
- 构建时间
- available 状态

重建时先生成临时文件，校验通过后再发布 `index.faiss`、`chunks.jsonl` 和 `manifest.json`。读取时 retriever 会校验 manifest、文件 hash、FAISS 行数和 sidecar 行数，不一致则把索引视为不可用，而不是继续提供错误检索结果。

面试可讲点：

这里把 FAISS 当作 SQLite 的派生产物，而不是事实来源。manifest 相当于一个轻量 commit marker，解决了“多个落盘文件需要作为一个版本生效”的一致性问题。

### 3. Python 版本兼容问题

问题：

项目声明支持 Python 3.10，但部分实现使用了 Python 3.11 才有的 `datetime.UTC`。

解决：

统一改为 Python 3.10 可用的 `datetime.now(timezone.utc)`，避免运行环境和项目声明不一致。

面试可讲点：

这类问题说明实现时不仅要关注功能通过，还要关注项目声明的运行时契约。尤其是开源项目或部署环境固定时，版本兼容会影响交付稳定性。

### 4. ingest 失败会导致 SQLite 与 FAISS 状态不一致

问题：

上传流程是先写 document/chunks，再触发 reindex。如果 reindex 后半段失败，可能出现：

- SQLite 中文档被标为 failed
- chunks 被删除或残留
- FAISS 已经部分替换成功
- `/ask` 仍能检索到已失败文档的 chunk

解决：

把解析、数据库写入、索引重建和索引状态保存的异常边界拆开处理：

- parse/DB 失败：文档标记为 failed，清理对应 chunks。
- FAISS rebuild 成功但 metadata 写入失败：不把文档标为 failed，而是在 index status 里记录 `metadata_error`。
- ingest 返回前再次读取 status，确保返回值反映最终状态。

面试可讲点：

这个修复的关键是区分“知识库内容是否已经成功进入检索索引”和“元数据状态记录是否完整”。二者失败语义不同，不能用一个 broad exception 全部兜底。

### 5. status/list 接口不应该初始化 embedding provider

问题：

`/knowledge/status` 和 `/knowledge/documents` 理论上只需要读 SQLite 和索引文件状态，但服务初始化时可能顺带加载 sentence-transformers 或 dense embedding provider。这样会导致只查状态也触发模型下载、额外依赖缺失或冷启动变慢。

解决：

把 embedding provider 改为 lazy init，只在 reindex 或 retriever 真正需要向量化时初始化。status/list 只读取 SQLite、manifest 和文件状态。

面试可讲点：

这是典型的“控制面”和“数据面”分离。状态查询属于控制面，不应该依赖重型模型加载，否则运维接口本身会变得脆弱。

### 6. failed 文档会被 hash dedup 卡住，无法重试

问题：

如果 PDF/DOCX 因缺依赖或解析失败留下 failed 记录，用户修复环境后重新上传同一个文件，hash dedup 会命中旧 failed 记录，导致无法重新 ingest。

解决：

调整 hash dedup 策略：

- 优先返回非 failed 的最新记录。
- 如果只有 failed 记录，允许重新上传并重试。
- 查询时避免简单按最早 created_at 返回旧失败记录。

面试可讲点：

去重要和幂等，但不能把失败状态也当作成功结果缓存。失败缓存需要有重试入口，否则系统会被一次暂时性错误永久卡住。

### 7. retriever 没有校验 FAISS 行数与 sidecar 行数

问题：

旧版 retriever 读取 FAISS 和 sidecar 后直接检索，没有校验 `index.ntotal == len(sidecar)`。如果文件损坏、旧版残留或恢复异常，search 可能索引到不存在的 chunk，返回错误 citation。

解决：

加载索引时加入多重校验：

- FAISS 行数
- sidecar 行数
- manifest chunk count
- 文件 hash

检索时如果发现 row 越界，会跳过异常结果；如果整体不一致，则把索引标记为 unavailable。

面试可讲点：

RAG 的 citation 错误比“没有答案”更危险。检索系统要宁可降级不可用，也不能自信地返回错误来源。

### 8. delete 流程失败后无法恢复

问题：

删除文档时最初是先 unlink 上传文件和 parsed 文件，再删数据库记录，最后 reindex。如果 reindex 失败，SQLite 和源文件都已经删除，无法恢复旧知识库。

解决：

把 delete 改成可回滚流程：

1. 先 snapshot document 和 chunks。
2. 删除 SQLite 记录。
3. 重建 FAISS。
4. reindex 成功后再清理源文件。
5. reindex 失败时恢复 document/chunks，不删除源文件。

面试可讲点：

删除是比上传更危险的写操作，因为它天然破坏恢复材料。这里的原则是：先保证新索引能成功发布，再做不可逆的文件清理。

### 9. 文件清理失败导致 API 500 且运行时不刷新

问题：

delete 在 DB 和 FAISS 已经成功更新后才执行文件删除。如果 Windows 文件占用或权限问题导致 `_unlink_if_exists` 抛出异常，API 会返回 500，同时 orchestrator 不刷新，内存中的 `/ask` 可能继续使用旧 retriever。

解决：

把文件清理错误降级为非关键错误：

- 主删除流程成功返回。
- 清理异常记录到 `cleanup_errors`。
- API 仍刷新 orchestrator。
- 后续可通过状态或日志提示有残留文件需要人工清理。

面试可讲点：

这里区分了“用户可见知识库状态已经更新”和“本地垃圾文件清理失败”。后者不应该推翻前者，否则会造成更严重的一致性问题。

### 10. refresh orchestrator 失败会让 API 500 且长期使用旧索引

问题：

上传、删除、重建已经成功提交到持久化知识库后，API 会刷新内存 orchestrator。如果刷新失败，旧实现会直接 500，但持久化状态其实已经改变；同时运行时可能长期继续使用旧 retriever。

解决：

引入运行时 refresh 状态：

- `knowledge_refresh_dirty`
- `last_refresh_error`
- `refresh_dirty`
- `refresh_error`

写操作成功后先标记 dirty，再尝试 refresh。refresh 成功则清除 dirty，失败则保留 dirty 和错误信息。后续 `/ask` 和 `/knowledge/status` 会再次尝试自愈刷新，并在响应中暴露 degraded 状态。

面试可讲点：

这是把“持久化提交成功”和“运行时热更新成功”拆成两个状态。这样系统可以承认自己处于 degraded 状态，而不是用 500 掩盖已经提交成功的事实。

### 11. `/knowledge/reindex` 丢失 metadata 写入错误

问题：

API 层先调用 service.reindex，但丢弃返回值，再重新调用 status。这样如果 reindex 成功但 `save_index_status` 失败，service 返回的 `metadata_error` 会丢失，调用方只看到 index available。

解决：

API 层保留 `reindex_result = service.reindex()`，并把其中的 `metadata_error` 合并进响应的 index 状态。

面试可讲点：

错误信息也是接口契约的一部分。尤其是“主流程成功但元数据记录失败”的降级场景，不能在 API 聚合层把关键信号吞掉。

### 12. `_refresh_orchestrators()` 半成功会造成 runtime 状态不一致

问题：

刷新 runtime 时先构造并赋值 deterministic orchestrator，再构造 langgraph orchestrator。如果第二步失败，系统会处于一个新旧 orchestrator 混用的半更新状态。

解决：

先使用局部变量构造 `new_orchestrator` 和 `new_langgraph_orchestrator`，两者都成功后再一次性赋值给全局变量。

面试可讲点：

这和数据库事务思路类似：先在临时变量里准备新状态，所有步骤成功后再提交到全局状态，避免半更新。

### 13. `/ask` 在 dirty refresh 失败时没有暴露 degraded 状态

问题：

`/ask` 开头会尝试刷新 dirty knowledge，但 `_try_refresh_dirty_knowledge()` 内部把 refresh 状态丢掉了。结果 refresh 连续失败时，响应体里没有 `refresh_dirty` 和 `refresh_error`，用户或前端无法知道当前回答可能基于旧 runtime。

解决：

让 `_try_refresh_dirty_knowledge()` 返回 refresh state，并在 `/ask` 生成响应后统一附加 `_current_refresh_state()`。这样即使问答本身成功，也能暴露运行时索引刷新失败的 degraded 状态。

面试可讲点：

对 RAG 系统来说，答案是否成功和知识库是否最新是两个维度。即使能回答，也要告诉调用方“这个回答可能基于旧索引”。

## 测试策略

本次按 TDD 推进，每类问题都先补失败测试，再写最小实现修复。重点覆盖：

- 上传 PDF/DOCX/TXT/MD 后能解析、chunk、入库并检索。
- FAISS、sidecar、manifest 不一致时索引不可用。
- failed 文档可重新上传。
- status/list 不触发 embedding provider 初始化。
- delete reindex 失败时 DB 和文件可恢复。
- 文件清理失败不影响已提交删除和 runtime refresh。
- refresh 失败后响应暴露 `refresh_dirty` / `refresh_error`。
- `_refresh_orchestrators()` 避免半更新。
- Streamlit API client 使用现有 FakeHttpClient 测试风格，不引入额外 httpx mock 依赖。

## 简历可写版本

可以把这段写进简历：

实现 DataCenter-HVAC-Copilot 的持久化 RAG 知识库模块，支持 PDF/DOCX/TXT/MD 上传解析、SQLite 文档与 chunk 元数据管理、FAISS 索引持久化、sidecar 行映射和 manifest 校验；设计全量重建与原子发布流程，处理上传/删除/reindex 中的回滚、降级和运行时自愈刷新；通过 TDD 覆盖索引一致性、失败重试、删除恢复和 API/Streamlit 集成路径。

## 面试讲述版本

可以这样讲：

这个项目里我做的是一个面向数据中心暖通运维文档的长期记忆模块。用户可以把设备手册、运维规范、故障记录等文档上传到系统，系统解析后切成 chunks，保存到 SQLite，同时构建 FAISS 向量索引。之后用户在 `/ask` 提问时，系统会自动检索这些上传知识，再结合原有 HVAC 推理逻辑回答。

实现过程中最大的挑战不是“把文档向量化”，而是让知识库在上传、删除、重建、API 刷新失败时仍然保持一致。比如 FAISS 文件和 chunk sidecar 是两个文件，如果只替换其中一个成功，就会出现 citation 对不上 chunk 的问题。我后来加了 manifest，把 FAISS hash、sidecar hash、chunk count 作为一个提交标记，加载时强校验，不一致就降级为不可用。

另一个比较有代表性的问题是删除流程。一开始删除是先删文件和数据库，再重建索引。如果重建失败，旧文档已经没了，无法恢复。后来我把删除改成可回滚流程：先 snapshot DB 里的 document/chunks，重建成功后才清理源文件；如果 reindex 失败，就恢复 DB 和 chunks，保证旧知识库仍然可用。

还有一个运行时一致性问题：知识库持久化成功后，API 内存里的 orchestrator 还要刷新。如果刷新失败，不能让上传接口直接 500，因为数据其实已经提交了；也不能沉默失败，否则 `/ask` 会长期用旧索引。所以我加了 dirty/degraded 状态，写操作成功后标记 dirty，刷新失败就把错误暴露给 `/ask` 和 `/knowledge/status`，后续请求会继续尝试自愈刷新。

最后，我用 TDD 给这些异常路径都补了回归测试。这个模块的重点不是 demo 式 RAG，而是把文档知识库当作一个有状态系统来做：有事实来源、有派生索引、有提交标记、有回滚、有降级状态，也有运行时自愈。

## 可强调的工程亮点

- 把 SQLite 作为事实来源，FAISS 作为可重建派生产物，降低索引损坏后的恢复成本。
- 用 manifest 解决多文件索引产物的一致性和版本提交问题。
- 用 lazy embedding provider 避免 status/list 这类轻接口触发重模型加载。
- 对上传、删除、重建分别定义失败语义，避免 broad exception 造成状态误判。
- 将 runtime refresh 与持久化提交解耦，通过 dirty/degraded 状态实现自愈。
- 用测试覆盖异常路径，而不是只测 happy path。

