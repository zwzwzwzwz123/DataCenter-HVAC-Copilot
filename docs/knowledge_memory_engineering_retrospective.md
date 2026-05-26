# Persistent Knowledge Memory Engineering Retrospective

> 面试讲述定位：这是 DataCenter-HVAC-Copilot 中的“长期知识记忆”能力。用户可以上传 PDF/DOCX/TXT/Markdown 运维文档，系统解析、切块、持久化元数据，并把向量索引写入 FAISS。后续 `/ask` 默认从这套上传知识库检索证据。

## 一句话项目描述

我为 DataCenter-HVAC-Copilot 实现了一套持久化知识库记忆机制：用 SQLite 保存 document/chunk/index 元数据，用 FAISS 保存向量索引，用 `chunks.jsonl` 保存 FAISS row 到 chunk metadata 的 sidecar 映射，并把上传、删除、重建后的知识库自动接入 RAG 问答链路。

这不是一个轻量 demo。实现过程中重点处理了工程里最容易被忽略的问题：索引和元数据一致性、失败回滚、原子替换、模型懒加载、重复上传去重、API 运行时刷新，以及异常场景下的可观测状态。

## 核心架构

存储分三层：

- `SQLite`: 保存文档、chunk、索引版本等结构化元数据，是业务侧 source of truth。
- `FAISS`: 保存 dense vector index，路径为 `data/knowledge/faiss/index.faiss`。
- `chunks.jsonl`: 保存 FAISS row 到 chunk metadata 的映射，保证检索结果能还原 citation。

索引目录里还增加了 `manifest.json`，记录 `index.faiss` 和 `chunks.jsonl` 的 sha256、chunk_count、embedding provider/model 等信息，用于验证磁盘文件是否成对一致。

API 层提供：

- 上传：`POST /knowledge/documents/upload`
- 列表：`GET /knowledge/documents`
- 单文档元数据：`GET /knowledge/documents/{document_id}`
- 状态：`GET /knowledge/status`
- 重建：`POST /knowledge/reindex`
- 删除：`DELETE /knowledge/documents/{document_id}`

运行时 `/ask` 会在知识库可用时优先使用上传后的持久化 FAISS 检索器，否则回退到项目自带的静态 demo 文档。

## 遇到的问题与解决方案

### 1. 上传文件名可能造成路径穿越

**问题**

上传接口一开始直接使用客户端传来的 `file.filename` 作为临时文件名。浏览器或恶意客户端可能传入 `../escape.md`、`C:\temp\escape.md` 这类路径，导致临时文件拼接时有路径穿越风险。

**解决**

在 API 层增加安全文件名处理：

- 先把反斜杠统一成 `/`。
- 只取 basename。
- 只保留字母、数字、点、下划线、横线等安全字符。

这样即使客户端传入路径，也只会保存成安全的文件名。

**面试可讲点**

“文件上传不能信任客户端 filename。我在进入持久化逻辑前做 basename 提取和字符白名单清洗，避免路径穿越和奇怪平台路径造成的问题。”

### 2. FAISS 和 sidecar 双文件替换容易不一致

**问题**

FAISS 本身只存向量，不存业务 metadata。业务 citation 依赖 `chunks.jsonl`。如果 `index.faiss` 替换成功，但 `chunks.jsonl` 替换失败，或者进程在两个文件之间崩溃，就会出现 FAISS row 和 chunk metadata 错配，最坏情况下会返回错误 citation。

**解决**

增加 `manifest.json`：

- 记录 `index.faiss` sha256。
- 记录 `chunks.jsonl` sha256。
- 记录 `chunk_count`。
- retriever 加载时校验 sha256 和行数一致性。
- 不一致时标记索引 unavailable，搜索返回空结果，而不是返回错误 citation。

同时 rebuild 时使用临时文件，写完后再替换正式文件，并在失败时从备份恢复。

**面试可讲点**

“我把 FAISS 和 sidecar 看成一个逻辑提交单元。因为文件系统没法天然保证多文件事务，所以我加了 manifest 做完整性校验。宁愿索引不可用，也不能返回错 citation。”

### 3. 备份/恢复流程本身也可能不安全

**问题**

早期备份逻辑用 `replace()` 把正式文件移动到 `.bak`。如果备份第一个文件成功、第二个文件失败，正式文件可能已经被移走，只剩 `.bak`，系统处于不清晰状态。

**解决**

把备份改为 copy 型备份：

- `_backup_existing()` 使用 `shutil.copy2()`，不移动正式文件。
- 只有替换过的目标才在失败时恢复。
- 临时文件和 `.bak` 文件在 finally 中清理。

这样备份阶段失败不会先破坏正式文件。

**面试可讲点**

“我后来意识到 backup 自己也是写路径的一部分，不能让 backup 动作先破坏线上文件。所以把 move backup 改成 copy backup，并加 replaced 标志控制恢复。”

### 4. Python 版本兼容性问题

**问题**

项目声明支持 Python 3.10，但代码里用了 `datetime.UTC`，这是 Python 3.11 才有的 API。

**解决**

统一改成：

```python
from datetime import timezone
datetime.now(timezone.utc)
```

**面试可讲点**

“这种问题不复杂，但工程上很常见。依赖和 Python 版本声明必须和代码 API 保持一致，否则 CI 或用户机器会炸。”

### 5. status/list 不应该初始化 embedding model

**问题**

`GET /knowledge/status` 和 `GET /knowledge/documents` 只是读 SQLite 和磁盘元数据，不应该加载 sentence-transformers/BGE。早期实现只要发现索引文件存在，就构造 retriever，间接触发 embedding provider 加载。

这会导致：

- 状态接口变慢。
- 未安装 dense extras 时 status 也可能失败。
- 模型下载失败会影响只读接口。

**解决**

让 `status()` 只读取：

- `manifest.json`
- `chunks.jsonl`
- 文件 hash
- SQLite document/chunk count

只有 `reindex()` 和真正 `search()` 时才构造 embedding provider。

API 的 RAG pipeline 也改成 lazy retriever：启动时先检查 status，真正搜索时再初始化 embedding provider。

**面试可讲点**

“我把控制面和数据面分开了。status/list 是控制面，只读元数据；embedding model 属于数据面，只在索引或检索时加载。”

### 6. 失败文档会被 hash dedup 卡住，无法重试

**问题**

文档按 file_hash 去重。如果第一次上传 PDF/DOCX 因缺依赖或解析失败，数据库里会留下 failed 记录。第二次安装依赖后上传同一个文件，如果 dedup 直接按 hash 命中旧 failed 记录，就无法重试。

后续又发现一个更细的问题：第一次 failed，第二次成功 indexed，第三次同文件上传时，查询仍可能按最早 created_at 返回 failed 记录，导致重复 ingest。

**解决**

`find_document_by_hash()` 查询时优先返回最新的非 failed 记录：

```sql
ORDER BY
  CASE WHEN status = 'failed' THEN 1 ELSE 0 END,
  created_at DESC
```

如果只有 failed 记录，允许重试；如果已有 indexed 记录，则正常 dedup。

**面试可讲点**

“去重不能只看 hash，还要看状态。failed 不是一个可复用结果，indexed 才是可复用结果。”

### 7. ingest 过程中 reindex 失败会导致 SQLite 与 FAISS 不一致

**问题**

ingest 流程是：

1. parse document
2. 写 document
3. 写 chunks
4. rebuild FAISS

如果 chunks 已写入 SQLite，但 FAISS rebuild 失败，SQLite 里会有 indexed 文档和 chunks，而磁盘索引没有更新。

**解决**

在 ingest 的 broad exception 中清理该 document 的 chunks，并把 document 标为 failed。这样失败文档不会留在可检索 chunks 中。

后续又发现更细的边界：如果 FAISS rebuild 已成功，但只是 `save_index_status()` 写 SQLite index_versions 失败，不应该把文档标 failed，因为业务索引和 chunks 已经是成功状态。

于是把 `save_index_status()` 失败降级为 `metadata_error`，返回给调用方，但不回滚 indexed 文档。

**面试可讲点**

“我把关键路径和非关键 metadata 写入分开了。FAISS rebuild 成功意味着业务索引成功，index_versions 写失败只是审计元数据失败，不能把整个 ingest 判成失败。”

### 8. delete 先删 DB/files 再 reindex，失败后不可恢复

**问题**

早期 delete 流程是：

1. 删除上传源文件和 parsed 文件
2. 删除 SQLite document/chunks
3. rebuild FAISS

如果第 3 步失败，SQLite 和源文件已经没了，但旧 FAISS 仍可能引用这些 chunk，形成悬空 citation。

**解决**

把 delete 改成可回滚流程：

1. 先 snapshot document 和 chunks。
2. 删除 DB 记录。
3. 执行 reindex。
4. 如果 reindex 失败，恢复 document 和 chunks。
5. 只有 reindex 成功后才真正删除 source/parsed 文件。

**面试可讲点**

“删除其实比上传更危险。上传失败最多多一条 failed 记录，删除失败可能造成索引引用不存在的文档。所以我把 delete 设计成先能恢复，再提交。”

### 9. 文件清理失败不应让已成功 delete 变成 API 500

**问题**

delete 的 DB 和 FAISS 已经成功更新后，如果最后清理 source/parsed 文件时遇到 Windows 文件占用、权限问题或杀毒锁文件，接口会抛 `PermissionError/OSError`。这会让客户端看到 500，但实际上知识库已经删除成功。

更糟的是，如果 API 因清理异常中断，后续 orchestrator refresh 不会执行，运行时 `/ask` 可能还持有旧索引。

**解决**

文件清理异常不再打断主流程：

- delete 成功返回。
- 清理失败信息放进 `cleanup_errors`。
- API 继续执行 orchestrator refresh。

**面试可讲点**

“我把主事务和善后清理分开了。DB/FAISS 删除成功是主结果，文件清理失败是 cleanup warning，应该暴露但不能把主流程判失败。”

### 10. upload/delete/reindex 提交成功后，orchestrator refresh 失败导致 API 500

**问题**

知识库写操作完成后会刷新内存里的 orchestrator，让 `/ask` 立刻使用新知识库。早期如果 refresh 失败，API 返回 500，但持久化操作已经提交成功，造成客户端语义混乱。

**解决**

把持久化提交结果和运行时 refresh 结果分离：

- upload/delete/reindex 成功后先返回业务结果。
- refresh 失败时不抛 500，而是返回 `refresh_error`。
- 同时维护 `knowledge_refresh_dirty` 和 `last_refresh_error` 状态。

**面试可讲点**

“这是典型的提交后副作用问题。持久化已经成功，不能因为运行时缓存刷新失败就告诉客户端操作失败。我把它降级成 degraded 状态。”

### 11. refresh 失败后没有自愈路径，`/ask` 会长期使用旧 orchestrator

**问题**

只是返回 `refresh_error` 还不够。如果 refresh 因短暂问题失败，后续 `/ask` 会一直使用旧 orchestrator，直到下一次写操作触发 refresh。

**解决**

增加 dirty refresh 状态机：

- 写操作提交后标记 `knowledge_refresh_dirty=True`。
- refresh 成功则清 dirty 和 error。
- refresh 失败则保留 dirty 和 last_refresh_error。
- `/ask` 开头如果 dirty，会尝试 refresh 一次。
- `/knowledge/status` 也会尝试 refresh，并返回 `refresh_dirty`/`refresh_error`。

**面试可讲点**

“我没有只把错误丢给用户，而是加了自愈机制。下一次读请求会尝试修复运行时缓存，同时如果仍失败，会明确返回 degraded 状态。”

### 12. `/ask` refresh 失败时没有暴露 degraded 状态

**问题**

`/ask` 开头会尝试 dirty refresh，但失败状态只存在闭包变量里，响应体没有带上 `refresh_dirty` 和 `refresh_error`。调用方无法知道这次回答可能来自旧 orchestrator。

**解决**

在 `/ask` 返回体里合并 `_current_refresh_state()`：

- `refresh_dirty`
- `refresh_error`

这样即使回答仍可生成，调用方也能知道运行时知识库缓存处于 degraded 状态。

**面试可讲点**

“降级不是静默降级。尤其 RAG 系统里，回答来自旧知识库和新知识库差别很大，必须把 degraded 状态返回给调用方。”

### 13. refresh 半成功会导致 deterministic/langgraph orchestrator 状态不一致

**问题**

早期 `_refresh_orchestrators()` 是先赋值 deterministic orchestrator，再构造 langgraph orchestrator。如果第二步失败，会出现：

- deterministic 已经使用新知识库
- langgraph 仍使用旧实例
- dirty 状态又显示 refresh 失败

这会造成同一个服务内两条 workflow engine 的知识状态不一致。

**解决**

改成局部变量构建：

```python
new_orchestrator = build_demo_orchestrator(...)
new_langgraph_orchestrator = LangGraphOrchestrator(new_orchestrator, ...)
orchestrator = new_orchestrator
langgraph_orchestrator = new_langgraph_orchestrator
```

只有两个对象都构建成功，才一次性更新闭包变量。

**面试可讲点**

“这是内存状态的原子更新问题。虽然不是数据库事务，但也要避免半更新。我用局部变量构建完成后再提交到运行时状态。”

## 我补的关键测试

这套机制不是靠手测兜底，而是用 TDD 一步步补回归测试。比较关键的测试包括：

- 上传同文件 hash 去重。
- failed 文档可重试。
- failed 后成功，再次上传能 dedup 到成功记录。
- reindex 失败时清理 chunks。
- index status metadata 写失败时不把 indexed 文档标 failed。
- delete reindex 失败时恢复 DB/chunks/files。
- delete 文件清理失败时返回 `cleanup_errors`。
- refresh 失败时 upload/delete/reindex 不返回 500，而返回 `refresh_error`。
- refresh dirty 后 `/ask` 可以自愈刷新。
- refresh 连续失败时 `/ask` 返回 `refresh_dirty=True`。
- refresh 半失败时 deterministic/langgraph 不半更新。
- retriever 加载时校验 FAISS row count 和 sidecar row count。
- manifest hash 不一致时索引 unavailable。
- status/list 不加载 embedding provider。

## 面试讲述版本

如果面试官问“这个项目里你遇到过什么有挑战的问题”，可以这样讲：

> 我做这个项目时最有挑战的是把 RAG 的知识库从 demo 变成可长期运行的持久化系统。上传文档本身不难，难的是失败场景下 SQLite、FAISS、sidecar、API 运行时缓存这几层要保持一致。
>
> 我遇到过几个典型问题。比如 FAISS 只存向量，citation metadata 在 sidecar 里，如果两个文件替换时只成功一个，就会返回错 citation。我的解决是引入 manifest，记录两个文件的 sha256 和 chunk_count，retriever 加载时严格校验，不一致就标记 unavailable。
>
> 还有 delete 流程，一开始是先删 DB 和文件再 rebuild FAISS。如果 rebuild 失败，旧索引会引用已经删除的文档。我后来改成先 snapshot DB 记录和 chunks，删除后 reindex，如果失败就恢复 SQLite；只有 reindex 成功后才删除源文件。
>
> 另外 API 层也有运行时一致性问题。上传或删除后需要刷新内存里的 orchestrator。如果 refresh 失败，持久化其实已经成功了，不能直接给用户 500。我把持久化提交和运行时 refresh 分开，refresh 失败返回 degraded 状态，并设置 dirty 标志。后续 `/ask` 或 `/knowledge/status` 会尝试自愈刷新，仍失败则把 `refresh_dirty` 和 `refresh_error` 暴露给调用方。
>
> 最后我还处理了重复上传、failed 文档重试、embedding provider 懒加载、Python 版本兼容、文件名路径穿越等边界问题。整体上，这个功能让我学到的不是“怎么接 FAISS”，而是怎么把一个 RAG demo 做成可以承受异常和重启的工程化知识记忆系统。

## 简历 bullet 可参考

- Built a persistent RAG knowledge memory subsystem supporting PDF/DOCX/TXT/Markdown ingestion, SQLite metadata storage, FAISS indexing, and citation-preserving retrieval.
- Designed a manifest-backed FAISS persistence protocol with sha256 and row-count validation to prevent vector/metadata mismatch and stale citations.
- Implemented failure-safe upload/delete/reindex workflows with rollback, cleanup warning reporting, and degraded runtime refresh states.
- Added lazy embedding-provider initialization so status/list endpoints remain lightweight and do not require dense model loading.
- Added regression tests for index corruption, failed-document retry, delete rollback, refresh failure recovery, and runtime orchestrator consistency.

## 最后总结

这个功能的核心价值不是“能上传文档”，而是让 Copilot 拥有一个可靠的长期知识记忆层。真正体现工程能力的地方在于：

- 多文件持久化的一致性设计。
- SQLite 和 FAISS 之间的失败恢复。
- RAG citation 的可靠性。
- 大模型/embedding 依赖的懒加载。
- API 提交结果和运行时缓存刷新结果的解耦。
- 异常状态可观测，而不是静默失败。

