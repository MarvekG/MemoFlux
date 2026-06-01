# MemoFlux Session 隔离设计

## 目标

MemoFlux 不再使用 `scope` 概念，统一使用 `session` 作为唯一隔离边界。代码、API、数据库、审计、测试和文档中都不保留 `scope` 字段或兼容逻辑。

## 背景

当前 MemoFlux 使用调用方传入的 `scope` 做强隔离。当同一个 `scope` 内写入多个项目、主题或业务主体的相似记忆时，pgvector `top_k` 会召回语义相近但主题不同的候选。虽然 Answer Synthesizer 可以通过 `used_memory_ids` 过滤对外 references，但候选噪声仍会增加误判风险。

新的隔离语义改为会话粒度。调用方必须把每次写入、召回、删除、预览和审计查询绑定到一个精确 `session`。MemoFlux 不解释 session 层级，不做跨 session 查询，不从 query 推断 session。

## API 设计

所有业务 API 只接受 `session` 字段或 query 参数：

- `POST /v1/ingest` 请求体包含 `session`、`content`、`occurred_at`。
- `POST /v1/recall` 请求体包含 `session`、`query`、`include_references`、`top_k`。
- `POST /v1/delete` 请求体包含 `session`、`memory_ids`、`query`、`dry_run`。
- `GET /v1/preview?session=...`。
- `GET /v1/audits?session=...&query_id=...`。

请求体继续使用 `extra="forbid"`。旧请求如果传入 `scope`，应返回 422，不做兼容映射。

响应中所有隔离字段都命名为 `session`，不返回 `scope`。

## 数据模型

领域模型字段替换为：

- `MemoryRecord.session`
- `QueryAuditRecord.session`
- `DeleteAuditRecord.session`

Repository 接口替换为：

- `insert_memory(session=...)`
- `search_memories(session=...)`
- `delete_memories(session=...)`
- `list_memories(session=...)`
- `list_audits(session=...)`
- `scrub_deleted_memory_from_audits(session=...)`

Service 层同样只接收 `session`。所有校验错误文案使用 `session is required`。

## 数据库结构

PostgreSQL ORM 模型不保留 `scope` 列，直接定义 `session` 列：

- `memories.session text not null`
- `memory_query_audits.session text not null`
- `memory_delete_audits.session text not null`

索引同步改名：

- `ix_memoflux_memories_session_occurred_at`
- `ix_memoflux_memories_session_created_at`
- `ix_memoflux_query_audits_session_created_at`
- `ix_memoflux_delete_audits_session_created_at`

候选检索、删除、预览和审计查询都必须按精确 `session` 过滤。

当前项目未上线，不在应用启动代码中加入一次性迁移或兼容逻辑。开发环境已有旧表时，由开发者手动清理并重建 `memoflux` schema，或手动执行列和索引 rename SQL。目标代码只描述新结构。

## 文档更新

更新 `memo/docs/001-memoflux-design.md` 和 `memo/README.md`：

- 将隔离边界描述为 `session`。
- 删除 `scope` 术语和示例。
- 明确 MemoFlux 不支持跨 session 查询、session 继承或自动扩展 session。
- 更新 100 条相似记忆评估发现，说明会话粒度隔离用于降低同一隔离边界内多主体混放造成的误判。

## 测试策略

更新 `memo/tests/test_api.py`：

- 所有请求从 `scope` 改为 `session`。
- `test_audits_are_scope_isolated_and_support_query_id_lookup` 改名为 session 隔离测试。
- 断言响应和 audit payload 返回 `session`。
- 增加或保留测试：旧 `scope` 字段请求返回 422，确保不存在兼容入口。

验证命令：

```bash
cd memo && pytest tests/test_api.py
cd memo && git diff --check
```

如需验证容器运行时结构，重建 `memo` 服务并执行 `/v1/ingest`、`/v1/recall`、`/v1/audits` smoke。若数据库仍有旧 `scope` 列结构，先手动重建 `memoflux` schema。

## 非目标

- 不实现跨 session 查询。
- 不兼容旧 `scope` API。
- 不在应用启动时自动迁移旧数据库结构。
- 不引入关键词规则来判断主体或会话关系。
