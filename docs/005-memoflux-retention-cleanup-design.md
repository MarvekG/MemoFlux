# MemoFlux 记忆保留期清理设计

## 目标

MemoFlux 新增一个独立后台任务，每天自动删除超过保留期的记忆。清理规则按 `memories.occurred_at` 判断，默认保留 180 天，可通过环境变量配置。

该机制用于控制长期记忆噪声和存储规模，避免过早引入复杂的 LLM 过时判断、软删除状态或人工确认流程。

## 非目标

- 不新增软删除、归档、恢复或 `outdated` 状态。
- 不让 LLM 判断哪些记忆过时。
- 不按 `created_at` 判断过期，避免写入时间影响经验时效性。
- 不新增独立 pgvector 表；embedding 仍随 `memories` 行一起删除。
- 不改变 `/v1/recall`、`/v1/preview`、`/v1/delete` 的外部语义。

## 保留期语义

过期判断使用事件发生时间：

```text
expired if memories.occurred_at < now_utc - retention_days
```

默认配置：

```text
retention_days = 180
```

选择 `occurred_at` 的原因：MemoFlux 要回答历史和因果问题，记忆时效性应由事实发生时间决定，而不是由补录时间决定。补录 1 年前的旧经验时，如果保留期是 180 天，该记忆会在下一次清理时被删除。

## 配置

新增 MemoFlux 环境变量：

```bash
MEMOFLUX_MEMORY_CLEANUP_ENABLED=true
MEMOFLUX_MEMORY_RETENTION_DAYS=180
MEMOFLUX_MEMORY_CLEANUP_HOUR=4
MEMOFLUX_MEMORY_CLEANUP_MINUTE=15
```

边界规则：

- `MEMOFLUX_MEMORY_CLEANUP_ENABLED` 默认为 `true`。
- `MEMOFLUX_MEMORY_RETENTION_DAYS` 默认为 `180`，允许范围 `1..3650`。
- `MEMOFLUX_MEMORY_CLEANUP_HOUR` 默认为 `4`，允许范围 `0..23`。
- `MEMOFLUX_MEMORY_CLEANUP_MINUTE` 默认为 `15`，允许范围 `0..59`。
- 无法解析的配置值回退到默认值，并在启动或执行时记录日志。

## 调度设计

MemoFlux 服务当前没有中心调度器。第一版使用 FastAPI lifespan 启动一个轻量级 `asyncio` 后台循环：

```text
create_app lifespan startup
  -> load settings
  -> if cleanup enabled: create_task(memory_retention_cleanup_loop)

memory_retention_cleanup_loop
  -> 计算下一次配置时间
  -> sleep 到执行时间
  -> 调用 service.cleanup_expired_memories(retention_days)
  -> 循环等待下一天

lifespan shutdown
  -> cancel cleanup task
```

任务失败不能导致 MemoFlux 服务退出。单次执行异常只记录日志，并等待下一次调度。

## 删除流程

清理任务复用 MemoFlux 现有“删除就是遗忘”的硬删除语义：

```text
cutoff = now_utc - retention_days

repository.delete_memories_before_occurred_at(cutoff)
  -> 按 session 查询 occurred_at < cutoff 的 memory_id
  -> 从 memories 删除匹配行
  -> 返回 {session: [memory_id, ...]}

service.cleanup_expired_memories(retention_days)
  -> 对每个 session 调 scrub_deleted_memory_from_audits(session, memory_ids)
  -> 对每个 session 写 memory_delete_audits
  -> 返回删除数量、session 数、retention_days 和 cutoff
```

因为 `memories.embedding` 和原文在同一行，删除 `memories` 行会同时删除原文和 pgvector 派生向量。

## 审计设计

`memory_delete_audits.session` 当前必填，因此 TTL 清理按 session 分组写审计，不新增跨 session 审计记录。

每个 session 写一条 delete audit：

```json
{
  "target": {
    "mode": "retention_cleanup",
    "retention_days": 180,
    "cutoff_occurred_at": "2025-12-03T04:15:00Z"
  },
  "dry_run": false,
  "matched_memory_ids": ["..."],
  "affected_memory_ids": ["..."],
  "status": "ok"
}
```

审计记录只保存删除目标、匹配 ID 和清理参数，不保存被删除记忆正文。

清理完成后必须继续执行现有 audit scrub：

- 历史 query audit 中被删除 memory 的 `retrieved[].content_preview` 置为 `null`。
- `selection_reasons` 中对应 memory_id 的理由被移除。
- 如果被删除 memory 曾参与最终回答，`final_answer` 置为 `null`。

## Repository 接口

新增 repository 方法：

```python
delete_memories_before_occurred_at(cutoff: datetime) -> dict[str, list[str]]
```

返回值按 session 分组：

```python
{
    "project:atlas": ["memory-id-1", "memory-id-2"],
    "user:1:stock:600519": ["memory-id-3"],
}
```

实现要求：

- 只匹配 `occurred_at < cutoff`。
- 删除和返回 ID 应在同一个数据库事务中完成。
- 空结果返回空字典。
- 内存 repository 用相同行为支持测试。

## Service 接口

新增 service 方法：

```python
cleanup_expired_memories(retention_days: int | None = None) -> dict[str, Any]
```

返回示例：

```json
{
  "status": "ok",
  "deleted": 15,
  "sessions": 3,
  "retention_days": 180,
  "cutoff_occurred_at": "2025-12-03T04:15:00Z"
}
```

该方法用于后台任务，也可被单元测试直接调用。第一版不新增公开 HTTP 清理接口，避免把批量破坏性操作暴露为 API。

## 与现有 API 的关系

- `/v1/recall`：不需要改；过期记忆被删除后自然不再召回。
- `/v1/preview`：不需要改；过期记忆不会出现在预览结果中。
- `/v1/delete`：继续处理用户显式删除，语义不变。
- `/v1/audits`：会显示 TTL 清理产生的 delete audit，但不会暴露被删除正文。
- `/v1/usage/stats`：TTL 清理不记录 LLM 用量，因为不调用 LLM。

## 错误处理和日志

- 配置关闭时启动日志记录 `memory retention cleanup scheduler is disabled`。
- 单次清理成功后记录删除条数、session 数、保留天数和 cutoff。
- 单次清理失败时记录异常和配置上下文，不终止服务。
- shutdown 时取消后台任务；取消异常不记录为错误。

## 测试策略

单元测试覆盖：

- 插入 181 天前和 10 天前的记忆，执行清理只删除旧记忆。
- 过期判断使用 `occurred_at`，不使用 `created_at`。
- 多 session 过期记忆按 session 分组删除，并分别写 delete audit。
- 清理后历史 query audit 的 `content_preview`、`selection_reasons` 和必要时的 `final_answer` 被 scrub。
- 无过期记忆时返回 `deleted=0`，不写 delete audit。
- 配置值越界或无法解析时回退到默认值或边界值。
- 后台任务 disabled 时不启动 cleanup loop。

验证命令：

```bash
cd memo && pytest tests/test_api.py
cd memo && git diff --check
```

## 迁移注意

当前数据库结构无需新增表或字段。只需要部署新代码并配置环境变量。

如果生产或公开环境需要更严格的数据保留策略，应显式设置 `MEMOFLUX_MEMORY_RETENTION_DAYS`，不要依赖默认值。
