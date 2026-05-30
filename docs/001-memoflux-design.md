# MemoFlux 设计文档

## 1. 定位

MemoFlux 是一个长期记忆服务，用于保存原始记忆文本，通过调用方显式传入的 `scope` 做强隔离，并通过 LLM 驱动的检索链路回答自然语言问题。

MemoFlux 按独立、长期演进的服务设计，而不是某个项目内部的辅助模块。第一版刻意保持业务数据模型极简，把复杂语义判断集中放在 LLM query rewrite 和答案整合阶段，同时通过详细审计日志保证可排查、可评估、可迭代。

## 2. 目标

- 以文本形式保存原始记忆内容，不切片、不抽取实体、不保存关系边、不保存因果边、不生成持久化摘要。
- 使用调用方显式传入的 `scope` 做严格隔离，不同 `scope` 之间的记忆互相不可见。
- 通过同一条三层查询链路支持直接查询、关联查询、历史查询和前因后果查询。
- 每条记忆必须提供事件发生时间 `occurred_at`，确保历史和因果回答按事件时间组织，而不是按写入时间误排。
- 原始记忆内容写入后不可修改，纠错通过追加新记忆表达。
- 支持彻底删除记忆；这里的“删除”和“遗忘”是同一个语义，都是不可恢复地删除原始记忆正文和派生检索数据。
- 第一版引入 pgvector 保存 embedding，用 LLM query rewrite、向量召回、PostgreSQL 文本/时间兜底和 LLM 答案整合验证效果。
- 记录足够详细的审计日志，用于定位 query rewrite、候选检索、LLM 答案整合、延迟和失败原因。
- LLM 用量不在单次业务响应中返回，也不提供按请求开启返回用量的开关；用量只通过聚合 usage stats 接口查询和清空。

## 3. 非目标

- 第一版不构建 entity-centered memory graph。
- 不持久化实体、事件、关系边、因果边或记忆切片。
- 不支持跨 `scope` 查询、`scope` 继承或自动扩展 `scope`。
- 不允许更新原始记忆文本；删除是唯一例外，它会不可恢复地移除原文。
- 不引入 LlamaIndex 或独立向量库；向量检索限定在 PostgreSQL pgvector 内，避免新增运行时依赖。
- 不用关键词、正则、白名单、黑名单或硬编码规则模拟语义判断、因果判断或相关性判断。
- 不提供批量写入接口；调用方需要多条写入时逐条调用 `/v1/ingest`。

## 4. 核心概念

### 4.1 Scope

`scope` 是调用方提供的不透明字符串。MemoFlux 只校验它存在，并在写入、读取、召回、候选检索、删除和审计查询中使用精确匹配。

示例：

```text
user:123:general
user:123:stock:600000.SH
org:alpha:project:atlas
```

MemoFlux 不解释 `scope` 的层级含义。查询 `user:123` 不会看到 `user:123:stock:600000.SH` 的记忆，除非调用方一开始就把相关记忆写入同一个精确 `scope`。

### 4.2 Memory

一条记忆是一条不可变的原始文本记录，并且必须带事件发生时间。

```text
memory_id
scope
content
occurred_at
created_at
```

`occurred_at` 必填。如果调用方无法提供，写入请求直接失败。MemoFlux 不使用 LLM 推断缺失的 `occurred_at`。

### 4.3 候选检索

第一版候选检索使用 PostgreSQL pgvector，并保留文本/时间检索作为降级和诊断能力：

```text
LLM Query Planner
  -> 生成 rewritten queries、关键词组、时间意图
  -> embedding 生成查询向量
  -> PostgreSQL pgvector 最近邻召回
  -> PostgreSQL 文本检索 / ILIKE / 时间排序兜底
  -> occurred_at 时间排序和窗口过滤
  -> LLM Answer Synthesizer
```

候选检索只负责在一个精确 `scope` 内找出可能相关的原始 memory。最终答案仍由 LLM 基于回填后的 `memories.content` 生成。

第一版在 `memories.embedding` 保存派生向量。原始 `content` 仍是事实来源；embedding 只用于同一 `scope` 内候选召回，删除记忆时随原文一起硬删除。

## 5. 数据模型

### 5.1 `memories`

```text
memory_id: UUID primary key
scope: text not null
content: text not null
embedding: vector(768) null
occurred_at: timestamptz not null
created_at: timestamptz not null
```

建议索引：

```text
(scope, occurred_at)
(scope, created_at)
```

规则：

- `content` 不允许更新。
- `occurred_at` 不允许更新。
- 删除是硬删除：从 `memories` 移除整条业务记录，并清理相关派生检索数据。
- 纠错通过新增 memory 表达。

### 5.2 检索派生数据

第一版不新增独立向量表，直接在 `memories.embedding` 保存派生向量，并依赖以下数据库能力：

- `(scope, occurred_at)` 支持同一 scope 内的历史查询和时间窗口过滤。
- `(scope, created_at)` 支持预览和调试分页。
- pgvector `embedding` 支持同一 scope 内最近邻召回。
- PostgreSQL 文本检索能力支持 rewritten query、关键词组和原文匹配兜底。

如果后续引入 LlamaIndex 或其他 node store，新增存储必须是派生数据，不能替代 `memories` 作为事实来源。

### 5.3 `memory_query_audits`

查询审计需要足够详细，能够定位三层查询链路中的问题。

```text
query_id: UUID primary key
scope: text not null
original_query: text not null
query_type: text not null
query_type_reason: text null
rewritten_queries: jsonb not null
candidate_limit: integer not null
candidate_filters: jsonb not null
retrieved: jsonb not null
selected_memory_ids: uuid[] not null
final_answer: text null
include_references: boolean not null
llm_usage: jsonb null
timings_ms: jsonb not null
error_stage: text null
error_message: text null
created_at: timestamptz not null
```

`retrieved` 保存候选诊断信息，不保存完整记忆正文：

```json
[
  {
    "memory_id": "...",
    "occurred_at": "2026-05-01T10:00:00Z",
    "score": 0.82,
    "source_query_index": 0,
    "content_preview": "前 300 字..."
  }
]
```

审计日志不是记忆事实来源。完整证据始终通过 `memory_id` 回查 `memories`。

审计日志可以在内部保存 LLM 用量，用于聚合统计与排障，但单次业务响应和 `/v1/audits` 响应不返回单次 token 消耗。

### 5.4 `memory_delete_audits`

删除需要单独审计，避免不可逆操作无法追踪。

```text
delete_id: UUID primary key
scope: text not null
target: jsonb not null
dry_run: boolean not null
matched_memory_ids: uuid[] not null
affected_memory_ids: uuid[] not null
status: ok | rejected | failed
error_code: text null
error_message: text null
created_at: timestamptz not null
```

`target` 只记录调用方请求和匹配条件，不保存完整记忆正文。删除成功后，审计中也不能残留被删除记忆的原文、quote 或 content preview。

## 6. 写入流程

```text
POST /v1/ingest
  -> 校验 scope/content/occurred_at
  -> 使用配置的 embedding provider 生成向量
  -> 写入 immutable memories 记录
  -> 返回 memory_id
```

校验规则：

- `scope` 必填。
- `content` 必填。
- `occurred_at` 必填。
- 缺少 `occurred_at` 直接返回校验错误，不写入任何数据。
- 服务不使用 LLM 推断缺失的 `occurred_at`。

写入落到 `memories` 事实表，并保存同一行的派生 embedding。embedding 生成失败时写入失败，避免记忆进入不可召回的半完成状态。

### 6.1 删除流程

MemoFlux 的删除就是遗忘，语义是不可恢复地删除原始记忆：

- 从 `memories` 删除整条业务记录。
- 删除相关检索派生数据；当前实现中 embedding 与原文在同一行，硬删除 `memories` 即同时清除原文和向量。
- 清理或重写相关审计中的 `content_preview`、quote、final answer 等可能包含原文的字段。
- 删除不可恢复。

按 `memory_id` 删除是确定性操作，不需要 LLM。按自然语言描述寻找待删除记忆时，可以通过 LLM query planner、向量召回和文本/时间检索生成候选，但不可直接执行删除；调用方必须先 dry-run 查看候选，再用明确 `memory_ids` 执行删除。

## 7. 查询流程

MemoFlux 使用固定三层查询架构：

```text
用户 query + 精确 scope
  -> LLM Query Planner
  -> Candidate Retriever
  -> LLM Answer Synthesizer
```

### 7.1 第一层：LLM Query Planner

Planner 接收原始 query，输出结构化结果：

```json
{
  "query_type": "direct|related|history|causal",
  "reason": "...",
  "rewritten_queries": [
    {
      "query": "...",
      "purpose": "direct|related|history|cause|effect|background"
    }
  ],
  "time_order": "none|ascending|descending",
  "answer_focus": "...",
  "uncertainties": ["..."]
}
```

Planner 自动判断查询类型。第一版调用方不传 `query_type`。

Planner 可以为一次用户 query 生成多个 rewritten query。这样可以在保持存储极简的前提下，提高关联查询、历史查询和前因后果查询的召回率。

### 7.2 第二层：Candidate Retriever

MemoFlux 对每个 rewritten query 执行同一 `scope` 内的候选检索。

必需过滤条件：

```text
scope == request.scope
```

第一版候选检索优先使用 pgvector 最近邻召回，并保留 PostgreSQL 文本匹配、可选全文检索、时间窗口过滤和 `occurred_at` 排序作为兜底与诊断能力。Retriever 合并所有 rewritten query 的候选，按 `memory_id` 去重，再从 `memories` 回填完整 memory，最后把候选集合交给答案整合层。

排序规则：

- 直接查询和关联查询主要按文本匹配度、时间约束和 LLM planner 给出的检索意图组织候选。
- 历史查询需要召回足够候选，并按 `occurred_at` 提供给答案整合层。
- 前因后果查询在候选支持时，应包含目标事件前后的相关记忆，并按 `occurred_at` 提供给答案整合层。

Retriever 不允许为了补充背景而搜索其他 `scope`。

### 7.3 第三层：LLM Answer Synthesizer

Synthesizer 接收：

```text
原始 query
query planner 输出
回填后的 memory 候选
只能基于候选记忆回答的严格指令
引用证据输出要求
```

输出结构化结果：

```json
{
  "answer": "...",
  "query_type": "direct|related|history|causal",
  "references": [
    {
      "memory_id": "...",
      "occurred_at": "...",
      "quote": "...",
      "relevance": "..."
    }
  ],
  "confidence": 0.0,
  "uncertainties": ["..."]
}
```

默认返回引用证据。调用方可以设置 `include_references=false`，让 API 响应体不返回引用；但 MemoFlux 仍应在审计记录中保存引用和候选信息，便于排障。

## 8. 查询类型

### 8.1 直接查询

直接查询用于回答可以由直接相关记忆支撑的具体问题。

示例意图：

```text
这个项目当前的阻塞是什么？
```

预期行为：

- 将用户 query 改写成更适合文本检索的表达。
- 在同一 `scope` 内召回最强候选。
- 只基于召回记忆回答。
- 引用被使用的记忆。

### 8.2 关联查询

关联查询用于查找语义相关记忆，即使这些记忆不直接回答字面问题。

示例意图：

```text
之前有哪些问题和这次部署风险相关？
```

预期行为：

- 生成多个相关 rewritten query。
- 在同一 `scope` 内召回语义相近记忆。
- 解释每条记忆为什么相关。
- 不把文本没有支持的关系说成显式关系。

### 8.3 历史查询

历史查询用于解释事情如何随时间变化。

示例意图：

```text
这个计划之前是怎么变化的？
```

预期行为：

- 召回相关记忆。
- 按 `occurred_at` 排列证据。
- 区分事件发生时间和写入时间。
- 不把较晚写入误当成较晚发生，除非 `occurred_at` 支持。

### 8.4 前因后果查询

前因后果查询用于回答 before/after、why/how、后续影响、触发因素等问题。

示例意图：

```text
为什么这次发布延期，后来又发生了什么？
```

预期行为：

- 生成面向原因和结果的 rewritten query。
- 在同一 `scope` 内召回候选。
- 按 `occurred_at` 排列证据。
- 谨慎表达因果关系。如果记忆只能支持先后顺序，不能证明因果，应明确说明。
- 每个因果判断都必须引用支撑记忆。

## 9. API 设计

### 9.1 基础约定

第一版 API 只保留 6 个顶层接口，并统一使用 `/v1` 版本前缀：

```text
/v1/ingest
/v1/recall
/v1/prompt-eval
/v1/delete
/v1/health
/v1/preview
/v1/audits
/v1/usage/stats
```

第一版 API 不内建认证，也不要求调用方提供 token、session 或 API key。MemoFlux 本身不解释用户、组织、项目或股票含义，只要求调用方在请求体或 query string 中传入精确 `scope`。如果部署场景需要访问控制，应由内网隔离、反向代理、API gateway 或上游业务系统负责；MemoFlux 内部仍只依赖精确 `scope` 过滤保证数据隔离。

统一响应格式：

```json
{
  "data": {},
  "error": null
}
```

错误响应格式：

```json
{
  "data": null,
  "error": {
    "code": "validation_error",
    "message": "occurred_at is required",
    "details": {
      "field": "occurred_at"
    },
    "query_id": null
  }
}
```

`query_id` 只在召回链路已经创建审计记录后返回。`/v1/delete` 返回 `delete_id`。写入、健康检查、预览和 usage stats 等接口通常不返回 `query_id`。

### 9.2 写入记忆：`POST /v1/ingest`

```http
POST /v1/ingest
```

请求：

```json
{
  "scope": "user:123:general",
  "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
  "occurred_at": "2026-05-01T10:00:00Z"
}
```

字段规则：

```text
scope: 必填，非空字符串，精确隔离边界。
content: 必填，非空字符串，原始记忆正文。
occurred_at: 必填，ISO 8601 时间，表示事件发生时间。
```

成功响应：

```json
{
  "data": {
    "memory_id": "018f7f3e-2d7d-7d63-b5d2-5d8b0f6a3b21",
    "scope": "user:123:general",
    "occurred_at": "2026-05-01T10:00:00Z",
    "created_at": "2026-05-30T08:30:00Z"
  },
  "error": null
}
```

语义：

- 写入成功后 `content` 和 `occurred_at` 不允许更新。
- 写入成功后即可被向量检索和文本/时间检索候选链路读取。

### 9.3 召回记忆：`POST /v1/recall`

```http
POST /v1/recall
```

请求：

```json
{
  "scope": "user:123:general",
  "query": "Atlas 发布为什么延期？",
  "include_references": true,
  "top_k": 12
}
```

字段规则：

```text
scope: 必填，只允许一个精确 scope。
query: 必填，自然语言查询。
include_references: 可选，默认 true；false 时响应不返回 references。
top_k: 可选，默认 12；限制每个 rewritten query 或合并后的候选规模，具体含义由实现固定并写入审计。
```

成功响应：

```json
{
  "data": {
    "query_id": "018f7f42-1a62-7b1a-87df-5ec04bbdc2d1",
    "query_type": "causal",
    "answer": "Atlas 发布延期的直接原因是数据库迁移回滚方案不完整，后续又要求补充回滚演练记录。",
    "references": [
      {
        "memory_id": "018f7f3e-2d7d-7d63-b5d2-5d8b0f6a3b21",
        "occurred_at": "2026-05-01T10:00:00Z",
        "quote": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
        "relevance": "支撑延期原因"
      }
    ],
    "confidence": 0.78,
    "uncertainties": []
  },
  "error": null
}
```

语义：

- `query_type` 由 LLM Query Planner 自动判断。
- 查询只在一个精确 `scope` 内执行。
- 默认返回引用证据。
- `include_references=false` 只影响响应体，不影响审计日志中保存的候选和引用信息。
- 如果候选证据不足，答案应明确说明无法从当前 `scope` 记忆中得出结论，而不是编造。

### 9.4 Prompt 调试：`POST /v1/prompt-eval`

```http
POST /v1/prompt-eval
```

请求：

```json
{
  "prompt_key": "query_planner",
  "payload": {
    "scope": "user:123:general",
    "query": "Atlas 发布为什么延期？"
  }
}
```

字段规则：

```text
prompt_key: 必填，指定要调试的内部 prompt，例如 query_planner 或 answer_synthesizer。
payload: 必填，传给该 prompt 的结构化输入。
```

成功响应：

```json
{
  "data": {
    "status": "ok",
    "prompt_key": "query_planner",
    "model": "memory/default",
    "latency_ms": 860,
    "output": {},
    "error_code": null,
    "error_message": null
  },
  "error": null
}
```

语义：

- 该接口用于开发、评估和排查 prompt，不参与正常记忆写入或召回链路。
- MemoFlux 第一版不内建认证；生产部署如果不希望暴露 prompt 调试能力，应在网络层或反向代理限制该接口。
- `payload` 不应包含密钥、Cookie、JWT 或外部供应商原始敏感 payload。
- 该接口可能调用 LLM，但响应不返回单次 token 消耗；用量只进入内部统计和 `/v1/usage/stats` 聚合结果。

### 9.5 删除记忆：`POST /v1/delete`

```http
POST /v1/delete
```

请求：

```json
{
  "scope": "user:123:general",
  "memory_ids": ["018f7f3e-2d7d-7d63-b5d2-5d8b0f6a3b21"],
  "dry_run": false
}
```

字段规则：

```text
scope: 必填，精确 scope。
memory_ids: 可选，明确要删除的 memory_id 列表。
query: 可选，自然语言描述，用于 dry-run 查找候选记忆。
dry_run: 可选，默认 true；query 模式必须先 dry_run。
```

请求可以使用两种目标选择方式：

```json
{
  "scope": "user:123:general",
  "memory_ids": ["..."],
  "dry_run": false
}
```

```json
{
  "scope": "user:123:general",
  "query": "删除 Atlas 发布延期的数据库迁移回滚方案细节",
  "dry_run": true
}
```

成功响应：

```json
{
  "data": {
    "delete_id": "018f809f-7dd8-7c90-8e6f-1f64a9c7c0fa",
    "scope": "user:123:general",
    "dry_run": false,
    "matched_memory_ids": ["018f7f3e-2d7d-7d63-b5d2-5d8b0f6a3b21"],
    "affected_memory_ids": ["018f7f3e-2d7d-7d63-b5d2-5d8b0f6a3b21"],
    "status": "ok"
  },
  "error": null
}
```

语义：

- 删除就是遗忘，不可恢复。
- 使用 `memory_ids` 时是确定性操作，不需要 LLM。
- 使用 `query` 时只能用于 `dry_run=true` 查找候选；不能直接执行删除。
- 通过 `query` 查找候选可能调用 LLM、向量召回和文本/时间检索，但响应不返回单次 token 消耗。
- 真正执行删除必须由调用方传入明确 `memory_ids`。
- 删除成功后，原文和相关审计中的原文片段都应被移除或清理。

### 9.6 健康检查：`GET /v1/health`

```http
GET /v1/health
```

响应应包含数据库、检索层和 LLM 状态：

```json
{
  "data": {
    "status": "ok",
    "db": "ok",
    "retrieval": "ok",
    "llm": "configured",
    "retrieval_strategy": "text_time"
  },
  "error": null
}
```

健康检查不需要认证，但不能泄露密钥、完整连接串、用户内容、scope 列表或记忆数量明细。

### 9.7 预览记忆：`GET /v1/preview`

```http
GET /v1/preview?scope=user:123:general&from=2026-05-01T00:00:00Z&to=2026-05-31T23:59:59Z&limit=50&cursor=...
```

查询参数：

```text
scope: 必填，精确 scope。
from: 可选，occurred_at 起始时间。
to: 可选，occurred_at 结束时间。
limit: 可选，默认 50，最大值由服务配置控制。
cursor: 可选，用于分页。
order: 可选，occurred_at_asc 或 occurred_at_desc，默认 occurred_at_asc。
```

成功响应：

```json
{
  "data": {
    "items": [
      {
        "memory_id": "...",
        "scope": "user:123:general",
        "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
        "occurred_at": "2026-05-01T10:00:00Z",
        "created_at": "2026-05-30T08:30:00Z"
      }
    ],
    "next_cursor": null
  },
  "error": null
}
```

语义：

- 该接口只用于预览原始记忆，不做 LLM 召回和答案整合。
- 必须按精确 `scope` 查询。
- 默认按 `occurred_at` 排序。
- 已删除记忆不会出现在 preview 中。

### 9.8 查询审计：`GET /v1/audits`

```http
GET /v1/audits?scope=user:123:general&query_id=018f7f42-1a62-7b1a-87df-5ec04bbdc2d1&limit=50&cursor=...
```

查询参数：

```text
scope: 必填，精确 scope。
query_id: 可选，传入时返回单条审计详情。
limit: 可选，默认 50。
cursor: 可选，用于分页。
status: 可选，按召回状态过滤。
error_code: 可选，按错误码过滤。
```

```json
{
  "data": {
    "items": [
      {
        "query_id": "...",
        "scope": "user:123:general",
        "original_query": "Atlas 发布为什么延期？",
        "query_type": "causal",
        "query_type_reason": "用户询问原因，属于前因后果查询。",
        "rewritten_queries": [
          {
            "query": "Atlas 发布延期 原因 数据库迁移 回滚方案",
            "purpose": "cause"
          }
        ],
        "candidate_limit": 12,
        "candidate_filters": {
          "scope": "user:123:general"
        },
        "retrieved": [
          {
            "memory_id": "...",
            "occurred_at": "2026-05-01T10:00:00Z",
            "score": 0.82,
            "source_query_index": 0,
            "content_preview": "Atlas 发布延期，因为数据库迁移回滚方案不完整。"
          }
        ],
        "selected_memory_ids": ["..."],
        "final_answer": "...",
        "include_references": true,
        "timings_ms": {},
        "error_stage": null,
        "error_message": null,
        "created_at": "2026-05-30T09:10:00Z"
      }
    ],
    "next_cursor": null
  },
  "error": null
}
```

审计接口不内建认证，调用方必须传入精确 `scope`。MemoFlux 只返回该 `scope` 下的审计记录；如果部署方不希望任意调用方读取审计信息，应在反向代理、API gateway 或网络层限制 `/v1/audits` 访问。审计响应不返回单次 LLM token 消耗。

### 9.9 查询聚合用量：`GET /v1/usage/stats`

```http
GET /v1/usage/stats?hours=24
```

查询参数：

```text
hours: 可选，统计最近 N 小时；不传时返回当前保留窗口内的全部统计。
```

成功响应：

```json
{
  "data": {
    "status": "ok",
    "llm_runs": 128,
    "total_calls": 128,
    "input_tokens": 120000,
    "output_tokens": 36000,
    "total_tokens": 156000,
    "cached_tokens": 8000,
    "cache_miss_tokens": 112000,
    "reasoning_tokens": 4000,
    "cache_hit_rate": 0.06,
    "by_operation": {
      "query_planner": {
        "calls": 64,
        "input_tokens": 32000,
        "output_tokens": 9000
      },
      "answer_synthesizer": {
        "calls": 64,
        "input_tokens": 88000,
        "output_tokens": 27000
      }
    }
  },
  "error": null
}
```

语义：

- 该接口只返回聚合用量，不返回任何单次调用的 token 消耗明细。
- 聚合维度可包含 operation、model、provider 等，但不应包含原始 prompt、query、memory content 或 scope 列表。
- `/v1/recall`、`/v1/prompt-eval` 和 `/v1/delete` 的 query dry-run 都可能产生 LLM 用量，这些用量进入聚合统计。

### 9.10 清空聚合用量：`DELETE /v1/usage/stats`

```http
DELETE /v1/usage/stats
```

成功响应：

```json
{
  "data": {
    "status": "ok",
    "deleted": 128
  },
  "error": null
}
```

语义：

- 清空 usage 统计记录，不删除 memory、query audit 或 delete audit。
- MemoFlux 第一版不内建认证；生产部署如果不希望暴露清空统计能力，应在网络层或反向代理限制该接口。

## 10. 错误处理

校验错误：

- 缺少 `scope`。
- 缺少 `content`。
- 缺少 `occurred_at`。
- 时间格式非法。
- memory 不存在于精确 `scope`。
- `/v1/delete` 使用 `query` 且 `dry_run=false`。
- `/v1/delete` 同时缺少 `memory_ids` 和 `query`。

LLM planner 错误：

- 结构化输出不合法。
- 超时。
- provider 失败。

候选检索错误：

- 数据库查询失败。
- 文本检索配置不可用。
- 候选为空。

答案整合错误：

- 结构化输出不合法。
- 超时。
- provider 失败。

所有查询失败都应尽量写入审计记录，包含 `error_stage`、`error_message` 和已经采集到的耗时字段。

## 11. 安全与隔离

- 所有查询必须包含且只能包含一个 `scope`。
- 服务不能扩展、推断或继承 `scope`。
- 每次候选检索必须包含精确 `scope`。
- 从 `memories` 回填候选时必须再次校验精确 `scope`，不能只信任候选来源。
- 删除必须按精确 `scope` 校验目标 memory，不能删除其他 `scope` 的记忆。
- 删除后不得在业务表、preview 或 audit 响应中继续暴露原始正文。
- 审计查询必须要求精确 `scope`，且不能返回其他 `scope` 的审计记录。
- MemoFlux 第一版不内建认证；生产部署如需访问控制，应由网络层、反向代理、API gateway 或上游业务系统负责。
- 默认日志不记录完整记忆正文；只有本地调试显式开启时才允许。

## 12. 可观测性

每次查询应记录以下耗时：

```text
planner_llm_ms
candidate_search_ms
candidate_hydration_ms
answer_llm_ms
total_ms
```

用量统计应聚合记录以下信息，不能在单次业务响应中返回：

```text
planner model 与 token usage
answer model 与 token usage
```

每条审计记录应能回答：

- Planner 是否正确判断查询类型？
- 生成了哪些 rewritten query？
- 候选检索是否严格限制在一个 `scope` 内？
- 哪些 memories 被召回，分数是多少？
- 最终答案使用了哪些召回记忆？
- 答案是否引用了证据？
- 哪个阶段失败或耗时最高？

usage stats 应能回答：

- 当前统计窗口内调用了多少次 LLM。
- 总 input/output/reasoning/cached token 数是多少。
- 不同 operation 的聚合用量是多少。

## 13. 测试策略

确定性测试：

- 缺少 `occurred_at` 时写入失败。
- 缺少 `scope` 时查询失败。
- 查询不会搜索多个 `scope`。
- candidate hydration 会排除其他 `scope` 的 memory。
- 删除 memory 后，业务表原文和审计响应中不再暴露原始正文。
- 原始 memory content 不能更新。
- `/v1/delete` 的 query 模式只能 dry-run，不能直接执行删除。
- `/v1/usage/stats` 只返回聚合用量，不返回单次调用明细。

Mock LLM 测试：

- Planner 能输出 direct、related、history、causal 四类查询计划。
- Planner 输出非法时返回明确失败并写入审计。
- Synthesizer 能基于候选记忆输出带引用答案。
- 当候选不支持问题时，Synthesizer 应拒绝臆测。

检索测试：

- 不同 `scope` 中相同文本不会互相泄漏。
- 关联查询可以使用多个 rewritten query。
- 历史查询按 `occurred_at` 组织证据。
- 前因后果查询在证据不足时能区分“先后顺序”和“因果关系”。

运维测试：

- 健康检查能报告数据库、检索层和 LLM 状态。
- 查询审计记录 planner、candidate retrieval、answer、usage 和 timing 详情。
- `DELETE /v1/usage/stats` 只清空 usage 统计，不删除 memory 或 audit。

## 14. 相似记忆评估发现

使用 100 条同一 `scope` 内的相似发布/延期记忆进行 smoke 评估后，当前召回链路暴露出以下问题：

- pgvector `top_k` 在同一 `scope` 内混放多个项目时，会召回语义相似但主题不同的候选。例如查询 Atlas 时，候选中可能混入 Zephyr、Orion 或 Nova 的延期记录。
- 最终答案通常比候选更干净，因为 Answer Synthesizer 能过滤部分无关候选；但 API `references` 当前如果直接返回候选集合，会把无关候选暴露给调用方。
- `references` 不应等同于 retrieved candidates。`retrieved` 是排障用候选集合，应保存在 audit；API `references` 应只包含最终答案实际使用的记忆。
- Query Planner 生成的 `rewritten_queries` 当前主要用于审计。后续如果要提升召回，应对 rewritten queries 做多路 embedding 检索、合并和去重。
- LLM 结构化输出必须包含 `used_memory_ids`，表示最终答案实际引用的候选记忆 ID。服务层必须过滤掉不在候选集合中的 ID。
- LLM 结构化输出中的 `confidence` 必须是 `0.0` 到 `1.0` 的数字。字符串值如 `"high"` 不合规，服务层应降级为默认置信度并通过 audit 暴露输出不合规事实。

因此，`POST /v1/recall` 的响应语义调整为：

- `references`：只返回 Answer Synthesizer 声明使用的 `used_memory_ids` 对应记忆。
- `audits[].retrieved`：保留完整候选集合，用于诊断召回噪声。
- `audits[].selected_memory_ids`：记录 `used_memory_ids` 过滤后的有效引用 ID。

## 15. 待定决策

- 是否在 pgvector 之外引入更复杂的 hybrid retrieval 编排。第一版先保持 SQLAlchemy + pgvector + 文本/时间兜底。
- 是否引入 LlamaIndex。第一版不引入；只有需要多向量后端或复杂 retriever 编排时再评估。
- 同一 `scope` 内重复写入完全相同 `content` 时，是否允许重复。第一版可以允许重复，避免为了去重增加非必要 hash 字段。
- 审计日志保留期。默认 30 天合理，但最终应由部署策略决定。
- `include_references=false` 是否只影响 API 响应，还是同时影响审计持久化。当前建议只影响响应；审计仍保留详细引用和候选信息。
