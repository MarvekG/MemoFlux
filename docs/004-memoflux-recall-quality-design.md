# MemoFlux Recall Quality Pipeline 设计

## 目标

提升 MemoFlux `/v1/recall` 在同一 `session` 内存在多条相似记忆时的可靠性。召回链路需要支持多路 rewritten query 检索、候选去重、Answer Synthesizer 内置候选判别、证据不足拒答，以及可解释的 references relevance。

## 设计原则

- 不新增独立 reranker LLM 调用。
- 不使用关键词、正则、白名单、黑名单或主体硬编码规则判断相关性。
- 语义判断集中在 Answer Synthesizer 的结构化输出中。
- `session` 仍是唯一隔离边界，不支持跨 session 查询。
- API `references` 只返回最终答案实际使用的记忆。
- audit 保留完整候选和最终选择原因，便于排障。

## Recall 流程

新流程：

```text
POST /v1/recall
  -> 校验 session/query/top_k
  -> Query Planner 输出 query_type 和 rewritten_queries
  -> 构建检索 query 列表：原始 query + rewritten_queries，去重保序
  -> 对每个检索 query 生成 embedding
  -> 每路在同一 session 内 pgvector 检索
  -> 合并候选，按 memory_id 去重，保留首次出现顺序
  -> 截断为 top_k 个候选
  -> Answer Synthesizer 同时完成候选判别和答案生成
  -> used_memory_ids 过滤 references
  -> references[].relevance 使用 relevance_by_id
  -> 写 query audit
```

`top_k` 表示最终合并去重后的候选上限，避免 rewritten query 数量增加导致候选无限膨胀。

## Query Planner

Planner 继续输出：

```json
{
  "query_type": "single",
  "rewritten_queries": ["Atlas 延期原因", "Atlas 项目延期的根因"]
}
```

服务层将原始 query 放在第一位，再追加 rewritten queries。空字符串、重复字符串会被过滤。Planner 输出缺失或为空时，只使用原始 query。

## Answer Synthesizer 输出契约

Answer Synthesizer 输出扩展为：

```json
{
  "answer": "数据库迁移回滚方案不完整",
  "confidence": 0.9,
  "used_memory_ids": ["memory_id_1"],
  "relevance_by_id": {
    "memory_id_1": "该记忆直接说明 Atlas 延期原因是数据库迁移回滚方案不完整"
  },
  "uncertainties": []
}
```

字段规则：

- `confidence` 必须是 `0.0` 到 `1.0` 的数字。
- `used_memory_ids` 必须来自候选 memory id。
- `relevance_by_id` 只允许解释候选 memory id。
- `uncertainties` 用于表达证据不足、候选不匹配或输出降级原因。

如果候选没有足够证据回答问题，Answer Synthesizer 必须返回：

```json
{
  "answer": "当前 session 中没有足够记忆支持回答该问题。",
  "confidence": 0.0,
  "used_memory_ids": [],
  "relevance_by_id": {},
  "uncertainties": ["候选记忆未包含问题所需证据"]
}
```

## References

`references` 只返回 `used_memory_ids` 对应的记忆。

`references[].relevance` 使用 `relevance_by_id[memory_id]`。如果 LLM 声明使用某条记忆但没有提供解释，服务层使用保守 fallback：

```text
Answer Synthesizer 声明使用该记忆，但未提供引用理由。
```

如果 `used_memory_ids` 为空，`references` 返回空数组，不再退回全部候选。这样可以避免证据不足时把噪声候选暴露给调用方。

## Audit

Query audit 保留：

- `retrieved`：完整合并候选集合。
- `selected_memory_ids`：最终使用的记忆 ID。
- `selection_reasons`：`memory_id -> relevance reason`。
- `final_answer`：最终答案或拒答文案。

如果删除记忆，审计 scrub 需要继续清理 `retrieved[].content_preview`，并在被删除记忆属于 `selected_memory_ids` 时清空 `final_answer`。`selection_reasons` 中对应 ID 的解释也应清理，避免残留被删除记忆的语义内容。

## 错误与降级

- Answer Synthesizer 输出非法 JSON 时，服务层使用候选正文 fallback，但 `confidence` 降为 `0.6`，`uncertainties` 包含 `invalid_llm_output`。
- `confidence` 非数字时降级为 `0.6`。
- `used_memory_ids` 包含非法 ID 时过滤非法 ID。
- `relevance_by_id` 包含非法 ID 时过滤非法 ID。
- 如果过滤后 `used_memory_ids` 为空，返回空 references，不暴露全部候选。

## 测试策略

单元/API 测试覆盖：

- 多个 rewritten queries 会触发多次 embedding 和 repository search。
- 多路检索结果按 memory_id 去重。
- `top_k` 应用于合并去重后的候选。
- `used_memory_ids` 为空时 references 为空。
- 非法 `used_memory_ids` 被过滤，不退回全部候选。
- `references[].relevance` 来自 `relevance_by_id`。
- 缺失 relevance reason 时使用 fallback 文案。
- audit 返回 `selection_reasons`。
- 删除 scrub 会清理被删除 memory 的 `selection_reasons`。
- 证据不足时返回固定拒答文案和低置信度。

Live smoke 覆盖：

- 同 session 内 Atlas 和 Zephyr 相似记忆并存时，问 Atlas 不应引用 Zephyr。
- 问一个候选未支持的问题时返回证据不足拒答。

## 非目标

- 不新增独立 reranker。
- 不新增 embedding 距离阈值作为硬过滤。
- 不实现跨 session 查询。
- 不引入关键词主体判断。
- 不改变 `/v1/recall` 的 URL 和基本响应结构。
