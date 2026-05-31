# MemoFlux 无强元数据召回优化设计

## 背景

MemoFlux 当前已经完成会话级隔离、pgvector 向量召回、多路 rewritten query 检索、Answer Synthesizer 严格引用过滤和 audit 持久化。最新 1000 条单 session live eval 显示，修复 fail-closed fallback 后，跨主体噪声已从 `3/100` 降到 `0/100`，无证据拒答从 `9/10` 提升到 `10/10`，但覆盖/纠错和历史查询仍有明显漏召回。

这类问题不适合通过新增强业务元数据字段解决。`subject`、`fact_type`、`temporal_kind`、`current_version` 等字段看似能提升过滤精度，但会把开放会话记忆系统改造成半结构化知识库。字段提取本身很难，一旦提取错误，检索会被错误元数据锁死；领域变化时还会不断扩 schema，增加维护成本。

本设计选择不新增强业务元数据字段，改为优化 query-time recall：宽召回、在 Answer Synthesizer 内完成证据选择、严格引用和可审计失败。

## 目标

- 不新增需要语义提取的业务元数据字段。
- 保留原始记忆为事实来源，不修改原文，不通过更新旧记忆表达纠错。
- 提升单 session 大规模记忆下的覆盖/纠错、关联和历史查询质量。
- 保持 `references` 只来自最终实际使用的 `used_memory_ids`。
- 保持 no-evidence 场景 fail-closed，不因召回候选存在而强行作答。
- 通过 audit 区分召回失败和答案综合失败。

## 非目标

- 不新增 `subject`、`fact_type`、`temporal_kind`、`current_version`、`entity` 等持久化业务字段。
- 不把 MemoFlux 改造成知识图谱、实体库或事实表。
- 不使用关键词、正则、白名单、黑名单或领域定制规则来判断主体、事实类型、纠错关系或相关性。
- 不在 ingest 阶段要求 LLM 提取通用事实 schema。
- 不改变 session 隔离边界，不支持跨 session 查询。

## 核心原则

### 原始记忆优先

MemoFlux 只把用户写入的原始记忆作为事实来源。纠错通过追加新记忆表达，系统不覆盖旧内容，也不把某条记忆标记为“当前值”。是否采用较新的纠错内容，应由 query-time LLM 在候选证据中判断。

### 不持久化强语义判断

主体识别、事实类型判断、历史/当前判断和纠错判断都属于高不确定语义判断。它们不应被写成持久字段并参与硬过滤。系统应把这些判断放在每次 recall 的 LLM 输入、结构化输出和 audit 中。

### 宽召回，严引用

召回阶段允许拿到较宽候选，甚至包含噪声；最终 API `references` 必须严格来自 LLM 声明实际使用的合法 `used_memory_ids`。LLM 输出不可解析或 schema validation 失败时，必须固定拒答且 `references=[]`。

### Answer Synthesizer 内选择证据

系统不引入额外 LLM 筛选阶段。Answer Synthesizer 同时负责候选判别、答案生成和引用声明。这样保持 recall 为三阶段，避免增加一次 LLM 调用、延迟和失败面。

## 方案概览

召回流程保持三阶段：

1. Query Planner 生成自然语言检索改写。
2. Candidate Collector 从多个候选池宽召回。
3. Answer Synthesizer 在候选内完成证据选择、答案生成和严格 references。

数据仍只依赖：

- `session`
- `content`
- `occurred_at`
- `embedding`
- query/delete/usage audit

## Query Planner

Query Planner 不输出强业务 metadata，只输出查询类型和自然语言检索改写。

示例输出：

```json
{
  "query_type": "history",
  "rewritten_queries": [
    "按历史记录总结 Atlas 发布暂停原因",
    "Atlas 发布暂停 历史 原因",
    "Atlas 项目过去为什么暂停发布",
    "Atlas 项目发布变化和暂停原因"
  ]
}
```

允许的 `query_type` 是弱控制信号，只用于候选预算和排序策略，不用于业务事实过滤：

- `direct`：普通事实查询。
- `history`：历史、时间线、变化过程查询。
- `comparison`：比较、差异、多个对象对照查询。
- `summary`：概括、归纳、复盘查询。

Planner 失败时退回原始 query 和 `direct`，这是安全 fallback。

## Candidate Collector

Candidate Collector 负责构建较宽候选池，不做最终相关性判断。

候选来源包括：

- `semantic_pool`：原始 query embedding 召回。
- `rewrite_pool`：每条 rewritten query embedding 召回。
- `recent_pool`：同 session 最近 N 条记忆。
- `history_pool`：当 `query_type` 为 `history` 或 `summary` 时，扩大候选池并按 `occurred_at` 保留时间跨度。
- `diversity_pool`：从多路召回结果中按 memory_id 去重，避免单一路径占满候选。

候选合并规则：

- 只在同一 `session` 内召回。
- 按 `memory_id` 去重。
- 保留候选来源信息到 audit，例如 `sources=["semantic_pool", "rewrite_pool"]`。
- 不根据关键词或主体字符串过滤。
- 不提前把候选截断到 API `top_k`；`top_k` 是引用规模提示，不是证据池上限。

建议默认候选预算：

| query_type | 初始候选预算 | 说明 |
|---|---:|---|
| direct | 36-60 | 覆盖普通事实和纠错 |
| history | 80-120 | 覆盖时间线和历史摘要 |
| comparison | 60-100 | 覆盖多对象候选 |
| summary | 80-120 | 覆盖复盘和归纳 |

## Answer Synthesizer

Answer Synthesizer 接收 Candidate Collector 的宽候选池。它负责在候选中判断哪些记忆真正支持答案，并生成最终答案、置信度、实际引用和不确定性。

输入示例：

```json
{
  "query": "Atlas 当前延期原因是什么？",
  "query_type": "direct",
  "memories": [
    {
      "memory_id": "m1",
      "content": "Atlas 项目延期，原因是数据库迁移回滚方案不完整。",
      "occurred_at": "2026-05-01T10:00:00Z"
    },
    {
      "memory_id": "m2",
      "content": "Atlas 项目延期原因纠正为权限模型变更未完成安全评审，之前记录的数据库迁移回滚方案不再作为当前判断依据。",
      "occurred_at": "2026-05-08T10:00:00Z"
    }
  ]
}
```

输出结构保持：

```json
{
  "answer": "权限模型变更未完成安全评审。",
  "confidence": 0.95,
  "used_memory_ids": ["m1", "m2"],
  "relevance_by_id": {
    "m1": "该记忆提供旧延期原因，作为被纠正背景。",
    "m2": "该记忆明确说明 Atlas 当前延期原因已纠正为权限模型变更未完成安全评审。"
  },
  "uncertainties": []
}
```

候选不足时固定输出：

```json
{
  "answer": "当前 session 中没有足够记忆支持回答该问题。",
  "confidence": 0.1,
  "used_memory_ids": [],
  "relevance_by_id": {},
  "uncertainties": ["候选记忆不足以回答问题"]
}
```

约束：

- `used_memory_ids` 必须来自候选。
- 候选不足以回答时返回空数组。
- `relevance_by_id` 只解释实际引用的候选记忆。
- 不返回未在候选中的 memory_id。
- 输出不可解析时 fail-closed：固定拒答、`used_memory_ids=[]`、`references=[]`。
- API `references` 继续只返回合法 `used_memory_ids` 对应的原始记忆。

## Audit 设计

为定位失败原因，query audit 应记录分阶段信息。

新增或扩展 audit payload，而不是 memory 表：

```json
{
  "retrieved": [
    {
      "memory_id": "m1",
      "content_preview": "...",
      "occurred_at": "...",
      "candidate_sources": ["semantic_pool", "rewrite_pool"]
    }
  ],
  "selected_memory_ids": ["m2"],
  "selection_reasons": {
    "m2": "最终答案实际引用该记忆。"
  }
}
```

这样 eval 可以区分：

- `retrieval_hit=false`：召回没拿到证据。
- `retrieval_hit=true` 且 `selected_memory_ids` 未包含期望证据：Answer Synthesizer 候选判别失败。
- `selected_memory_ids` 包含期望证据但 `answer_correct=false`：Answer Synthesizer 答案综合失败。
- `selected_memory_ids` 含噪声：严格引用失败。

## Eval 设计

1000 条单 session eval 应从只看最终答案，扩展为分阶段指标。

核心指标：

- `retrieval_hit`：期望记忆是否进入 audit `retrieved`。
- `selection_hit`：期望记忆是否进入最终 `selected_memory_ids`。
- `answer_correct`：最终答案是否包含期望事实。
- `ref_noise`：references 是否包含非目标证据。
- `answer_noise`：答案是否引入非目标主体或无证据事实。
- `no_evidence_correct`：无证据查询是否固定拒答且 references 为空。

关联查询需要拆分口径：

- `current_association`：问当前依赖，需要候选中有明确纠错或最新上下文。
- `association_history`：问历史依赖，允许返回多个同项目依赖记录。

如果 eval 只把“最后一次写入”作为唯一正确答案，而用户问题没有表达“当前”，则评估会误判历史完整回答为错误。

## 错误处理

- Query Planner 输出不可解析：退回原始 query 和 `direct`。
- Answer Synthesizer 输出不可解析：固定拒答，`references=[]`。
- LLM 请求失败：记录 audit error stage，返回固定拒答或服务错误；第一版优先固定拒答并记录 `uncertainties`。
- 候选为空：不调用 Answer Synthesizer，直接固定拒答。

## 预期效果

在保持 `answer_noise=0`、`ref_noise=0` 的前提下，目标是提升：

- `latest_delay`：从 `21/30` 提升到 `26/30` 以上。
- `history`：从 `12/20` 提升到 `16/20` 以上。
- `no_evidence`：维持 `10/10`。
- `owner`：维持 `15/15`。

关联查询的目标需要在 eval 口径拆分后重新定义。如果问题是“依赖哪个服务”，默认可回答历史上所有明确相关依赖；如果问题是“当前依赖哪个服务”，则 Answer Synthesizer 必须基于候选中的纠错/时间上下文判断当前值。

## 实施顺序

1. 扩展 eval 输出，增加 `retrieval_hit`、`selection_hit`、`answer_correct` 分阶段指标。
2. 扩展 Query Planner prompt，只生成更丰富的自然语言 rewritten queries，不新增强业务字段。
3. 实现 Candidate Collector 多池召回和候选来源 audit。
4. 强化 Answer Synthesizer prompt 和 schema，使其在宽候选中完成证据选择、答案生成和引用理由输出。
5. 跑 1000 条 single-session eval，对比修复前后的分阶段指标。

## 风险与取舍

- 候选池扩大后 prompt token 增加，需要设置预算上限。
- 不使用强 metadata 意味着某些查询不会像结构化数据库一样稳定，但更适合开放会话记忆。
- Answer Synthesizer 同时负责候选判别和答案综合，单次 prompt 压力更大；audit 必须足够完整，便于复盘。

## 结论

MemoFlux 不应通过新增强业务元数据字段来解决大规模 session 召回问题。更稳妥的方向是保持存储简单，把语义判断留在 query-time：Query Planner 生成自然语言检索改写，Candidate Collector 宽召回获取足够候选，Answer Synthesizer 在候选内完成证据选择、答案生成、严格引用并 fail-closed。这样既能保持三阶段流程和调用成本，又避免把不可靠的语义抽取结果写成持久事实。
