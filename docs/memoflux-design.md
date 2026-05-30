# MemoFlux Design

## 1. Positioning

MemoFlux is a long-term memory service that stores original memory text, isolates all data by caller-provided scope, and answers natural-language questions through an LLM-driven retrieval pipeline.

The service is designed as an independent, long-lived system rather than a project-specific helper. Its first version intentionally keeps the business data model small and makes semantic behavior observable through detailed query audit logs.

## 2. Goals

- Store original memory content as text without chunking, entity extraction, relation edges, causal edges, or generated summaries.
- Enforce strict scope isolation. Memories in different scopes must never affect each other.
- Support direct query, related query, history query, and cause/effect query through the same three-stage retrieval pipeline.
- Require event time for every memory so historical and causal answers can be ordered by when things happened, not only by when they were written.
- Keep original memory content immutable. Corrections are represented by appending new memories.
- Use a vector engine modeled after the existing `memory` directory approach: LlamaIndex retrieval orchestration with PostgreSQL/pgvector storage.
- Record enough audit detail to debug query rewriting, vector retrieval, LLM answer synthesis, latency, and failures.

## 3. Non-Goals

- Do not build an entity-centered memory graph in the first version.
- Do not persist extracted entities, events, relation edges, causal edges, or memory chunks.
- Do not support cross-scope querying, scope inheritance, or automatic scope expansion.
- Do not update original memory text after creation.
- Do not make the vector index a source of truth.
- Do not encode semantic query classification, causality, or relevance rules in deterministic keyword logic.

## 4. Core Concepts

### Scope

`scope` is an opaque string supplied by the caller. MemoFlux validates that it is present and uses exact matching for all writes, reads, deletes, vector searches, and audits.

Examples:

```text
user:123:general
user:123:stock:600000.SH
org:alpha:project:atlas
```

MemoFlux does not interpret scope hierarchy. A query for `user:123` cannot see `user:123:stock:600000.SH` unless the caller explicitly wrote those memories into the exact same scope.

### Memory

A memory is one immutable original text record with an event time.

```text
memory_id
scope
content
occurred_at
created_at
deleted_at
```

`occurred_at` is required. If the caller cannot provide it, the write is rejected.

### Vector Index

The vector index is derived data. It can be rebuilt from `memories` and must not be treated as business truth.

MemoFlux should follow the same technical direction as the existing memory retrieval design:

```text
LlamaIndex RetrievalIndex
  -> PGVectorStore
  -> PostgreSQL + pgvector
  -> fixed vector index table, for example memory_node_index
```

Each vector node represents one full memory record. The node text is the original `content`. Node metadata should stay minimal but must include the fields needed for filtering and hydration:

```json
{
  "memory_id": "...",
  "scope": "...",
  "occurred_at": "...",
  "deleted": false
}
```

Embedding provider, model, dimension, and projection version belong to service configuration and index status, not to each business memory row.

## 5. Data Model

### `memories`

```text
memory_id: UUID primary key
scope: text not null
content: text not null
occurred_at: timestamptz not null
created_at: timestamptz not null
deleted_at: timestamptz null
```

Indexes:

```text
(scope, occurred_at)
(scope, created_at)
(scope, deleted_at)
```

Rules:

- `content` cannot be updated.
- `occurred_at` cannot be updated.
- Soft delete sets `deleted_at`.
- Restore clears `deleted_at`.
- Normal queries exclude deleted memories.
- Corrections are represented as new memory rows.

### Vector Index Table

MemoFlux should let LlamaIndex `PGVectorStore` manage the physical vector table. The application should treat it as a rebuildable retrieval index.

Required behavior:

- Every indexed node must be filterable by exact `scope`.
- Deleted memories must be removed from the index or excluded by metadata filter.
- A reindex operation must rebuild vector nodes from active `memories`.
- Changing embedding model, embedding dimension, or projection version requires index rebuild.

### `memory_query_audits`

Query audits should be detailed enough to diagnose the three-stage pipeline.

```text
query_id: UUID primary key
scope: text not null
original_query: text not null
query_type: text not null
query_type_reason: text null
rewritten_queries: jsonb not null
vector_top_k: integer not null
vector_filters: jsonb not null
retrieved: jsonb not null
selected_memory_ids: uuid[] not null
final_answer: text null
include_references: boolean not null
llm_usage: jsonb null
embedding_usage: jsonb null
timings_ms: jsonb not null
error_stage: text null
error_message: text null
created_at: timestamptz not null
```

`retrieved` should contain candidate diagnostics, not full memory content:

```json
[
  {
    "memory_id": "...",
    "occurred_at": "2026-05-01T10:00:00Z",
    "score": 0.82,
    "source_query_index": 0,
    "content_preview": "first 300 characters..."
  }
]
```

Audit logs are not a memory source of truth. Full evidence is always read from `memories` by `memory_id`.

## 6. Write Flow

```text
POST /memories
  -> validate scope/content/occurred_at
  -> insert immutable row into memories
  -> index one vector node with text=content and metadata={memory_id, scope, occurred_at, deleted=false}
  -> return memory_id
```

Validation rules:

- `scope` is required.
- `content` is required.
- `occurred_at` is required.
- Missing `occurred_at` returns a validation error and does not write anything.
- The service does not use an LLM to infer missing `occurred_at`.

If indexing fails after the database write, the memory remains the source of truth and the index becomes stale. The service should surface index health and provide a reindex command instead of hiding the failure.

## 7. Query Flow

MemoFlux uses a fixed three-stage query architecture:

```text
User query + exact scope
  -> LLM Query Planner
  -> Vector Retriever
  -> LLM Answer Synthesizer
```

### Stage 1: LLM Query Planner

The planner receives the original query and returns structured output:

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

The planner decides the query type automatically. The caller does not pass `query_type` in the first version.

The planner may produce multiple rewritten queries for one user query. This improves recall for related, history, and causal questions while keeping storage simple.

### Stage 2: Vector Retriever

For each rewritten query, MemoFlux runs vector search over the same exact `scope`.

Required filters:

```text
scope == request.scope
```

The retriever merges candidates from all rewritten queries, removes duplicate `memory_id`s, hydrates full memory rows from `memories`, and passes the candidate set to answer synthesis.

Sorting rules:

- Direct and related queries primarily use vector score.
- History queries should include enough candidates and present them to the answer synthesizer ordered by `occurred_at`.
- Causal queries should include candidates before and after the likely focal event when those candidates are retrieved, then present them ordered by `occurred_at`.

The retriever must not search other scopes to find background context.

### Stage 3: LLM Answer Synthesizer

The synthesizer receives:

```text
original query
query planner output
hydrated memory candidates
strict instruction to answer only from supplied memories
reference output requirement
```

It returns structured output:

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

References are returned by default. The caller may set `include_references=false` to omit references from the response body, but MemoFlux still records references in the audit record for troubleshooting.

## 8. Query Types

### Direct Query

Answers a specific question from directly matching memories.

Example intent:

```text
What is the current blocker for this project?
```

Expected behavior:

- Rewrite the query into retrieval-friendly wording.
- Retrieve the strongest same-scope candidates.
- Answer only from retrieved memories.
- Cite the memories used.

### Related Query

Finds memories that are semantically related even if they do not answer the query literally.

Example intent:

```text
What prior issues are related to this deployment risk?
```

Expected behavior:

- Generate multiple related rewritten queries.
- Retrieve semantically similar memories in the same scope.
- Explain why each memory is related.
- Avoid claiming explicit relationships that are not supported by the text.

### History Query

Explains what happened over time.

Example intent:

```text
How did the plan change over time?
```

Expected behavior:

- Retrieve relevant memories.
- Order evidence by `occurred_at`.
- Distinguish event time from write time.
- Avoid treating newer writes as newer events unless `occurred_at` says so.

### Causal Query

Answers before/after, why/how, consequence, and trigger questions.

Example intent:

```text
Why did the rollout get delayed, and what happened after that?
```

Expected behavior:

- Generate cause-oriented and effect-oriented rewritten queries.
- Retrieve candidates in the same scope.
- Present evidence ordered by `occurred_at`.
- Express causality cautiously. If memories show sequence but not causation, say so.
- Cite the memories that support each causal claim.

## 9. API Shape

### Write Memory

```http
POST /memories
```

Request:

```json
{
  "scope": "user:123:general",
  "content": "The Atlas rollout was delayed because the migration rollback plan was incomplete.",
  "occurred_at": "2026-05-01T10:00:00Z"
}
```

Response:

```json
{
  "memory_id": "..."
}
```

### Query Memory

```http
POST /query
```

Request:

```json
{
  "scope": "user:123:general",
  "query": "Why was the Atlas rollout delayed?",
  "include_references": true,
  "top_k": 12
}
```

Response:

```json
{
  "query_id": "...",
  "query_type": "causal",
  "answer": "...",
  "references": [
    {
      "memory_id": "...",
      "occurred_at": "2026-05-01T10:00:00Z",
      "quote": "...",
      "relevance": "..."
    }
  ],
  "confidence": 0.78,
  "uncertainties": []
}
```

### Read History

```http
GET /memories?scope=...&from=...&to=...
```

Returns raw memories in one exact scope ordered by `occurred_at`.

### Soft Delete

```http
DELETE /memories/{memory_id}?scope=...
```

Soft deletes one memory only if it belongs to the exact scope.

### Restore

```http
POST /memories/{memory_id}/restore
```

Restores one soft-deleted memory only if it belongs to the exact scope.

### Health

```http
GET /health
```

Response should include:

```json
{
  "status": "ok",
  "db": "ok",
  "retrieval": "ok",
  "index_status": "ready",
  "embedding_provider": "local",
  "embedding_model": "BAAI/bge-base-zh-v1.5",
  "embedding_dim": 768,
  "projection_version": "v1",
  "embedding_prewarm_status": "ready",
  "embedding_prewarm_error": null
}
```

## 10. Error Handling

Validation errors:

- Missing `scope`.
- Missing `content`.
- Missing `occurred_at`.
- Invalid timestamp.
- Memory not found in exact scope.

LLM planner errors:

- Invalid structured output.
- Timeout.
- Provider failure.

Vector retrieval errors:

- Index unavailable.
- Embedding provider unavailable.
- pgvector query failure.
- Index stale or empty.

Answer synthesis errors:

- Invalid structured output.
- Timeout.
- Provider failure.

All query failures should write an audit record with `error_stage`, `error_message`, and timing fields when possible.

## 11. Security And Isolation

- All queries must include exactly one `scope`.
- The service must never expand, infer, or inherit scope.
- Vector metadata filters must include exact `scope` on every search.
- Hydration from `memories` must re-check exact `scope`, even if vector metadata already filtered it.
- Audit lookup must be scoped or protected by an administrative boundary.
- Logs should avoid writing full memory content unless explicitly configured for local debugging.

## 12. Observability

Each query should record timings for:

```text
planner_llm_ms
embedding_ms
vector_search_ms
candidate_hydration_ms
answer_llm_ms
```

Each query should record usage for:

```text
planner model and token usage
answer model and token usage
embedding provider and request count
```

Each query audit should make it possible to answer:

- Did the planner classify the query correctly?
- What rewritten queries were used?
- Did vector search stay inside one scope?
- Which memories were retrieved and with what score?
- Which retrieved memories were used in the final answer?
- Did the answer cite its evidence?
- Which stage failed or dominated latency?

## 13. Testing Strategy

Deterministic tests:

- Write rejects missing `occurred_at`.
- Query rejects missing `scope`.
- Query never searches multiple scopes.
- Hydration excludes memories from other scopes.
- Soft-deleted memories are excluded from normal query results.
- Original memory content cannot be updated.
- Reindex can rebuild vector nodes from `memories`.

Mocked LLM tests:

- Planner output for direct, related, history, and causal queries.
- Invalid planner output returns a clear failure and writes audit.
- Synthesizer returns cited answer from provided candidates.
- Synthesizer refuses to answer when candidates do not support the query.

Retrieval tests:

- Same text in different scopes does not leak.
- Related query can use multiple rewritten queries.
- History answer orders evidence by `occurred_at`.
- Causal answer distinguishes sequence from causality when evidence is weak.

Operational tests:

- Health reports vector index status.
- Embedding model or dimension change requires rebuild.
- Query audit records planner, retrieval, answer, usage, and timing details.

## 14. Open Decisions

- Whether to keep an optional BM25 retriever alongside vector search. The first design only requires vector search, but LlamaIndex can support hybrid retrieval later without changing the business model.
- Whether duplicate identical `content` in the same `scope` should be accepted or rejected. The first design can accept duplicates to avoid adding non-essential hash fields.
- Audit retention period. A default such as 30 days is reasonable, but deployment policy should decide.
- Whether `include_references=false` should remove references only from the API response or also from persisted audit details. The current recommendation is response-only; audit should remain detailed.
