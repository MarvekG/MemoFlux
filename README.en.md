# MemoFlux

MemoFlux is a lightweight memory service for LLM applications. It stores original memory text under an explicit `session`, retrieves relevant memories with embeddings and pgvector, and asks an LLM to synthesize an answer with auditable references.

## What It Does

MemoFlux provides a small HTTP service for long-term memory workflows:

- Ingest raw memory text with an occurrence time.
- Keep memories isolated by `session`.
- Recall memories through query planning, batch embedding, pgvector retrieval, chronological fallback, and answer synthesis.
- Return only the memories that the Answer Synthesizer actually used.
- Record query/delete audits without exposing raw provider responses or per-call token details.
- Aggregate token usage by operation for cost and cache analysis.

## Why MemoFlux

- **Session isolation:** every ingest, recall, delete, preview, and audit operation is scoped by `session`.
- **Original-text storage:** the service stores the source memory text instead of lossy summaries.
- **Vector-first retrieval:** memories are embedded and searched through PostgreSQL with pgvector.
- **Batch recall embeddings:** the original query and rewritten retrieval queries are embedded in one batch before per-query vector search.
- **Evidence-aware answers:** references come from `used_memory_ids` selected by the Answer Synthesizer, not from every retrieved candidate.
- **Auditable behavior:** query audits preserve rewritten queries, retrieved candidates, selected IDs, answerability diagnostics, and final answers.
- **Provider boundary:** LLM keys and real model routing stay in LiteLLM or another OpenAI-compatible gateway; MemoFlux only consumes the compatible API.

## Architecture

```text
Client
  -> POST /v1/ingest
     -> embedding provider
     -> PostgreSQL / pgvector

Client
  -> POST /v1/recall
     -> Query Planner LLM
     -> original query + rewritten queries
     -> batch embedding
     -> pgvector search per retrieval query
     -> chronological list fallback
     -> merge candidates by memory_id
     -> Answer Synthesizer LLM
     -> references filtered by used_memory_ids
     -> query audit
```

Runtime components:

- `memoflux/api.py`: FastAPI routes and response shaping.
- `memoflux/service.py`: ingest, recall, delete, preview, usage, and audit orchestration.
- `memoflux/services/embedding_service.py`: local sentence-transformer or OpenAI-compatible embedding client.
- `memoflux/llm.py`: local fallback LLM and OpenAI-compatible Chat Completions client.
- `memoflux/storage/postgres.py`: SQLAlchemy repository backed by PostgreSQL and pgvector.
- `memoflux/storage/schema.py`: table definitions under the `memoflux` schema by default.

## Requirements

- Python 3.11+
- PostgreSQL with pgvector, for example `pgvector/pgvector:pg17`
- An OpenAI-compatible Chat Completions endpoint for production recall quality, usually LiteLLM
- A local sentence-transformer model or an OpenAI-compatible embeddings endpoint

## Configuration

Create `.env` from `.env.example` and adjust the values for your deployment.

Core settings:

```bash
MEMOFLUX_DATABASE_URL=postgresql+psycopg2://tradeuser:tradepassword@memo-postgres:5432/memory
MEMOFLUX_DATABASE_SCHEMA=memoflux
MEMOFLUX_SERVICE_PORT=8020
MEMOFLUX_APP_RELOAD=true
```

LLM settings:

```bash
MEMOFLUX_LLM_BASE_URL=http://litellm:4000/v1
MEMOFLUX_LLM_API_KEY=sk-litellm-gateway-key
MEMOFLUX_LLM_MODEL=memory
```

Local embedding settings:

```bash
MEMOFLUX_EMBEDDING_PROVIDER=local
MEMOFLUX_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
MEMOFLUX_EMBEDDING_DIM=768
MEMOFLUX_EMBEDDING_CACHE_DIR=/home/memo/.memoflux/data/models
MEMOFLUX_EMBEDDING_PREWARM_ON_STARTUP=true
```

OpenAI-compatible embedding settings:

```bash
MEMOFLUX_EMBEDDING_PROVIDER=openai_compatible
MEMOFLUX_EMBEDDING_BASE_URL=http://litellm:4000/v1
MEMOFLUX_EMBEDDING_API_KEY=sk-litellm-gateway-key
MEMOFLUX_EMBEDDING_MODEL=your-embedding-model
MEMOFLUX_EMBEDDING_DIM=768
```

Real provider keys, real provider model names, and routing rules should stay in the LiteLLM configuration. Do not put provider keys in MemoFlux source code.

## Deployment

### Standalone Docker Compose

The `memo/docker-compose.yml` file starts MemoFlux with a local pgvector database. From this directory:

```bash
cp .env.example .env
docker compose up -d
```

The standalone compose file exposes:

- MemoFlux API on `http://127.0.0.1:8020`
- PostgreSQL on host port `5433`

If you use the provided standalone compose file without editing it, make sure `.env` matches the standalone database service name, database, and credentials:

```bash
MEMOFLUX_DATABASE_URL=postgresql+psycopg2://memouser:memopassword@memo-postgres:5432/memo
MEMOFLUX_SERVICE_PORT=8010
```

`MEMOFLUX_SERVICE_PORT=8010` matches the container port used by `memo/docker-compose.yml`; the API is still published on host port `8020`.

## API

Current endpoints:

```text
POST   /v1/ingest
POST   /v1/recall
POST   /v1/prompt-eval
POST   /v1/delete
GET    /v1/health
GET    /v1/preview
GET    /v1/audits
GET    /v1/usage/stats
DELETE /v1/usage/stats
```

### Ingest

```bash
curl -X POST http://127.0.0.1:8020/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "session": "project:atlas",
    "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
    "occurred_at": "2026-05-01T10:00:00Z"
  }'
```

### Recall

```bash
curl -X POST http://127.0.0.1:8020/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{
    "session": "project:atlas",
    "query": "Atlas 为什么延期？",
    "top_k": 12
  }'
```

Recall uses the original query plus LLM rewritten queries for vector retrieval. It embeds those retrieval queries in one batch, searches each vector in the same session, merges candidates by `memory_id`, appends a chronological fallback candidate pool, and lets the Answer Synthesizer choose the final answer and references.

If the candidate memories do not support the question, the answer is `当前 session 中没有足够记忆支持回答该问题。` and references are empty.

### Audits And Usage

Query audits expose the validated planner output as `query`, `rewritten_queries`, and `query_plan`, while omitting prompts, raw provider responses, and single-call token usage.

`/v1/usage/stats` returns aggregate token usage, including `cached_tokens`, `cache_miss_tokens`, `reasoning_tokens`, and `cache_hit_rate` for cache efficiency analysis.

## Performance Notes

- Recall currently performs one Query Planner LLM call and, when candidates exist, one Answer Synthesizer LLM call.
- The dominant latency is usually the Chat Completions provider, not PostgreSQL.
- Retrieval query embeddings are batched with `embed_texts(retrieval_queries)`.
- The `memories.embedding` column does not currently create an ANN index by default; large sessions can make vector search scale linearly with the number of memories in that session.
- A measured local Docker run with 10 concurrent recall requests completed with average latency around `4.22s` and p95 around `5.09s`. Treat this as an environment-specific reference, not a guarantee.

Recommended next optimization: evaluate a planner fallback mode that first tries the original query and calls the Query Planner only when the initial candidate pool is insufficient. Run recall evals before changing the default, because skipping the planner can reduce accuracy on complex or paraphrased queries.

## Development

Install dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

Run the server locally:

```bash
python run.py
```

Run API tests:

```bash
pytest tests/test_api.py
```

## Recall Evals

Generate the mixed recall suite and run it against a local MemoFlux service:

```bash
python -m evals.scripts.generate_mixed_1000_suite --output evals/cases/mixed_1000.json
python -m evals.scripts.eval_memoflux_recall --suite evals/cases/mixed_1000.json --base-url http://127.0.0.1:8020
```

For shorter failure-focused reports, keep only failed details and write the JSON report to a file:

```bash
python -m evals.scripts.eval_memoflux_recall \
  --suite evals/cases/mixed_1000.json \
  --base-url http://127.0.0.1:8020 \
  --only-failures \
  --output /tmp/memoflux_failures.json
```

## License

MemoFlux is released under the MIT License. See [LICENSE](LICENSE).
