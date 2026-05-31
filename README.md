# MemoFlux

MemoFlux is a lightweight memory service that stores original memory text by explicit session and recalls memories through query rewrite, pgvector retrieval, PostgreSQL text/time fallback, and answer synthesis.

## Runtime LLM

MemoFlux talks to a real LLM through an OpenAI-compatible Chat Completions endpoint. In the Best-AI-Trader stack this should point to LiteLLM.

```bash
export MEMOFLUX_LLM_BASE_URL='http://127.0.0.1:4000/v1'
export MEMOFLUX_LLM_API_KEY='sk-change-me'
export MEMOFLUX_LLM_MODEL='memory/default'
```

When running inside Docker Compose, use service names:

```bash
MEMOFLUX_LLM_BASE_URL=http://litellm:4000/v1
```

Real provider keys and model aliases stay in the LiteLLM configuration. Do not put provider keys in MemoFlux.

## Runtime Database

MemoFlux uses the root Compose `memory-postgres` service for persistence. That container uses `pgvector/pgvector:pg17`, so this version stores vectors and performs vector recall through SQLAlchemy and pgvector.

Use an independent environment variable for MemoFlux:

```bash
export MEMOFLUX_DATABASE_URL='postgresql+psycopg2://tradeuser:tradepassword@memory-postgres:5432/memory'
```

MemoFlux keeps its tables under the `memoflux` schema by default.

```bash
export MEMOFLUX_DATABASE_SCHEMA='memoflux'
```

The service uses SQLAlchemy ORM mappings and creates the schema/tables on startup if they do not exist. The `memories.embedding` column uses pgvector.

Embedding configuration:

```bash
export MEMOFLUX_EMBEDDING_PROVIDER='local'
export MEMOFLUX_EMBEDDING_MODEL='BAAI/bge-base-zh-v1.5'
export MEMOFLUX_EMBEDDING_DIM='768'
export MEMOFLUX_EMBEDDING_CACHE_DIR='/home/memory/.insight_memory/data/models'
```

In the Best-AI-Trader dev Compose stack, `memo` mounts the same `memory_runtime_data` volume used by the legacy memory service so local sentence-transformer models are shared under `/home/memory/.insight_memory/data/models`.

## Run With Docker

MemoFlux is included in the root `docker-compose.yml`. In `docker-compose.dev.yml`, the legacy `memory` service is replaced by `memo` on port `8020`. For standalone development, create `.env` from `.env.example`, set the LiteLLM key, then run:

```bash
docker compose -f docker-compose.example.yml up -d --build
```

The example compose file joins the existing Best-AI-Trader Docker network so `memory-postgres` and `litellm` are reachable by service name.

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

Recall uses the original query plus LLM rewritten queries for vector retrieval, merges candidates by `memory_id`, and returns references only for `used_memory_ids` selected by the Answer Synthesizer. If the candidate memories do not support the question, the answer is `当前 session 中没有足够记忆支持回答该问题。` and references are empty.

Query audits expose the validated planner output as `query`, `query_type`, `rewritten_queries`, and `query_plan`, while omitting prompts, raw provider responses, and single-call token usage. `/v1/usage/stats` returns aggregate token usage, including `cached_tokens`, `cache_miss_tokens`, `reasoning_tokens`, and `cache_hit_rate` for cache efficiency analysis.

## Run Tests

```bash
pytest tests/test_api.py
```

## Run Recall Evals

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
