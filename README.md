# MemoFlux

[English](README.en.md)

MemoFlux 是一个面向大语言模型应用的轻量级长期记忆服务。它按显式 `session` 存储原始记忆文本，使用向量表示和 pgvector 找回相关记忆，再交给大语言模型生成带可审计引用的答案。

在当前本地配置的 `mixed_1000` 评测集中，MemoFlux 在 1000 条混合记忆、100 个问题上达到 `100%` 答案正确率、`100%` 严格正确率、`100%` 召回命中率和 `100%` 证据选择命中率。这个结果来自“先找准证据，再基于证据回答”的链路设计：写入阶段保留原始记忆文本，避免摘要丢失细节；召回阶段同时使用原始问题和大语言模型改写后的问题扩大语义覆盖；候选阶段按 `session` 严格隔离，降低跨用户、跨项目污染；生成阶段要求答案生成器只基于候选记忆回答，并明确声明实际使用的记忆编号。如果候选记忆无法支持问题，MemoFlux 会返回证据不足的固定拒答，而不是让模型自由猜测。

## 作用

MemoFlux 提供一组简单的 HTTP 接口，用于长期记忆工作流：

- 写入带发生时间的原始记忆文本。
- 按 `session` 隔离不同用户、项目、任务或智能体的记忆。
- 通过查询规划、批量向量化、pgvector 检索、时间顺序补充候选和答案生成完成召回。
- 只返回答案生成器实际使用的记忆引用。
- 记录查询和删除审计，便于排查召回质量问题。
- 聚合记录模型调用用量，用于成本和缓存命中率分析。

## 优势

- **显式隔离：** 所有写入、召回、删除、预览、审计操作都必须带 `session`，默认不跨 session 查询。
- **保留原文：** 存储原始记忆文本，避免提前摘要造成信息损失。
- **向量优先：** 使用 PostgreSQL + pgvector 做语义召回，同时保留时间序候选补充。
- **批量召回向量化：** 原始问题和大语言模型改写后的问题会一次性批量生成向量，再分别执行向量检索。
- **引用更克制：** 接口返回的引用只包含最终答案声明使用的记忆，不会把所有候选都暴露给调用方。
- **可审计：** 审计记录会保存改写后的问题、候选记忆、最终选择、是否可回答的诊断和最终答案。
- **模型网关解耦：** MemoFlux 只对接 OpenAI 兼容接口；真实供应商密钥、模型名和路由规则放在 LiteLLM 等网关中。

## 架构

```text
客户端
  -> POST /v1/ingest
     -> 向量模型
     -> PostgreSQL / pgvector

客户端
  -> POST /v1/recall
     -> 查询规划模型
     -> 原始问题 + 改写后的问题
     -> 批量生成向量
     -> 每个检索问题执行 pgvector 检索
     -> 按时间顺序补充候选
     -> 按 memory_id 合并候选
     -> 答案生成模型
     -> 只保留答案实际使用的引用
     -> 写入查询审计
```

核心模块：

- `memoflux/api.py`：FastAPI 路由和响应结构。
- `memoflux/service.py`：写入、召回、删除、预览、用量统计和审计编排。
- `memoflux/services/embedding_service.py`：本地向量模型或 OpenAI 兼容向量接口客户端。
- `memoflux/llm.py`：本地占位模型与 OpenAI 兼容对话接口客户端。
- `memoflux/storage/postgres.py`：基于 SQLAlchemy、PostgreSQL 和 pgvector 的数据访问层。
- `memoflux/storage/schema.py`：默认 `memoflux` schema 下的数据表定义。

## 运行要求

- Python 3.11+
- PostgreSQL + pgvector，例如 `pgvector/pgvector:pg17`
- OpenAI 兼容的对话模型接口，生产建议走 LiteLLM
- 本地 sentence-transformer 向量模型或 OpenAI 兼容的向量接口

## 配置

从 `.env.example` 复制 `.env`，再按部署环境调整。

核心配置：

```bash
MEMOFLUX_SERVICE_PORT=8020
```

默认数据库连接为 `postgresql+asyncpg://tradeuser:tradepassword@memo-postgres:5432/memory`，通常不需要在 `.env` 中配置。

LLM 配置：

```bash
MEMOFLUX_LLM_BASE_URL=http://litellm:4000/v1
MEMOFLUX_LLM_API_KEY=sk-litellm-gateway-key
MEMOFLUX_LLM_MODEL=openai-compatible
```

本地向量模型配置：

```bash
MEMOFLUX_EMBEDDING_PROVIDER=local
MEMOFLUX_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
MEMOFLUX_EMBEDDING_DIM=768
```

官方镜像在构建阶段会把默认向量模型下载到 `/home/memo/.memoflux/data/models`，运行时默认只读取本地权重，不再联网下载。

OpenAI 兼容向量接口配置：

```bash
MEMOFLUX_EMBEDDING_PROVIDER=openai_compatible
MEMOFLUX_EMBEDDING_BASE_URL=http://litellm:4000/v1
MEMOFLUX_EMBEDDING_API_KEY=sk-litellm-gateway-key
MEMOFLUX_EMBEDDING_MODEL=your-embedding-model
MEMOFLUX_EMBEDDING_DIM=768
```

真实供应商密钥、真实供应商模型名和模型路由规则应放在 LiteLLM 等网关配置中，不要写入 MemoFlux 代码。

## 部署

### 独立 Docker Compose

`memo/docker-compose.yml` 会启动 MemoFlux 和本地 pgvector 数据库。在 `memo/` 目录下运行：

```bash
cp .env.example .env
docker compose up -d
```

独立 Docker Compose 默认暴露：

- MemoFlux API：`http://127.0.0.1:8020`
- PostgreSQL：host port `5433`

如果直接使用当前 `memo/docker-compose.yml`，只需要在 `.env` 中按需调整容器端口：

```bash
MEMOFLUX_SERVICE_PORT=8010
```

`MEMOFLUX_SERVICE_PORT=8010` 对应 `memo/docker-compose.yml` 的容器内端口；宿主机访问仍是 `http://127.0.0.1:8020`。

## API

当前 HTTP 接口：

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

### 写入记忆

```bash
curl -X POST http://127.0.0.1:8020/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "session": "project:atlas",
    "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
    "occurred_at": "2026-05-01T10:00:00Z"
  }'
```

### 召回记忆

```bash
curl -X POST http://127.0.0.1:8020/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{
    "session": "project:atlas",
    "query": "Atlas 为什么延期？",
    "top_k": 12
  }'
```

召回会使用原始问题和大语言模型改写后的问题做向量检索。服务会批量生成这些检索问题的向量，在同一 session 内逐条执行向量查询，按 `memory_id` 合并候选，再追加按时间顺序排列的候选池，最后由答案生成器输出答案和引用。

如果候选记忆不足以回答问题，默认答案是 `当前 session 中没有足够记忆支持回答该问题。`，引用列表为空。

### 审计和用量

查询审计会暴露 `query`、`rewritten_queries`、`query_plan`、候选和选择结果，但不会返回提示词、供应商原始响应或单次模型用量明细。

`/v1/usage/stats` 返回按操作类型聚合的模型用量，包括已命中缓存的 token、未命中缓存的 token、推理 token 和缓存命中率。

## 性能说明

- 召回当前通常包含一次查询规划模型调用；有候选时还会包含一次答案生成模型调用。
- 主要延迟通常来自对话模型供应商，而不是 PostgreSQL。
- 检索问题的向量已通过 `embed_texts(retrieval_queries)` 批量生成。
- `memories.embedding` 当前默认不创建 ANN 索引；单个 session 记忆量很大时，向量检索会随 session 内记忆数量线性增长。
- 本地 Docker 实测 10 并发召回的平均时延约 `4.22s`，p95 约 `5.09s`。该数据只作为当前环境参考，不代表稳定服务承诺。

推荐的下一步优化是评估“查询规划按需启用”模式：先用原始问题召回，候选不足时再调用查询规划模型。跳过查询规划可能降低复杂改写问题的准确率，因此变更默认行为前应先跑召回评测对比。

## 开发

安装依赖：

```bash
pip install -r requirements.txt
```

本地启动：

```bash
python run.py
```

运行 API 测试：

```bash
pytest tests/test_api.py
```

## 召回评测

生成混合召回评测集，并对本地 MemoFlux 服务执行评测：

```bash
python -m evals.scripts.generate_mixed_1000_suite --output evals/cases/mixed_1000.json
python -m evals.scripts.eval_memoflux_recall --suite evals/cases/mixed_1000.json --base-url http://127.0.0.1:8020
```

只保留失败明细并写入 JSON 文件：

```bash
python -m evals.scripts.eval_memoflux_recall \
  --suite evals/cases/mixed_1000.json \
  --base-url http://127.0.0.1:8020 \
  --only-failures \
  --output /tmp/memoflux_failures.json
```

## 许可证

MemoFlux 使用 MIT License 发布，详见 [LICENSE](LICENSE)。
