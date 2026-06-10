from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from memoflux.config import load_settings
from memoflux.i18n import i18n_service, normalize_language
from memoflux.llm import LocalLLMClient, OpenAICompatibleLLMClient
from memoflux.models import DeleteAuditRecord, MemoryRecord, QueryAuditRecord, RecallReference
from memoflux.retention import start_memory_retention_cleanup
from memoflux.service import MemoFluxService
from memoflux.storage.postgres import PostgresMemoryRepository

logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    occurred_at: datetime


class RecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    include_references: bool = True
    top_k: int = Field(default=12, ge=1, le=100)


class DeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: str = Field(..., min_length=1)
    memory_ids: list[str] | None = None
    query: str | None = None
    dry_run: bool = True


class PromptEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_key: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


def create_app(*, repository=None, llm_client=None, embedding_client=None, database_path: Path | None = None) -> FastAPI:
    """创建 MemoFlux FastAPI 应用。"""

    settings = load_settings()
    service = MemoFluxService(
        repository or _build_default_repository(),
        llm_client=llm_client or _build_default_llm_client(),
        embedding_client=embedding_client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cleanup_task = None
        init_schema = getattr(service.repository, "init_schema", None)
        if init_schema is not None:
            await init_schema()
        if settings.memory_cleanup_enabled:
            cleanup_task = start_memory_retention_cleanup(service=service, settings=settings)
        try:
            yield
        finally:
            tasks = [task for task in (cleanup_task,) if task is not None]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="MemoFlux", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """按请求语言返回统一 HTTP 错误响应。"""

        lang = _request_language(request)
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=_error("http_error", lang=lang, details={"reason": str(exc.detail)}))

    @app.post("/v1/ingest")
    async def ingest(request: IngestRequest):
        """异步写入一条记忆。"""

        memory = await service.ingest(session=request.session, content=request.content, occurred_at=request.occurred_at)
        return _ok(_memory_payload(memory, include_content=False))

    @app.post("/v1/recall")
    async def recall(request: RecallRequest, accept_language: str | None = Header(default=None)):
        """异步召回记忆并生成答案。"""

        result = await service.recall(
            session=request.session,
            query=request.query,
            top_k=request.top_k,
            lang=normalize_language(accept_language),
        )
        payload = {
            "query_id": result.query_id,
            "answer": result.answer,
            "references": [_reference_payload(item) for item in result.references] if request.include_references else [],
            "confidence": result.confidence,
            "uncertainties": result.uncertainties,
        }
        return _ok(payload)

    @app.post("/v1/delete")
    async def delete(request: DeleteRequest, accept_language: str | None = Header(default=None)):
        """异步删除记忆或执行删除 dry-run。"""

        lang = normalize_language(accept_language)
        try:
            result = await service.delete(
                session=request.session,
                memory_ids=request.memory_ids,
                query=request.query,
                dry_run=request.dry_run,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=_error("validation_error", lang=lang, details={"reason": str(error)}),
            ) from error
        return _ok(result.__dict__)

    @app.get("/v1/i18n/{lang}")
    async def i18n(lang: str):
        """返回指定语言的本地化文案。"""

        return _ok(i18n_service.get_locale(lang))

    @app.post("/v1/prompt-eval")
    async def prompt_eval(request: PromptEvalRequest):
        """异步执行 prompt 调试并记录聚合用量。"""

        result = await service.llm_client.run_prompt(prompt_key=request.prompt_key, payload=request.payload)
        await service.repository.record_usage(
            operation=f"prompt_eval:{request.prompt_key}",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
        )
        logger.info(
            "MemoFlux prompt eval completed",
            extra={
                "prompt_key": request.prompt_key,
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cached_tokens": result.cached_tokens,
                "reasoning_tokens": result.reasoning_tokens,
            },
        )
        return _ok({"status": "ok", "prompt_key": request.prompt_key, "model": result.model, "latency_ms": 0, "output": result.output, "error_code": None, "error_message": None})

    @app.get("/v1/health")
    async def health():
        """返回 MemoFlux 服务健康状态。"""

        return _ok({"status": "ok", "db": "ok", "retrieval": "ok", "llm": "local", "retrieval_strategy": "text_time"})

    @app.get("/v1/preview")
    async def preview(session: str | None = None, limit: int = 50, offset: int = 0):
        """异步预览原始记忆。"""

        items, total = await service.preview(session=session, limit=limit, offset=offset)
        return _ok(
            {
                "items": [_memory_payload(item, include_content=True) for item in items],
                "total": total,
                "limit": limit,
                "offset": offset,
                "next_cursor": None,
            }
        )

    @app.get("/v1/audits")
    async def audits(session: str | None = None, query_id: str | None = None, limit: int = 50, offset: int = 0):
        """异步列出查询和删除审计记录。"""

        items, total = await service.repository.list_audits(
            session=session,
            query_id=query_id,
            limit=limit,
            offset=offset,
        )
        return _ok(
            {
                "items": [_audit_payload(item) for item in items],
                "total": total,
                "limit": limit,
                "offset": offset,
                "next_cursor": None,
            }
        )

    @app.get("/v1/usage/stats")
    async def usage_stats():
        """异步返回聚合用量统计。"""

        stats = await service.usage_stats()
        return _ok({"status": "ok", **stats.__dict__})

    @app.delete("/v1/usage/stats")
    async def clear_usage_stats():
        """异步清空聚合用量统计。"""

        return _ok({"status": "ok", "deleted": await service.clear_usage_stats()})

    return app


def _build_default_repository():
    settings = load_settings()
    if not settings.database_url:
        raise RuntimeError("MEMOFLUX_DATABASE_URL is required")
    return PostgresMemoryRepository(
        database_url=settings.database_url,
        schema=settings.database_schema,
        embedding_dimension=settings.embedding_dimension,
    )

def _build_default_llm_client():
    settings = load_settings()
    if settings.llm_base_url and settings.llm_api_key:
        return OpenAICompatibleLLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    return LocalLLMClient()


def _ok(data):
    return {"data": data, "error": None}


def _error(code: str, *, lang: str | None = None, details: dict[str, Any] | None = None):
    return {
        "data": None,
        "error": {
            "code": code,
            "message": i18n_service.t(f"errors.{code}", lang=lang),
            "details": details or {},
            "query_id": None,
        },
    }


def _request_language(request: Request) -> str:
    """从请求中解析语言。"""

    lang = request.query_params.get("lang")
    if lang:
        return normalize_language(lang)
    return normalize_language(request.headers.get("accept-language"))


def _memory_payload(memory: MemoryRecord, *, include_content: bool) -> dict[str, Any]:
    payload = {
        "memory_id": memory.memory_id,
        "session": memory.session,
        "occurred_at": memory.occurred_at.isoformat(),
        "created_at": memory.created_at.isoformat(),
    }
    if include_content:
        payload["content"] = memory.content
    return payload


def _reference_payload(reference: RecallReference) -> dict[str, Any]:
    return {
        "memory_id": reference.memory_id,
        "occurred_at": reference.occurred_at.isoformat(),
        "quote": reference.quote,
        "relevance": reference.relevance,
    }


def _audit_payload(audit: QueryAuditRecord | DeleteAuditRecord) -> dict[str, Any]:
    """序列化审计记录，避免返回单次 token 消耗。"""

    if isinstance(audit, QueryAuditRecord):
        return {
            "audit_type": "query",
            "query_id": audit.query_id,
            "session": audit.session,
            "query": audit.original_query,
            "original_query": audit.original_query,
            "rewritten_queries": audit.rewritten_queries,
            "query_plan": {
                "rewritten_queries": audit.rewritten_queries,
            },
            "candidate_limit": audit.candidate_limit,
            "retrieved": audit.retrieved,
            "selected_memory_ids": audit.selected_memory_ids,
            "selection_reasons": {
                key: value
                for key, value in audit.selection_reasons.items()
                if not str(key).startswith("__")
            },
            "answerability": audit.selection_reasons.get("__answerability", "unknown"),
            "answerability_reason": audit.selection_reasons.get("__answerability_reason", ""),
            "final_answer": audit.final_answer,
            "status": audit.status,
            "error_stage": audit.error_stage,
            "error_message": audit.error_message,
            "created_at": audit.created_at.isoformat(),
        }
    return {
        "audit_type": "delete",
        "delete_id": audit.delete_id,
        "session": audit.session,
        "target": audit.target,
        "dry_run": audit.dry_run,
        "matched_memory_ids": audit.matched_memory_ids,
        "affected_memory_ids": audit.affected_memory_ids,
        "status": audit.status,
        "error_code": audit.error_code,
        "error_message": audit.error_message,
        "created_at": audit.created_at.isoformat(),
    }
