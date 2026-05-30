from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from memoflux.config import load_settings
from memoflux.llm import LocalLLMClient, OpenAICompatibleLLMClient
from memoflux.models import DeleteAuditRecord, MemoryRecord, QueryAuditRecord, RecallReference
from memoflux.service import MemoFluxService
from memoflux.storage.postgres import PostgresMemoryRepository


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

    app = FastAPI(title="MemoFlux", version="0.1.0")
    service = MemoFluxService(
        repository or _build_default_repository(),
        llm_client=llm_client or _build_default_llm_client(),
        embedding_client=embedding_client,
    )
    app.state.service = service

    @app.exception_handler(HTTPException)
    def http_exception_handler(_, exc: HTTPException):
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=_error("http_error", str(exc.detail)))

    @app.post("/v1/ingest")
    def ingest(request: IngestRequest):
        memory = service.ingest(session=request.session, content=request.content, occurred_at=request.occurred_at)
        return _ok(_memory_payload(memory, include_content=False))

    @app.post("/v1/recall")
    def recall(request: RecallRequest):
        result = service.recall(session=request.session, query=request.query, top_k=request.top_k)
        payload = {
            "query_id": result.query_id,
            "query_type": result.query_type,
            "answer": result.answer,
            "references": [_reference_payload(item) for item in result.references] if request.include_references else [],
            "confidence": result.confidence,
            "uncertainties": result.uncertainties,
        }
        return _ok(payload)

    @app.post("/v1/delete")
    def delete(request: DeleteRequest):
        try:
            result = service.delete(
                session=request.session,
                memory_ids=request.memory_ids,
                query=request.query,
                dry_run=request.dry_run,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=_error("validation_error", str(error))) from error
        return _ok(result.__dict__)

    @app.post("/v1/prompt-eval")
    def prompt_eval(request: PromptEvalRequest):
        result = service.llm_client.run_prompt(prompt_key=request.prompt_key, payload=request.payload)
        service.repository.record_usage(operation=f"prompt_eval:{request.prompt_key}", input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        return _ok({"status": "ok", "prompt_key": request.prompt_key, "model": result.model, "latency_ms": 0, "output": result.output, "error_code": None, "error_message": None})

    @app.get("/v1/health")
    def health():
        return _ok({"status": "ok", "db": "ok", "retrieval": "ok", "llm": "local", "retrieval_strategy": "text_time"})

    @app.get("/v1/preview")
    def preview(session: str, limit: int = 50):
        return _ok({"items": [_memory_payload(item, include_content=True) for item in service.preview(session=session, limit=limit)], "next_cursor": None})

    @app.get("/v1/audits")
    def audits(session: str, query_id: str | None = None, limit: int = 50):
        items = service.repository.list_audits(session=session, query_id=query_id, limit=limit)
        return _ok({"items": [_audit_payload(item) for item in items], "next_cursor": None})

    @app.get("/v1/usage/stats")
    def usage_stats():
        stats = service.usage_stats()
        return _ok({"status": "ok", **stats.__dict__})

    @app.delete("/v1/usage/stats")
    def clear_usage_stats():
        return _ok({"status": "ok", "deleted": service.clear_usage_stats()})

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


def _error(code: str, message: str):
    return {"data": None, "error": {"code": code, "message": message, "details": {}, "query_id": None}}


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
            "original_query": audit.original_query,
            "query_type": audit.query_type,
            "rewritten_queries": audit.rewritten_queries,
            "candidate_limit": audit.candidate_limit,
            "retrieved": audit.retrieved,
            "selected_memory_ids": audit.selected_memory_ids,
            "selection_reasons": audit.selection_reasons,
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
