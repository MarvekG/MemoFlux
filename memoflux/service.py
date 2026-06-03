from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from memoflux.i18n import i18n_service
from memoflux.llm import LLMResult, LocalLLMClient
from memoflux.models import DeleteAuditRecord, DeleteResult, QueryAuditRecord, RecallReference, RecallResult
from memoflux.services.embedding_service import embedding_service

NO_ANSWER_KEY = "answers.no_memory"


class MemoFluxService:
    """MemoFlux 业务服务。"""

    def __init__(self, repository, llm_client=None, embedding_client=None) -> None:
        self.repository = repository
        self.llm_client = llm_client or LocalLLMClient()
        self.embedding_service = embedding_client or embedding_service

    def ingest(self, *, session: str, content: str, occurred_at: datetime):
        """写入一条记忆。"""

        if not session:
            raise ValueError("session is required")
        if not content:
            raise ValueError("content is required")
        if occurred_at is None:
            raise ValueError("occurred_at is required")
        embedding = LLMResult(output={"embedding": self.embedding_service.embed_text(content)})
        self.repository.record_usage(
            operation="embedding:ingest",
            input_tokens=embedding.input_tokens,
            output_tokens=embedding.output_tokens,
            cached_tokens=embedding.cached_tokens,
            reasoning_tokens=embedding.reasoning_tokens,
        )
        return self.repository.insert_memory(
            session=session,
            content=content,
            occurred_at=occurred_at,
            embedding=embedding.output.get("embedding"),
        )

    def recall(self, *, session: str, query: str, top_k: int = 12, lang: str | None = None) -> RecallResult:
        """召回同一 session 内的候选记忆并生成答案。"""

        if not session:
            raise ValueError("session is required")
        if not query:
            raise ValueError("query is required")
        query_id = uuid4().hex
        plan = self.llm_client.plan_query(query=query)
        rewritten_queries = plan.output.get("rewritten_queries") or [query]
        retrieval_queries = _build_retrieval_queries(query, rewritten_queries)
        candidate_limit = _candidate_pool_limit(top_k)
        memory_batches = []
        query_embeddings = self.embedding_service.embed_texts(retrieval_queries)
        if len(query_embeddings) != len(retrieval_queries):
            raise ValueError("recall embedding count does not match retrieval query count")
        for embedding in query_embeddings:
            query_embedding = LLMResult(output={"embedding": embedding})
            self.repository.record_usage(
                operation="embedding:recall",
                input_tokens=query_embedding.input_tokens,
                output_tokens=query_embedding.output_tokens,
                cached_tokens=query_embedding.cached_tokens,
                reasoning_tokens=query_embedding.reasoning_tokens,
            )
            memory_batches.append(
                self.repository.search_memories(
                    session=session,
                    terms=set(),
                    limit=candidate_limit,
                    query_embedding=query_embedding.output.get("embedding"),
                )
            )
        listed_memories, _ = self.repository.list_memories(session=session, limit=candidate_limit)
        memory_batches.append(listed_memories)
        memories = _merge_memories_by_id(memory_batches, limit=candidate_limit)
        self.repository.record_usage(
            operation="query_planner",
            input_tokens=plan.input_tokens,
            output_tokens=plan.output_tokens,
            cached_tokens=plan.cached_tokens,
            reasoning_tokens=plan.reasoning_tokens,
        )
        if not memories:
            result = RecallResult(query_id=query_id, answer=i18n_service.t(NO_ANSWER_KEY, lang=lang), references=[])
            self.repository.record_query_audit(
                QueryAuditRecord(
                    query_id=query_id,
                    session=session,
                    original_query=query,
                    rewritten_queries=[str(item) for item in rewritten_queries],
                    candidate_limit=top_k,
                retrieved=[],
                selected_memory_ids=[],
                selection_reasons={},
                final_answer=result.answer,
                    status="ok",
                    error_stage=None,
                    error_message=None,
                    created_at=_now_utc(),
                )
            )
            return result
        synthesis = self.llm_client.synthesize_answer(query=query, memories=memories[:top_k])
        self.repository.record_usage(
            operation="answer_synthesizer",
            input_tokens=synthesis.input_tokens,
            output_tokens=synthesis.output_tokens,
            cached_tokens=synthesis.cached_tokens,
            reasoning_tokens=synthesis.reasoning_tokens,
        )
        selection_reasons = _filter_selection_reasons(memories, synthesis.output.get("relevance_by_id") or {})
        used_memories = _filter_used_memories(memories, list(synthesis.output.get("used_memory_ids") or []))
        used_memory_ids = {memory.memory_id for memory in used_memories}
        selected_reasons = {
            memory_id: reason
            for memory_id, reason in selection_reasons.items()
            if memory_id in used_memory_ids
        }
        selected_reasons["__answerability"] = str(synthesis.output.get("answerability") or "unknown")
        selected_reasons["__answerability_reason"] = str(synthesis.output.get("answerability_reason") or "")
        references = [
            RecallReference(
                memory_id=memory.memory_id,
                occurred_at=memory.occurred_at,
                quote=memory.content,
                relevance=selected_reasons.get(
                    memory.memory_id,
                    "Answer Synthesizer 声明使用该记忆，但未提供引用理由。",
                ),
            )
            for memory in used_memories
        ]
        result = RecallResult(
            query_id=query_id,
            answer=str(synthesis.output.get("answer") or "\n".join(memory.content for memory in memories)),
            references=references,
            confidence=_coerce_confidence(synthesis.output.get("confidence")),
            uncertainties=[str(item) for item in synthesis.output.get("uncertainties") or []],
        )
        self.repository.record_query_audit(
            QueryAuditRecord(
                query_id=query_id,
                session=session,
                original_query=query,
                rewritten_queries=[str(item) for item in rewritten_queries],
                candidate_limit=top_k,
                retrieved=[_audit_memory_item(memory) for memory in memories],
                selected_memory_ids=[memory.memory_id for memory in used_memories],
                selection_reasons=selected_reasons,
                final_answer=result.answer,
                status="ok",
                error_stage=None,
                error_message=None,
                created_at=_now_utc(),
            )
        )
        return result

    def delete(self, *, session: str, memory_ids: list[str] | None, query: str | None, dry_run: bool) -> DeleteResult:
        """删除记忆；query 模式只允许 dry-run。"""

        if not session:
            raise ValueError("session is required")
        if query and not dry_run:
            raise ValueError("query delete requires dry_run=true")
        if not memory_ids and not query:
            raise ValueError("memory_ids or query is required")
        delete_id = uuid4().hex
        if query:
            query_embedding = LLMResult(output={"embedding": self.embedding_service.embed_text(query)})
            self.repository.record_usage(
                operation="embedding:delete_dry_run",
                input_tokens=query_embedding.input_tokens,
                output_tokens=query_embedding.output_tokens,
                cached_tokens=query_embedding.cached_tokens,
                reasoning_tokens=query_embedding.reasoning_tokens,
            )
            candidates = self.repository.search_memories(
                session=session,
                terms=set(),
                limit=12,
                query_embedding=query_embedding.output.get("embedding"),
            )
            matched = [memory.memory_id for memory in candidates]
            self.repository.record_usage(
                operation="delete_dry_run",
                input_tokens=_estimate_tokens(query),
                output_tokens=0,
            )
            self.repository.record_delete_audit(
                DeleteAuditRecord(
                    delete_id=delete_id,
                    session=session,
                    target={"query": query},
                    dry_run=True,
                    matched_memory_ids=matched,
                    affected_memory_ids=[],
                    status="ok",
                    error_code=None,
                    error_message=None,
                    created_at=_now_utc(),
                )
            )
            return DeleteResult(delete_id=delete_id, matched_memory_ids=matched, affected_memory_ids=[], status="ok")
        matched = self.repository.delete_memories(session=session, memory_ids=memory_ids or [])
        self.repository.scrub_deleted_memory_from_audits(session=session, memory_ids=matched)
        self.repository.record_delete_audit(
            DeleteAuditRecord(
                delete_id=delete_id,
                session=session,
                target={"memory_ids": memory_ids or []},
                dry_run=dry_run,
                matched_memory_ids=matched,
                affected_memory_ids=matched,
                status="ok",
                error_code=None,
                error_message=None,
                created_at=_now_utc(),
            )
        )
        return DeleteResult(delete_id=delete_id, matched_memory_ids=matched, affected_memory_ids=matched, status="ok")

    def preview(self, *, session: str | None = None, limit: int = 50, offset: int = 0):
        """预览原始记忆，未指定 session 时返回所有 session。"""

        return self.repository.list_memories(session=session, limit=limit, offset=offset)

    def usage_stats(self):
        """返回聚合用量统计。"""

        return self.repository.usage_stats()

    def clear_usage_stats(self) -> int:
        """清空聚合用量统计。"""

        return self.repository.clear_usage_stats()

    def cleanup_expired_memories(self, *, retention_days: int | None = None) -> dict[str, Any]:
        """按发生时间清理超过保留期的记忆。

        Args:
            retention_days: 记忆保留天数；为空时使用默认保留期。

        Returns:
            清理结果摘要，包含删除数量、session 数、保留天数和截止时间。
        """

        normalized_retention_days = _coerce_retention_days(retention_days)
        cutoff = _now_utc() - timedelta(days=normalized_retention_days)
        deleted_by_session = self.repository.delete_memories_before_occurred_at(cutoff)
        deleted_total = 0
        for session, memory_ids in deleted_by_session.items():
            if not memory_ids:
                continue
            deleted_total += len(memory_ids)
            self.repository.scrub_deleted_memory_from_audits(session=session, memory_ids=memory_ids)
            self.repository.record_delete_audit(
                DeleteAuditRecord(
                    delete_id=uuid4().hex,
                    session=session,
                    target={
                        "mode": "retention_cleanup",
                        "retention_days": normalized_retention_days,
                        "cutoff_occurred_at": cutoff.isoformat(),
                    },
                    dry_run=False,
                    matched_memory_ids=memory_ids,
                    affected_memory_ids=memory_ids,
                    status="ok",
                    error_code=None,
                    error_message=None,
                    created_at=_now_utc(),
                )
            )
        return {
            "status": "ok",
            "deleted": deleted_total,
            "sessions": sum(1 for memory_ids in deleted_by_session.values() if memory_ids),
            "retention_days": normalized_retention_days,
            "cutoff_occurred_at": cutoff.isoformat(),
        }


def _filter_used_memories(memories, used_memory_ids: list[str]) -> list:
    """按 LLM 声明使用的记忆 ID 严格过滤候选。

    Args:
        memories: 候选记忆列表。
        used_memory_ids: LLM 输出的实际引用记忆 ID。

    Returns:
        用于 API references 的记忆列表。缺失或非法 ID 不退回全部候选。
    """

    if not used_memory_ids:
        return []
    by_id = {memory.memory_id: memory for memory in memories}
    return [by_id[memory_id] for memory_id in used_memory_ids if memory_id in by_id]


def _filter_selection_reasons(memories, relevance_by_id: dict) -> dict[str, str]:
    """过滤 Answer Synthesizer 输出的引用理由，只保留候选记忆 ID。

    Args:
        memories: 候选记忆列表。
        relevance_by_id: LLM 输出的 memory_id 到引用理由映射。

    Returns:
        只包含候选记忆 ID 的引用理由。
    """

    candidate_ids = {memory.memory_id for memory in memories}
    return {
        str(memory_id): str(reason)
        for memory_id, reason in dict(relevance_by_id or {}).items()
        if memory_id in candidate_ids
    }


def _build_retrieval_queries(query: str, rewritten_queries: list) -> list[str]:
    """构建去重后的检索 query 列表。

    Args:
        query: 原始查询。
        rewritten_queries: Query Planner 输出的改写查询。

    Returns:
        原始查询优先、去重保序的检索 query 列表。
    """

    queries: list[str] = []
    seen: set[str] = set()
    for item in [query, *rewritten_queries]:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append(text)
    return queries


def _merge_memories_by_id(memory_batches: list[list], *, limit: int) -> list:
    """按 memory_id 合并多路召回结果并保留首次出现顺序。

    Args:
        memory_batches: 多路检索返回的候选记忆列表。
        limit: 合并后的最大候选数。

    Returns:
        去重并截断后的候选记忆。
    """

    merged = []
    seen: set[str] = set()
    for memories in memory_batches:
        for memory in memories:
            if memory.memory_id in seen:
                continue
            seen.add(memory.memory_id)
            merged.append(memory)
            if len(merged) >= limit:
                return merged
    return merged


def _candidate_pool_limit(top_k: int) -> int:
    """计算进入 Answer Synthesizer 的候选池大小。

    Args:
        top_k: API 请求的引用规模提示。

    Returns:
        扩大后的候选池大小，避免在语义判别前过早截断。
    """

    return max(top_k, min(top_k * 3, 60))


def _audit_memory_item(memory) -> dict:
    """构造不超过 120 字的审计候选摘要。"""

    return {
        "memory_id": memory.memory_id,
        "occurred_at": memory.occurred_at.isoformat(),
        "content_preview": memory.content[:120],
    }


def _now_utc() -> datetime:
    """返回当前 UTC 时间。"""

    return datetime.now(tz=UTC)


def _coerce_retention_days(value: int | None) -> int:
    """将记忆保留天数限制在允许范围内。

    Args:
        value: 调用方传入的保留天数。

    Returns:
        应用于清理任务的保留天数。
    """

    try:
        parsed = int(value) if value is not None else 180
    except (TypeError, ValueError):
        parsed = 180
    return max(1, min(3650, parsed))


def _coerce_confidence(value) -> float:
    """将 LLM 返回的置信度转换为浮点数，非法值降级为默认置信度。

    Args:
        value: LLM 结构化输出中的置信度字段。

    Returns:
        可用于 API 响应的浮点置信度。
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.6


def _estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数，用于无真实 LLM 用量时的聚合统计。

    Args:
        text: 待估算的文本。

    Returns:
        至少为 1 的估算 token 数。
    """

    return max(1, len(text) // 2)
