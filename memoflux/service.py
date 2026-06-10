from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any
from uuid import uuid4

from memoflux.i18n import i18n_service
from memoflux.llm import LLMResult, LocalLLMClient
from memoflux.models import DeleteAuditRecord, DeleteResult, QueryAuditRecord, RecallReference, RecallResult
from memoflux.services.embedding_service import embedding_service

NO_ANSWER_KEY = "answers.no_memory"
logger = logging.getLogger(__name__)


class MemoFluxService:
    """MemoFlux 业务服务。"""

    def __init__(self, repository, llm_client=None, embedding_client=None) -> None:
        self.repository = repository
        self.llm_client = llm_client or LocalLLMClient()
        self.embedding_service = embedding_client or embedding_service

    async def ingest(self, *, session: str, content: str, occurred_at: datetime):
        """异步写入一条记忆。

        Args:
            session: 记忆所属会话。
            content: 需要写入的记忆内容。
            occurred_at: 记忆发生时间。

        Returns:
            已写入的记忆记录。
        """

        if not session:
            raise ValueError("session is required")
        if not content:
            raise ValueError("content is required")
        if occurred_at is None:
            raise ValueError("occurred_at is required")
        embedding = LLMResult(output={"embedding": await self.embedding_service.embed_text(content)})
        await self.repository.record_usage(
            operation="embedding:ingest",
            input_tokens=embedding.input_tokens,
            output_tokens=embedding.output_tokens,
            cached_tokens=embedding.cached_tokens,
            reasoning_tokens=embedding.reasoning_tokens,
        )
        memory = await self.repository.insert_memory(
            session=session,
            content=content,
            occurred_at=occurred_at,
            embedding=embedding.output.get("embedding"),
        )
        logger.info(
            "MemoFlux ingest completed",
            extra={
                "memo_session": session,
                "memory_id": memory.memory_id,
                "content_length": len(content),
                "embedding_recorded": bool(embedding.output.get("embedding")),
            },
        )
        return memory

    async def recall(self, *, session: str, query: str, top_k: int = 12, lang: str | None = None) -> RecallResult:
        """异步召回同一 session 内的候选记忆并生成答案。

        Args:
            session: 记忆所属会话。
            query: 用户查询内容。
            top_k: 返回引用数量上限。
            lang: 本地化语言标识。

        Returns:
            召回答案、引用和置信度信息。
        """

        if not session:
            raise ValueError("session is required")
        if not query:
            raise ValueError("query is required")
        query_id = uuid4().hex
        recall_started = perf_counter()
        logger.info(
            "MemoFlux recall started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "query_length": len(query),
                "top_k": top_k,
            },
        )
        logger.info(
            "MemoFlux recall stage started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "query_planner",
            },
        )
        stage_started = perf_counter()
        plan = await self.llm_client.plan_query(query=query)
        planner_elapsed_ms = _elapsed_ms(stage_started)
        rewritten_queries = plan.output.get("rewritten_queries") or [query]
        retrieval_queries = _build_retrieval_queries(query, rewritten_queries)
        retrieval_query_lengths = _text_lengths_summary(retrieval_queries)
        candidate_limit = _candidate_pool_limit(top_k)
        logger.info(
            "MemoFlux recall stage completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "query_planner",
                "elapsed_ms": planner_elapsed_ms,
                "rewritten_query_count": len(rewritten_queries),
                "retrieval_query_count": len(retrieval_queries),
                "retrieval_query_length_min": retrieval_query_lengths["min"],
                "retrieval_query_length_max": retrieval_query_lengths["max"],
                "retrieval_query_length_avg": retrieval_query_lengths["avg"],
                "candidate_limit": candidate_limit,
                "input_tokens": plan.input_tokens,
                "output_tokens": plan.output_tokens,
                "cached_tokens": plan.cached_tokens,
                "reasoning_tokens": plan.reasoning_tokens,
            },
        )
        memory_batches = []
        embedding_metadata = _embedding_metadata(self.embedding_service)
        logger.info(
            "MemoFlux recall stage started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "embedding",
                "provider": embedding_metadata["provider"],
                "model": embedding_metadata["model"],
                "retrieval_query_count": len(retrieval_queries),
                "retrieval_query_length_min": retrieval_query_lengths["min"],
                "retrieval_query_length_max": retrieval_query_lengths["max"],
                "retrieval_query_length_avg": retrieval_query_lengths["avg"],
            },
        )
        stage_started = perf_counter()
        query_embeddings = await self.embedding_service.embed_texts(retrieval_queries)
        embedding_elapsed_ms = _elapsed_ms(stage_started)
        logger.info(
            "MemoFlux recall stage completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "embedding",
                "elapsed_ms": embedding_elapsed_ms,
                "retrieval_query_count": len(retrieval_queries),
                "embedding_count": len(query_embeddings),
            },
        )
        if len(query_embeddings) != len(retrieval_queries):
            raise ValueError("recall embedding count does not match retrieval query count")
        logger.info(
            "MemoFlux recall stage started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "vector_search",
                "embedding_count": len(query_embeddings),
                "candidate_limit": candidate_limit,
            },
        )
        stage_started = perf_counter()
        vector_batch_sizes: list[int] = []
        vector_batch_elapsed_ms: list[float] = []
        for index, embedding in enumerate(query_embeddings):
            query_embedding = LLMResult(output={"embedding": embedding})
            await self.repository.record_usage(
                operation="embedding:recall",
                input_tokens=query_embedding.input_tokens,
                output_tokens=query_embedding.output_tokens,
                cached_tokens=query_embedding.cached_tokens,
                reasoning_tokens=query_embedding.reasoning_tokens,
            )
            batch_started = perf_counter()
            vector_batch = await self.repository.search_memories(
                session=session,
                terms=set(),
                limit=candidate_limit,
                query_embedding=query_embedding.output.get("embedding"),
            )
            vector_batch_elapsed_ms.append(_elapsed_ms(batch_started))
            vector_batch_sizes.append(len(vector_batch))
            memory_batches.append(vector_batch)
            logger.info(
                "MemoFlux recall vector search batch completed",
                extra={
                    "memo_session": session,
                    "query_id": query_id,
                    "stage": "vector_search",
                    "batch_index": index,
                    "elapsed_ms": vector_batch_elapsed_ms[-1],
                    "candidate_count": len(vector_batch),
                    "candidate_limit": candidate_limit,
                },
            )
        vector_search_elapsed_ms = _elapsed_ms(stage_started)
        logger.info(
            "MemoFlux recall stage completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "vector_search",
                "elapsed_ms": vector_search_elapsed_ms,
                "batch_count": len(vector_batch_sizes),
                "batch_sizes": vector_batch_sizes,
                "batch_elapsed_ms": vector_batch_elapsed_ms,
                "candidate_limit": candidate_limit,
            },
        )
        logger.info(
            "MemoFlux recall stage started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "list_fallback",
                "candidate_limit": candidate_limit,
            },
        )
        stage_started = perf_counter()
        listed_memories, listed_total = await self.repository.list_memories(
            session=session,
            limit=candidate_limit,
        )
        list_elapsed_ms = _elapsed_ms(stage_started)
        logger.info(
            "MemoFlux recall stage completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "list_fallback",
                "elapsed_ms": list_elapsed_ms,
                "listed_count": len(listed_memories),
                "listed_total": listed_total,
                "candidate_limit": candidate_limit,
            },
        )
        memory_batches.append(listed_memories)
        logger.info(
            "MemoFlux recall stage started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "merge_candidates",
                "batch_count": len(memory_batches),
                "candidate_limit": candidate_limit,
            },
        )
        stage_started = perf_counter()
        memories = _merge_memories_by_id(memory_batches, limit=candidate_limit)
        merge_elapsed_ms = _elapsed_ms(stage_started)
        logger.info(
            "MemoFlux recall stage completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "merge_candidates",
                "elapsed_ms": merge_elapsed_ms,
                "batch_count": len(memory_batches),
                "candidate_count": len(memories),
                "candidate_limit": candidate_limit,
            },
        )
        await self.repository.record_usage(
            operation="query_planner",
            input_tokens=plan.input_tokens,
            output_tokens=plan.output_tokens,
            cached_tokens=plan.cached_tokens,
            reasoning_tokens=plan.reasoning_tokens,
        )
        if not memories:
            result = RecallResult(query_id=query_id, answer=i18n_service.t(NO_ANSWER_KEY, lang=lang), references=[])
            logger.info(
                "MemoFlux recall stage started",
                extra={
                    "memo_session": session,
                    "query_id": query_id,
                    "stage": "audit_write",
                    "candidate_count": 0,
                    "selected_count": 0,
                },
            )
            stage_started = perf_counter()
            await self.repository.record_query_audit(
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
            audit_elapsed_ms = _elapsed_ms(stage_started)
            final_stage_elapsed = {
                "query_planner": planner_elapsed_ms,
                "embedding": embedding_elapsed_ms,
                "vector_search": vector_search_elapsed_ms,
                "list_fallback": list_elapsed_ms,
                "merge_candidates": merge_elapsed_ms,
                "answer_synthesizer": 0.0,
                "audit_write": audit_elapsed_ms,
            }
            dominant_stage, dominant_elapsed_ms = _dominant_stage(final_stage_elapsed)
            logger.info(
                "MemoFlux recall stage completed",
                extra={
                    "memo_session": session,
                    "query_id": query_id,
                    "stage": "audit_write",
                    "elapsed_ms": audit_elapsed_ms,
                    "candidate_count": 0,
                    "selected_count": 0,
                },
            )
            logger.info(
                "MemoFlux recall completed",
                extra={
                    "memo_session": session,
                    "query_id": query_id,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "reference_count": 0,
                    "rewritten_query_count": len(rewritten_queries),
                    "retrieval_query_count": len(retrieval_queries),
                    "answerability": "no_memory",
                    "total_elapsed_ms": _elapsed_ms(recall_started),
                    "planner_elapsed_ms": planner_elapsed_ms,
                    "embedding_elapsed_ms": embedding_elapsed_ms,
                    "vector_search_elapsed_ms": vector_search_elapsed_ms,
                    "list_elapsed_ms": list_elapsed_ms,
                    "merge_elapsed_ms": merge_elapsed_ms,
                    "synthesis_elapsed_ms": 0.0,
                    "audit_elapsed_ms": audit_elapsed_ms,
                    "dominant_stage": dominant_stage,
                    "dominant_elapsed_ms": dominant_elapsed_ms,
                },
            )
            return result
        logger.info(
            "MemoFlux recall stage started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "answer_synthesizer",
                "candidate_count": len(memories),
                "synthesis_memory_count": min(len(memories), top_k),
            },
        )
        stage_started = perf_counter()
        synthesis = await self.llm_client.synthesize_answer(query=query, memories=memories[:top_k])
        synthesis_elapsed_ms = _elapsed_ms(stage_started)
        logger.info(
            "MemoFlux recall stage completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "answer_synthesizer",
                "elapsed_ms": synthesis_elapsed_ms,
                "candidate_count": len(memories),
                "synthesis_memory_count": min(len(memories), top_k),
                "input_tokens": synthesis.input_tokens,
                "output_tokens": synthesis.output_tokens,
                "cached_tokens": synthesis.cached_tokens,
                "reasoning_tokens": synthesis.reasoning_tokens,
            },
        )
        await self.repository.record_usage(
            operation="answer_synthesizer",
            input_tokens=synthesis.input_tokens,
            output_tokens=synthesis.output_tokens,
            cached_tokens=synthesis.cached_tokens,
            reasoning_tokens=synthesis.reasoning_tokens,
        )
        logger.info(
            "MemoFlux recall stage started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "postprocess",
                "candidate_count": len(memories),
            },
        )
        stage_started = perf_counter()
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
        postprocess_elapsed_ms = _elapsed_ms(stage_started)
        logger.info(
            "MemoFlux recall stage completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "postprocess",
                "elapsed_ms": postprocess_elapsed_ms,
                "candidate_count": len(memories),
                "used_memory_count": len(used_memories),
                "reference_count": len(references),
                "uncertainty_count": len(result.uncertainties),
            },
        )
        logger.info(
            "MemoFlux recall stage started",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "audit_write",
                "candidate_count": len(memories),
                "selected_count": len(used_memories),
            },
        )
        stage_started = perf_counter()
        await self.repository.record_query_audit(
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
        audit_elapsed_ms = _elapsed_ms(stage_started)
        final_stage_elapsed = {
            "query_planner": planner_elapsed_ms,
            "embedding": embedding_elapsed_ms,
            "vector_search": vector_search_elapsed_ms,
            "list_fallback": list_elapsed_ms,
            "merge_candidates": merge_elapsed_ms,
            "answer_synthesizer": synthesis_elapsed_ms,
            "postprocess": postprocess_elapsed_ms,
            "audit_write": audit_elapsed_ms,
        }
        dominant_stage, dominant_elapsed_ms = _dominant_stage(final_stage_elapsed)
        logger.info(
            "MemoFlux recall stage completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "stage": "audit_write",
                "elapsed_ms": audit_elapsed_ms,
                "candidate_count": len(memories),
                "selected_count": len(used_memories),
            },
        )
        logger.info(
            "MemoFlux recall completed",
            extra={
                "memo_session": session,
                "query_id": query_id,
                "candidate_count": len(memories),
                "selected_count": len(used_memories),
                "reference_count": len(references),
                "rewritten_query_count": len(rewritten_queries),
                "retrieval_query_count": len(retrieval_queries),
                "answerability": selected_reasons.get("__answerability"),
                "confidence": result.confidence,
                "total_elapsed_ms": _elapsed_ms(recall_started),
                "planner_elapsed_ms": planner_elapsed_ms,
                "embedding_elapsed_ms": embedding_elapsed_ms,
                "vector_search_elapsed_ms": vector_search_elapsed_ms,
                "list_elapsed_ms": list_elapsed_ms,
                "merge_elapsed_ms": merge_elapsed_ms,
                "synthesis_elapsed_ms": synthesis_elapsed_ms,
                "postprocess_elapsed_ms": postprocess_elapsed_ms,
                "audit_elapsed_ms": audit_elapsed_ms,
                "dominant_stage": dominant_stage,
                "dominant_elapsed_ms": dominant_elapsed_ms,
            },
        )
        return result

    async def delete(self, *, session: str, memory_ids: list[str] | None, query: str | None, dry_run: bool) -> DeleteResult:
        """异步删除记忆；query 模式只允许 dry-run。

        Args:
            session: 记忆所属会话。
            memory_ids: 待删除的记忆 ID 列表。
            query: query dry-run 删除条件。
            dry_run: 是否仅预览删除结果。

        Returns:
            删除匹配、影响范围和状态。

        Raises:
            ValueError: 入参缺失或 query 删除未使用 dry-run。
        """

        if not session:
            raise ValueError("session is required")
        if query and not dry_run:
            raise ValueError("query delete requires dry_run=true")
        if not memory_ids and not query:
            raise ValueError("memory_ids or query is required")
        delete_id = uuid4().hex
        if query:
            query_embedding = LLMResult(output={"embedding": await self.embedding_service.embed_text(query)})
            await self.repository.record_usage(
                operation="embedding:delete_dry_run",
                input_tokens=query_embedding.input_tokens,
                output_tokens=query_embedding.output_tokens,
                cached_tokens=query_embedding.cached_tokens,
                reasoning_tokens=query_embedding.reasoning_tokens,
            )
            candidates = await self.repository.search_memories(
                session=session,
                terms=set(),
                limit=12,
                query_embedding=query_embedding.output.get("embedding"),
            )
            matched = [memory.memory_id for memory in candidates]
            await self.repository.record_usage(
                operation="delete_dry_run",
                input_tokens=_estimate_tokens(query),
                output_tokens=0,
            )
            await self.repository.record_delete_audit(
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
            logger.info(
                "MemoFlux delete completed",
                extra={
                    "memo_session": session,
                    "delete_id": delete_id,
                    "mode": "query_dry_run",
                    "dry_run": True,
                    "matched_count": len(matched),
                    "affected_count": 0,
                },
            )
            return DeleteResult(delete_id=delete_id, matched_memory_ids=matched, affected_memory_ids=[], status="ok")
        matched = await self.repository.delete_memories(session=session, memory_ids=memory_ids or [])
        await self.repository.scrub_deleted_memory_from_audits(session=session, memory_ids=matched)
        await self.repository.record_delete_audit(
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
        logger.info(
            "MemoFlux delete completed",
            extra={
                "memo_session": session,
                "delete_id": delete_id,
                "mode": "memory_ids",
                "dry_run": dry_run,
                "target_count": len(memory_ids or []),
                "matched_count": len(matched),
                "affected_count": len(matched),
            },
        )
        return DeleteResult(delete_id=delete_id, matched_memory_ids=matched, affected_memory_ids=matched, status="ok")

    async def preview(self, *, session: str | None = None, limit: int = 50, offset: int = 0):
        """异步预览原始记忆。

        Args:
            session: 可选的会话过滤条件。
            limit: 返回数量上限。
            offset: 分页偏移量。

        Returns:
            记忆列表和总数。
        """

        return await self.repository.list_memories(session=session, limit=limit, offset=offset)

    async def usage_stats(self):
        """异步返回聚合用量统计。

        Returns:
            聚合后的 LLM/Embedding 用量统计。
        """

        stats = await self.repository.usage_stats()
        logger.info(
            "MemoFlux usage stats returned",
            extra={
                "total_calls": stats.total_calls,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "cached_tokens": stats.cached_tokens,
                "cache_hit_rate": stats.cache_hit_rate,
                "operation_count": len(stats.by_operation),
            },
        )
        return stats

    async def clear_usage_stats(self) -> int:
        """异步清空聚合用量统计。

        Returns:
            已删除的用量记录数量。
        """

        deleted = await self.repository.clear_usage_stats()
        logger.info("MemoFlux usage stats cleared", extra={"deleted_count": deleted})
        return deleted

    async def cleanup_expired_memories(self, *, retention_days: int | None = None) -> dict[str, Any]:
        """异步按发生时间清理超过保留期的记忆。

        Args:
            retention_days: 记忆保留天数；为空时使用默认保留期。

        Returns:
            清理结果摘要，包含删除数量、session 数、保留天数和截止时间。
        """

        normalized_retention_days = _coerce_retention_days(retention_days)
        cutoff = _now_utc() - timedelta(days=normalized_retention_days)
        deleted_by_session = await self.repository.delete_memories_before_occurred_at(cutoff)
        deleted_total = 0
        for session, memory_ids in deleted_by_session.items():
            if not memory_ids:
                continue
            deleted_total += len(memory_ids)
            await self.repository.scrub_deleted_memory_from_audits(session=session, memory_ids=memory_ids)
            await self.repository.record_delete_audit(
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


def _elapsed_ms(started: float) -> float:
    """计算从指定时间点到当前的毫秒耗时。

    Args:
        started: `perf_counter()` 返回的起始时间。

    Returns:
        保留两位小数的耗时毫秒数。
    """

    return round((perf_counter() - started) * 1000, 2)


def _text_lengths_summary(texts: list[str]) -> dict[str, float | int]:
    """统计文本长度分布，避免在日志中输出原文。

    Args:
        texts: 待统计的文本列表。

    Returns:
        文本长度的最小值、最大值和平均值摘要。
    """

    lengths = [len(text) for text in texts]
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0.0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
    }


def _embedding_metadata(embedding_client) -> dict[str, str]:
    """提取向量客户端元信息，兼容测试 fake 和外部实现。

    Args:
        embedding_client: 提供 `embed_texts` 能力的向量客户端实例。

    Returns:
        可安全写入日志的 provider/model 摘要。
    """

    return {
        "provider": str(getattr(embedding_client, "provider", "unknown")),
        "model": str(getattr(embedding_client, "model", "unknown")),
    }


def _dominant_stage(stage_elapsed_ms: dict[str, float]) -> tuple[str, float]:
    """从阶段耗时中找出最慢阶段。

    Args:
        stage_elapsed_ms: 阶段名称到耗时毫秒数的映射。

    Returns:
        最慢阶段名称和耗时毫秒数。
    """

    if not stage_elapsed_ms:
        return "unknown", 0.0
    stage, elapsed_ms = max(stage_elapsed_ms.items(), key=lambda item: item[1])
    return stage, elapsed_ms


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
