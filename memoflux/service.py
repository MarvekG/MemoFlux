from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from memoflux.llm import LLMResult, LocalLLMClient
from memoflux.models import DeleteAuditRecord, DeleteResult, QueryAuditRecord, RecallReference, RecallResult
from memoflux.services.embedding_service import embedding_service

NO_ANSWER = "未找到可用于回答该问题的记忆。"


class MemoFluxService:
    """MemoFlux 业务服务。"""

    def __init__(self, repository, llm_client=None, embedding_client=None) -> None:
        self.repository = repository
        self.llm_client = llm_client or LocalLLMClient()
        self.embedding_service = embedding_client or embedding_service

    def ingest(self, *, scope: str, content: str, occurred_at: datetime):
        """写入一条记忆。"""

        if not scope:
            raise ValueError("scope is required")
        if not content:
            raise ValueError("content is required")
        if occurred_at is None:
            raise ValueError("occurred_at is required")
        embedding = LLMResult(output={"embedding": self.embedding_service.embed_text(content)})
        self.repository.record_usage(
            operation="embedding:ingest",
            input_tokens=embedding.input_tokens,
            output_tokens=embedding.output_tokens,
        )
        return self.repository.insert_memory(
            scope=scope,
            content=content,
            occurred_at=occurred_at,
            embedding=embedding.output.get("embedding"),
        )

    def recall(self, *, scope: str, query: str, top_k: int = 12) -> RecallResult:
        """召回同一 scope 内的候选记忆并生成答案。"""

        if not scope:
            raise ValueError("scope is required")
        if not query:
            raise ValueError("query is required")
        query_id = uuid4().hex
        plan = self.llm_client.plan_query(query=query)
        rewritten_queries = plan.output.get("rewritten_queries") or [query]
        query_embedding = LLMResult(output={"embedding": self.embedding_service.embed_text(query)})
        self.repository.record_usage(
            operation="embedding:recall",
            input_tokens=query_embedding.input_tokens,
            output_tokens=query_embedding.output_tokens,
        )
        memories = self.repository.search_memories(
            scope=scope,
            terms=set(),
            limit=top_k,
            query_embedding=query_embedding.output.get("embedding"),
        )
        query_type = str(plan.output.get("query_type") or "direct")
        if query_type in {"history", "temporal", "temporal_summary"}:
            memories.sort(key=lambda memory: memory.occurred_at)
        self.repository.record_usage(operation="query_planner", input_tokens=plan.input_tokens, output_tokens=plan.output_tokens)
        if not memories:
            result = RecallResult(query_id=query_id, query_type="direct", answer=NO_ANSWER, references=[])
            self.repository.record_query_audit(
                QueryAuditRecord(
                    query_id=query_id,
                    scope=scope,
                    original_query=query,
                    query_type="direct",
                    rewritten_queries=[str(item) for item in rewritten_queries],
                    candidate_limit=top_k,
                    retrieved=[],
                    selected_memory_ids=[],
                    final_answer=result.answer,
                    status="ok",
                    error_stage=None,
                    error_message=None,
                    created_at=_now_utc(),
                )
            )
            return result
        synthesis = self.llm_client.synthesize_answer(query=query, memories=memories)
        self.repository.record_usage(
            operation="answer_synthesizer",
            input_tokens=synthesis.input_tokens,
            output_tokens=synthesis.output_tokens,
        )
        references = [
            RecallReference(
                memory_id=memory.memory_id,
                occurred_at=memory.occurred_at,
                quote=memory.content,
                relevance="文本/时间候选匹配",
            )
            for memory in memories
        ]
        result = RecallResult(
            query_id=query_id,
            query_type=query_type,
            answer=str(synthesis.output.get("answer") or "\n".join(memory.content for memory in memories)),
            references=references,
            confidence=_coerce_confidence(synthesis.output.get("confidence")),
            uncertainties=[],
        )
        self.repository.record_query_audit(
            QueryAuditRecord(
                query_id=query_id,
                scope=scope,
                original_query=query,
                query_type=result.query_type,
                rewritten_queries=[str(item) for item in rewritten_queries],
                candidate_limit=top_k,
                retrieved=[_audit_memory_item(memory) for memory in memories],
                selected_memory_ids=[memory.memory_id for memory in memories],
                final_answer=result.answer,
                status="ok",
                error_stage=None,
                error_message=None,
                created_at=_now_utc(),
            )
        )
        return result

    def delete(self, *, scope: str, memory_ids: list[str] | None, query: str | None, dry_run: bool) -> DeleteResult:
        """删除记忆；query 模式只允许 dry-run。"""

        if not scope:
            raise ValueError("scope is required")
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
            )
            candidates = self.repository.search_memories(
                scope=scope,
                terms=set(),
                limit=12,
                query_embedding=query_embedding.output.get("embedding"),
            )
            matched = [memory.memory_id for memory in candidates]
            self.repository.record_usage(operation="delete_dry_run", input_tokens=_estimate_tokens(query), output_tokens=0)
            self.repository.record_delete_audit(
                DeleteAuditRecord(
                    delete_id=delete_id,
                    scope=scope,
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
        matched = self.repository.delete_memories(scope=scope, memory_ids=memory_ids or [])
        self.repository.scrub_deleted_memory_from_audits(scope=scope, memory_ids=matched)
        self.repository.record_delete_audit(
            DeleteAuditRecord(
                delete_id=delete_id,
                scope=scope,
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

    def preview(self, *, scope: str, limit: int = 50):
        """预览同一 scope 内的原始记忆。"""

        return self.repository.list_memories(scope=scope, limit=limit)

    def usage_stats(self):
        """返回聚合用量统计。"""

        return self.repository.usage_stats()

    def clear_usage_stats(self) -> int:
        """清空聚合用量统计。"""

        return self.repository.clear_usage_stats()


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
