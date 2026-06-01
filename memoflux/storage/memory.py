from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from memoflux.models import DeleteAuditRecord, MemoryRecord, QueryAuditRecord, UsageStats


class MemoryRepository:
    """测试与本地开发使用的内存 repository。"""

    def __init__(self) -> None:
        self._memories: dict[str, MemoryRecord] = {}
        self._usage: list[tuple[str, int, int, int, int]] = []
        self._query_audits: dict[str, QueryAuditRecord] = {}
        self._delete_audits: dict[str, DeleteAuditRecord] = {}

    def insert_memory(self, *, session: str, content: str, occurred_at: datetime, embedding=None) -> MemoryRecord:
        """写入一条记忆。"""

        memory = MemoryRecord(
            memory_id=uuid4().hex,
            session=session,
            content=content,
            occurred_at=occurred_at,
            created_at=datetime.now(tz=UTC),
        )
        self._memories[memory.memory_id] = memory
        return memory

    def search_memories(self, *, session: str, terms: set[str], limit: int, query_embedding=None) -> list[MemoryRecord]:
        """按 session 召回候选。"""

        memories = [memory for memory in self._memories.values() if memory.session == session]
        memories.sort(key=lambda memory: memory.occurred_at)
        return memories[:limit]

    def delete_memories(self, *, session: str, memory_ids: list[str]) -> list[str]:
        """硬删除同一 session 内的记忆。"""

        matched = [
            memory_id
            for memory_id in memory_ids
            if memory_id in self._memories and self._memories[memory_id].session == session
        ]
        for memory_id in matched:
            del self._memories[memory_id]
        return matched

    def delete_memories_before_occurred_at(self, cutoff) -> dict[str, list[str]]:
        """按发生时间硬删除过期记忆并按 session 返回删除 ID。

        Args:
            cutoff: 过期判断截止时间，早于该时间的记忆会被删除。

        Returns:
            按 session 分组的已删除 memory_id 列表。
        """

        grouped: dict[str, list[str]] = {}
        for memory_id, memory in list(self._memories.items()):
            if memory.occurred_at < cutoff:
                grouped.setdefault(memory.session, []).append(memory_id)
        for memory_ids in grouped.values():
            for memory_id in memory_ids:
                del self._memories[memory_id]
        return grouped

    def list_memories(self, *, session: str, limit: int) -> list[MemoryRecord]:
        """列出同一 session 内的原始记忆。"""

        memories = [memory for memory in self._memories.values() if memory.session == session]
        memories.sort(key=lambda memory: memory.occurred_at)
        return memories[:limit]

    def record_usage(
        self,
        *,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        """记录聚合 LLM 用量。"""

        self._usage.append((operation, input_tokens, output_tokens, cached_tokens, reasoning_tokens))

    def usage_stats(self) -> UsageStats:
        """返回聚合用量统计。"""

        by_operation: dict[str, dict[str, int]] = {}
        for operation, input_tokens, output_tokens, cached_tokens, reasoning_tokens in self._usage:
            item = by_operation.setdefault(
                operation,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cache_miss_tokens": 0, "reasoning_tokens": 0},
            )
            item["calls"] += 1
            item["input_tokens"] += input_tokens
            item["output_tokens"] += output_tokens
            item["cached_tokens"] += cached_tokens
            item["cache_miss_tokens"] += max(input_tokens - cached_tokens, 0)
            item["reasoning_tokens"] += reasoning_tokens
        total_input = sum(item[1] for item in self._usage)
        total_output = sum(item[2] for item in self._usage)
        total_cached = sum(item[3] for item in self._usage)
        total_reasoning = sum(item[4] for item in self._usage)
        return UsageStats(
            llm_runs=len(self._usage),
            total_calls=len(self._usage),
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cached_tokens=total_cached,
            cache_miss_tokens=max(total_input - total_cached, 0),
            reasoning_tokens=total_reasoning,
            cache_hit_rate=(total_cached / total_input) if total_input else 0.0,
            by_operation=by_operation,
        )

    def clear_usage_stats(self) -> int:
        """清空聚合用量统计。"""

        deleted = len(self._usage)
        self._usage.clear()
        return deleted

    def record_query_audit(self, audit: QueryAuditRecord) -> None:
        """记录召回查询审计。"""

        self._query_audits[audit.query_id] = audit

    def record_delete_audit(self, audit: DeleteAuditRecord) -> None:
        """记录删除操作审计。"""

        self._delete_audits[audit.delete_id] = audit

    def list_audits(
        self,
        *,
        session: str,
        limit: int,
        query_id: str | None = None,
    ) -> list[QueryAuditRecord | DeleteAuditRecord]:
        """列出同一 session 内的查询和删除审计。"""

        if query_id:
            audit = self._query_audits.get(query_id)
            return [audit] if audit and audit.session == session else []
        audits: list[QueryAuditRecord | DeleteAuditRecord] = [
            audit for audit in self._query_audits.values() if audit.session == session
        ]
        audits.extend(audit for audit in self._delete_audits.values() if audit.session == session)
        audits.sort(key=lambda audit: audit.created_at, reverse=True)
        return audits[:limit]

    def scrub_deleted_memory_from_audits(self, *, session: str, memory_ids: list[str]) -> None:
        """从审计记录中清理已硬删除记忆的原文片段。"""

        deleted = set(memory_ids)
        for query_id, audit in list(self._query_audits.items()):
            if audit.session != session:
                continue
            retrieved = [
                {**item, "content_preview": None}
                if item.get("memory_id") in deleted
                else item
                for item in audit.retrieved
            ]
            final_answer = None if any(memory_id in audit.selected_memory_ids for memory_id in deleted) else audit.final_answer
            self._query_audits[query_id] = QueryAuditRecord(
                query_id=audit.query_id,
                session=audit.session,
                original_query=audit.original_query,
                rewritten_queries=audit.rewritten_queries,
                candidate_limit=audit.candidate_limit,
                retrieved=retrieved,
                selected_memory_ids=audit.selected_memory_ids,
                selection_reasons={
                    memory_id: reason
                    for memory_id, reason in audit.selection_reasons.items()
                    if memory_id not in deleted
                },
                final_answer=final_answer,
                status=audit.status,
                error_stage=audit.error_stage,
                error_message=audit.error_message,
                created_at=audit.created_at,
            )
