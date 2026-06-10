from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.schema import CreateSchema
from pgvector.sqlalchemy import Vector

from memoflux.models import DeleteAuditRecord, MemoryRecord, QueryAuditRecord, UsageStats
from memoflux.storage.schema import Base, DeleteAuditRow, MemoryRow, QueryAuditRow, UsageRunRow


class PostgresMemoryRepository:
    """使用 SQLAlchemy ORM 持久化 MemoFlux 数据。"""

    def __init__(self, *, database_url: str, schema: str = "memoflux", embedding_dimension: int = 768) -> None:
        self.schema = schema
        self.embedding_dimension = embedding_dimension
        self.engine: AsyncEngine = create_async_engine(database_url, future=True)

    async def init_schema(self) -> None:
        """异步初始化 MemoFlux 表结构。"""

        Base.metadata.schema = self.schema
        for table in Base.metadata.tables.values():
            table.schema = self.schema
        MemoryRow.__table__.c.embedding.type = Vector(self.embedding_dimension)
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(CreateSchema(self.schema, if_not_exists=True))
            await conn.run_sync(Base.metadata.create_all)

    async def insert_memory(self, *, session: str, content: str, occurred_at: datetime, embedding=None) -> MemoryRecord:
        """异步写入一条记忆。"""

        row = MemoryRow(
            memory_id=uuid4().hex,
            session=session,
            content=content,
            embedding=embedding,
            occurred_at=occurred_at,
            created_at=datetime.now(tz=UTC),
        )
        async with AsyncSession(self.engine, expire_on_commit=False) as db_session:
            db_session.add(row)
            await db_session.commit()
        return _memory_from_row(row)

    async def search_memories(self, *, session: str, terms: set[str], limit: int, query_embedding=None) -> list[MemoryRecord]:
        """异步按 session 召回候选。"""

        statement = select(MemoryRow).where(MemoryRow.session == session)
        if query_embedding:
            statement = statement.order_by(MemoryRow.embedding.l2_distance(query_embedding).asc())
        elif terms:
            predicates = [MemoryRow.content.ilike(f"%{term}%") for term in terms]
            statement = statement.where(or_(*predicates)).order_by(MemoryRow.occurred_at.asc())
        else:
            return []
        statement = statement.limit(limit)
        async with AsyncSession(self.engine) as db_session:
            rows = list((await db_session.scalars(statement)).all())
        return [_memory_from_row(row) for row in rows]

    async def delete_memories(self, *, session: str, memory_ids: list[str]) -> list[str]:
        """异步硬删除同一 session 内的记忆。"""

        if not memory_ids:
            return []
        select_statement = select(MemoryRow.memory_id).where(MemoryRow.session == session, MemoryRow.memory_id.in_(memory_ids))
        delete_statement = delete(MemoryRow).where(MemoryRow.session == session, MemoryRow.memory_id.in_(memory_ids))
        async with AsyncSession(self.engine) as db_session:
            matched = list((await db_session.scalars(select_statement)).all())
            await db_session.execute(delete_statement)
            await db_session.commit()
        return matched

    async def delete_memories_before_occurred_at(self, cutoff: datetime) -> dict[str, list[str]]:
        """异步按发生时间硬删除过期记忆并按 session 返回删除 ID。

        Args:
            cutoff: 过期判断截止时间，早于该时间的记忆会被删除。

        Returns:
            按 session 分组的已删除 memory_id 列表。
        """

        select_statement = select(MemoryRow.session, MemoryRow.memory_id).where(MemoryRow.occurred_at < cutoff)
        delete_statement = delete(MemoryRow).where(MemoryRow.occurred_at < cutoff)
        grouped: dict[str, list[str]] = {}
        async with AsyncSession(self.engine) as db_session:
            rows = (await db_session.execute(select_statement)).all()
            for memory_session, memory_id in rows:
                grouped.setdefault(str(memory_session), []).append(str(memory_id))
            if grouped:
                await db_session.execute(delete_statement)
            await db_session.commit()
        return grouped

    async def list_memories(self, *, session: str | None, limit: int, offset: int = 0) -> tuple[list[MemoryRecord], int]:
        """异步列出原始记忆，未指定 session 时返回所有 session。"""

        statement = select(MemoryRow).order_by(MemoryRow.occurred_at.asc()).offset(offset).limit(limit)
        count_statement = select(func.count(MemoryRow.memory_id))
        if session is not None:
            statement = statement.where(MemoryRow.session == session)
            count_statement = count_statement.where(MemoryRow.session == session)
        async with AsyncSession(self.engine) as db_session:
            rows = list((await db_session.scalars(statement)).all())
            total = int((await db_session.scalar(count_statement)) or 0)
        return [_memory_from_row(row) for row in rows], total

    async def record_usage(
        self,
        *,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        """异步记录聚合 LLM 用量。"""

        row = UsageRunRow(
            run_id=uuid4().hex,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            created_at=datetime.now(tz=UTC),
        )
        async with AsyncSession(self.engine) as db_session:
            db_session.add(row)
            await db_session.commit()

    async def usage_stats(self) -> UsageStats:
        """异步返回聚合用量统计。"""

        statement = (
            select(
                UsageRunRow.operation,
                func.count(UsageRunRow.run_id),
                func.sum(UsageRunRow.input_tokens),
                func.sum(UsageRunRow.output_tokens),
                func.sum(UsageRunRow.cached_tokens),
                func.sum(UsageRunRow.reasoning_tokens),
            )
            .group_by(UsageRunRow.operation)
        )
        async with AsyncSession(self.engine) as db_session:
            rows = (await db_session.execute(statement)).all()
        by_operation = {
            operation: {
                "calls": int(calls or 0),
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "cached_tokens": int(cached_tokens or 0),
                "cache_miss_tokens": max(int(input_tokens or 0) - int(cached_tokens or 0), 0),
                "reasoning_tokens": int(reasoning_tokens or 0),
            }
            for operation, calls, input_tokens, output_tokens, cached_tokens, reasoning_tokens in rows
        }
        total_input = sum(item["input_tokens"] for item in by_operation.values())
        total_output = sum(item["output_tokens"] for item in by_operation.values())
        total_cached = sum(item["cached_tokens"] for item in by_operation.values())
        total_reasoning = sum(item["reasoning_tokens"] for item in by_operation.values())
        total_calls = sum(item["calls"] for item in by_operation.values())
        return UsageStats(
            llm_runs=total_calls,
            total_calls=total_calls,
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cached_tokens=total_cached,
            cache_miss_tokens=max(total_input - total_cached, 0),
            reasoning_tokens=total_reasoning,
            cache_hit_rate=(total_cached / total_input) if total_input else 0.0,
            by_operation=by_operation,
        )

    async def clear_usage_stats(self) -> int:
        """异步清空聚合用量统计。"""

        async with AsyncSession(self.engine) as db_session:
            result = await db_session.execute(delete(UsageRunRow))
            await db_session.commit()
        return int(result.rowcount or 0)

    async def record_query_audit(self, audit: QueryAuditRecord) -> None:
        """异步记录召回查询审计。"""

        row = QueryAuditRow(**audit.__dict__)
        async with AsyncSession(self.engine) as db_session:
            db_session.add(row)
            await db_session.commit()

    async def record_delete_audit(self, audit: DeleteAuditRecord) -> None:
        """异步记录删除操作审计。"""

        row = DeleteAuditRow(**{**audit.__dict__, "dry_run": int(audit.dry_run)})
        async with AsyncSession(self.engine) as db_session:
            db_session.add(row)
            await db_session.commit()

    async def list_audits(
        self,
        *,
        session: str | None,
        limit: int,
        offset: int = 0,
        query_id: str | None = None,
    ) -> tuple[list[QueryAuditRecord | DeleteAuditRecord], int]:
        """异步列出查询和删除审计，未指定 session 时返回所有 session。"""

        async with AsyncSession(self.engine) as db_session:
            if query_id:
                row = await db_session.get(QueryAuditRow, query_id)
                items = [_query_audit_from_row(row)] if row and (session is None or row.session == session) else []
                return items[offset:offset + limit], len(items)
            fetch_limit = offset + limit
            query_statement = select(QueryAuditRow).order_by(QueryAuditRow.created_at.desc()).limit(fetch_limit)
            delete_statement = select(DeleteAuditRow).order_by(DeleteAuditRow.created_at.desc()).limit(fetch_limit)
            query_count_statement = select(func.count(QueryAuditRow.query_id))
            delete_count_statement = select(func.count(DeleteAuditRow.delete_id))
            if session is not None:
                query_statement = query_statement.where(QueryAuditRow.session == session)
                delete_statement = delete_statement.where(DeleteAuditRow.session == session)
                query_count_statement = query_count_statement.where(QueryAuditRow.session == session)
                delete_count_statement = delete_count_statement.where(DeleteAuditRow.session == session)
            query_rows = list((await db_session.scalars(query_statement)).all())
            delete_rows = list((await db_session.scalars(delete_statement)).all())
            total = int((await db_session.scalar(query_count_statement)) or 0) + int(
                (await db_session.scalar(delete_count_statement)) or 0
            )
        audits: list[QueryAuditRecord | DeleteAuditRecord] = [_query_audit_from_row(row) for row in query_rows]
        audits.extend(_delete_audit_from_row(row) for row in delete_rows)
        audits.sort(key=lambda audit: audit.created_at, reverse=True)
        return audits[offset:offset + limit], total

    async def scrub_deleted_memory_from_audits(self, *, session: str, memory_ids: list[str]) -> None:
        """异步从查询审计中清理已硬删除记忆的原文片段。"""

        if not memory_ids:
            return
        deleted = set(memory_ids)
        async with AsyncSession(self.engine) as db_session:
            rows = list((await db_session.scalars(select(QueryAuditRow).where(QueryAuditRow.session == session))).all())
            for row in rows:
                selected = set(row.selected_memory_ids or [])
                row.retrieved = [
                    {**item, "content_preview": None}
                    if item.get("memory_id") in deleted
                    else item
                    for item in row.retrieved
                ]
                row.selection_reasons = {
                    memory_id: reason
                    for memory_id, reason in dict(row.selection_reasons or {}).items()
                    if memory_id not in deleted
                }
                if selected & deleted:
                    row.final_answer = None
            await db_session.commit()


def _memory_from_row(row: MemoryRow) -> MemoryRecord:
    """将数据库行转换为领域记忆对象。

    Args:
        row: SQLAlchemy ORM 记忆行。

    Returns:
        可供服务层返回和检索使用的记忆记录。
    """

    return MemoryRecord(
        memory_id=row.memory_id,
        session=row.session,
        content=row.content,
        occurred_at=row.occurred_at,
        created_at=row.created_at,
    )


def _query_audit_from_row(row: QueryAuditRow) -> QueryAuditRecord:
    """将查询审计行转换为领域对象。"""

    return QueryAuditRecord(
        query_id=row.query_id,
        session=row.session,
        original_query=row.original_query,
        rewritten_queries=list(row.rewritten_queries or []),
        candidate_limit=row.candidate_limit,
        retrieved=list(row.retrieved or []),
        selected_memory_ids=list(row.selected_memory_ids or []),
        selection_reasons=dict(row.selection_reasons or {}),
        final_answer=row.final_answer,
        status=row.status,
        error_stage=row.error_stage,
        error_message=row.error_message,
        created_at=row.created_at,
    )


def _delete_audit_from_row(row: DeleteAuditRow) -> DeleteAuditRecord:
    """将删除审计行转换为领域对象。"""

    return DeleteAuditRecord(
        delete_id=row.delete_id,
        session=row.session,
        target=dict(row.target or {}),
        dry_run=bool(row.dry_run),
        matched_memory_ids=list(row.matched_memory_ids or []),
        affected_memory_ids=list(row.affected_memory_ids or []),
        status=row.status,
        error_code=row.error_code,
        error_message=row.error_message,
        created_at=row.created_at,
    )
