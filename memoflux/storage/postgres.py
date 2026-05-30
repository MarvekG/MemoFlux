from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, Index, Integer, JSON, MetaData, String, Text, create_engine, delete, func, or_, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.schema import CreateSchema
from pgvector.sqlalchemy import Vector

from memoflux.models import DeleteAuditRecord, MemoryRecord, QueryAuditRecord, UsageStats


class Base(DeclarativeBase):
    """MemoFlux SQLAlchemy ORM 基类。"""

    metadata = MetaData(schema="memoflux")


class MemoryRow(Base):
    """原始记忆 ORM 映射。"""

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memoflux_memories_session_occurred_at", "session", "occurred_at"),
        Index("ix_memoflux_memories_session_created_at", "session", "created_at"),
    )

    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    session: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class UsageRunRow(Base):
    """LLM 用量 ORM 映射。"""

    __tablename__ = "usage_runs"

    __table_args__ = (Index("ix_memoflux_usage_runs_operation_created_at", "operation", "created_at"),)

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    operation: Mapped[str] = mapped_column(String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QueryAuditRow(Base):
    """召回查询审计 ORM 映射。"""

    __tablename__ = "memory_query_audits"
    __table_args__ = (Index("ix_memoflux_query_audits_session_created_at", "session", "created_at"),)

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    session: Mapped[str] = mapped_column(String, nullable=False)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String, nullable=False)
    rewritten_queries: Mapped[list] = mapped_column(JSON, nullable=False)
    candidate_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieved: Mapped[list] = mapped_column(JSON, nullable=False)
    selected_memory_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_stage: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DeleteAuditRow(Base):
    """删除操作审计 ORM 映射。"""

    __tablename__ = "memory_delete_audits"
    __table_args__ = (Index("ix_memoflux_delete_audits_session_created_at", "session", "created_at"),)

    delete_id: Mapped[str] = mapped_column(String, primary_key=True)
    session: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[dict] = mapped_column(JSON, nullable=False)
    dry_run: Mapped[int] = mapped_column(Integer, nullable=False)
    matched_memory_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    affected_memory_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PostgresMemoryRepository:
    """使用 SQLAlchemy ORM 持久化 MemoFlux 数据。"""

    def __init__(self, *, database_url: str, schema: str = "memoflux", embedding_dimension: int = 768) -> None:
        self.schema = schema
        self.embedding_dimension = embedding_dimension
        self.engine: Engine = create_engine(database_url, future=True)
        self.init_schema()

    def init_schema(self) -> None:
        """初始化 MemoFlux 表结构。"""

        Base.metadata.schema = self.schema
        for table in Base.metadata.tables.values():
            table.schema = self.schema
        MemoryRow.__table__.c.embedding.type = Vector(self.embedding_dimension)
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(CreateSchema(self.schema, if_not_exists=True))
        Base.metadata.create_all(bind=self.engine)

    def insert_memory(self, *, session: str, content: str, occurred_at: datetime, embedding=None) -> MemoryRecord:
        """写入一条记忆。"""

        row = MemoryRow(
            memory_id=uuid4().hex,
            session=session,
            content=content,
            embedding=embedding,
            occurred_at=occurred_at,
            created_at=datetime.now(tz=UTC),
        )
        with Session(self.engine, expire_on_commit=False) as session:
            session.add(row)
            session.commit()
        return _memory_from_row(row)

    def search_memories(self, *, session: str, terms: set[str], limit: int, query_embedding=None) -> list[MemoryRecord]:
        """按 session 召回候选。"""

        statement = select(MemoryRow).where(MemoryRow.session == session)
        if query_embedding:
            statement = statement.order_by(MemoryRow.embedding.l2_distance(query_embedding).asc())
        elif terms:
            predicates = [MemoryRow.content.ilike(f"%{term}%") for term in terms]
            statement = statement.where(or_(*predicates)).order_by(MemoryRow.occurred_at.asc())
        else:
            return []
        statement = statement.limit(limit)
        with Session(self.engine) as session:
            rows = session.scalars(statement).all()
        return [_memory_from_row(row) for row in rows]

    def delete_memories(self, *, session: str, memory_ids: list[str]) -> list[str]:
        """硬删除同一 session 内的记忆。"""

        if not memory_ids:
            return []
        select_statement = select(MemoryRow.memory_id).where(MemoryRow.session == session, MemoryRow.memory_id.in_(memory_ids))
        delete_statement = delete(MemoryRow).where(MemoryRow.session == session, MemoryRow.memory_id.in_(memory_ids))
        with Session(self.engine) as session:
            matched = list(session.scalars(select_statement).all())
            session.execute(delete_statement)
            session.commit()
        return matched

    def list_memories(self, *, session: str, limit: int) -> list[MemoryRecord]:
        """列出同一 session 内的原始记忆。"""

        statement = select(MemoryRow).where(MemoryRow.session == session).order_by(MemoryRow.occurred_at.asc()).limit(limit)
        with Session(self.engine) as session:
            rows = session.scalars(statement).all()
        return [_memory_from_row(row) for row in rows]

    def record_usage(self, *, operation: str, input_tokens: int, output_tokens: int) -> None:
        """记录聚合 LLM 用量。"""

        row = UsageRunRow(
            run_id=uuid4().hex,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            created_at=datetime.now(tz=UTC),
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def usage_stats(self) -> UsageStats:
        """返回聚合用量统计。"""

        statement = (
            select(
                UsageRunRow.operation,
                func.count(UsageRunRow.run_id),
                func.sum(UsageRunRow.input_tokens),
                func.sum(UsageRunRow.output_tokens),
            )
            .group_by(UsageRunRow.operation)
        )
        with Session(self.engine) as session:
            rows = session.execute(statement).all()
        by_operation = {
            operation: {
                "calls": int(calls or 0),
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
            }
            for operation, calls, input_tokens, output_tokens in rows
        }
        total_input = sum(item["input_tokens"] for item in by_operation.values())
        total_output = sum(item["output_tokens"] for item in by_operation.values())
        total_calls = sum(item["calls"] for item in by_operation.values())
        return UsageStats(
            llm_runs=total_calls,
            total_calls=total_calls,
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cached_tokens=0,
            cache_miss_tokens=total_input,
            reasoning_tokens=0,
            cache_hit_rate=0.0,
            by_operation=by_operation,
        )

    def clear_usage_stats(self) -> int:
        """清空聚合用量统计。"""

        with Session(self.engine) as session:
            result = session.execute(delete(UsageRunRow))
            session.commit()
        return int(result.rowcount or 0)

    def record_query_audit(self, audit: QueryAuditRecord) -> None:
        """记录召回查询审计。"""

        row = QueryAuditRow(**audit.__dict__)
        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def record_delete_audit(self, audit: DeleteAuditRecord) -> None:
        """记录删除操作审计。"""

        row = DeleteAuditRow(**{**audit.__dict__, "dry_run": int(audit.dry_run)})
        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def list_audits(
        self,
        *,
        session: str,
        limit: int,
        query_id: str | None = None,
    ) -> list[QueryAuditRecord | DeleteAuditRecord]:
        """列出同一 session 内的查询和删除审计。"""

        with Session(self.engine) as db_session:
            if query_id:
                row = db_session.get(QueryAuditRow, query_id)
                return [_query_audit_from_row(row)] if row and row.session == session else []
            query_rows = list(
                db_session.scalars(
                    select(QueryAuditRow)
                    .where(QueryAuditRow.session == session)
                    .order_by(QueryAuditRow.created_at.desc())
                    .limit(limit)
                ).all()
            )
            delete_rows = list(
                db_session.scalars(
                    select(DeleteAuditRow)
                    .where(DeleteAuditRow.session == session)
                    .order_by(DeleteAuditRow.created_at.desc())
                    .limit(limit)
                ).all()
            )
        audits: list[QueryAuditRecord | DeleteAuditRecord] = [_query_audit_from_row(row) for row in query_rows]
        audits.extend(_delete_audit_from_row(row) for row in delete_rows)
        audits.sort(key=lambda audit: audit.created_at, reverse=True)
        return audits[:limit]

    def scrub_deleted_memory_from_audits(self, *, session: str, memory_ids: list[str]) -> None:
        """从查询审计中清理已硬删除记忆的原文片段。"""

        if not memory_ids:
            return
        deleted = set(memory_ids)
        with Session(self.engine) as db_session:
            rows = list(db_session.scalars(select(QueryAuditRow).where(QueryAuditRow.session == session)).all())
            for row in rows:
                selected = set(row.selected_memory_ids or [])
                row.retrieved = [
                    {**item, "content_preview": None}
                    if item.get("memory_id") in deleted
                    else item
                    for item in row.retrieved
                ]
                if selected & deleted:
                    row.final_answer = None
            db_session.commit()


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
        query_type=row.query_type,
        rewritten_queries=list(row.rewritten_queries or []),
        candidate_limit=row.candidate_limit,
        retrieved=list(row.retrieved or []),
        selected_memory_ids=list(row.selected_memory_ids or []),
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
