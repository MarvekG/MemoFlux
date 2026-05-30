from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, Index, Integer, MetaData, String, Text, create_engine, delete, func, or_, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.schema import CreateSchema
from pgvector.sqlalchemy import Vector

from memoflux.models import MemoryRecord, UsageStats


class Base(DeclarativeBase):
    """MemoFlux SQLAlchemy ORM 基类。"""

    metadata = MetaData(schema="memoflux")


class MemoryRow(Base):
    """原始记忆 ORM 映射。"""

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memoflux_memories_scope_occurred_at", "scope", "occurred_at"),
        Index("ix_memoflux_memories_scope_created_at", "scope", "created_at"),
    )

    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    scope: Mapped[str] = mapped_column(String, nullable=False)
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

    def insert_memory(self, *, scope: str, content: str, occurred_at: datetime, embedding=None) -> MemoryRecord:
        """写入一条记忆。"""

        row = MemoryRow(
            memory_id=uuid4().hex,
            scope=scope,
            content=content,
            embedding=embedding,
            occurred_at=occurred_at,
            created_at=datetime.now(tz=UTC),
        )
        with Session(self.engine, expire_on_commit=False) as session:
            session.add(row)
            session.commit()
        return _memory_from_row(row)

    def search_memories(self, *, scope: str, terms: set[str], limit: int, query_embedding=None) -> list[MemoryRecord]:
        """按 scope 和文本词召回候选。"""

        statement = select(MemoryRow).where(MemoryRow.scope == scope)
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

    def delete_memories(self, *, scope: str, memory_ids: list[str]) -> list[str]:
        """硬删除同一 scope 内的记忆。"""

        if not memory_ids:
            return []
        select_statement = select(MemoryRow.memory_id).where(MemoryRow.scope == scope, MemoryRow.memory_id.in_(memory_ids))
        delete_statement = delete(MemoryRow).where(MemoryRow.scope == scope, MemoryRow.memory_id.in_(memory_ids))
        with Session(self.engine) as session:
            matched = list(session.scalars(select_statement).all())
            session.execute(delete_statement)
            session.commit()
        return matched

    def list_memories(self, *, scope: str, limit: int) -> list[MemoryRecord]:
        """列出同一 scope 内的原始记忆。"""

        statement = select(MemoryRow).where(MemoryRow.scope == scope).order_by(MemoryRow.occurred_at.asc()).limit(limit)
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


def _memory_from_row(row: MemoryRow) -> MemoryRecord:
    """将数据库行转换为领域记忆对象。

    Args:
        row: SQLAlchemy ORM 记忆行。

    Returns:
        可供服务层返回和检索使用的记忆记录。
    """

    return MemoryRecord(
        memory_id=row.memory_id,
        scope=row.scope,
        content=row.content,
        occurred_at=row.occurred_at,
        created_at=row.created_at,
    )
