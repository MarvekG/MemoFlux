from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, JSON, MetaData, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
    cached_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QueryAuditRow(Base):
    """召回查询审计 ORM 映射。"""

    __tablename__ = "memory_query_audits"
    __table_args__ = (Index("ix_memoflux_query_audits_session_created_at", "session", "created_at"),)

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    session: Mapped[str] = mapped_column(String, nullable=False)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_queries: Mapped[list] = mapped_column(JSON, nullable=False)
    candidate_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieved: Mapped[list] = mapped_column(JSON, nullable=False)
    selected_memory_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    selection_reasons: Mapped[dict] = mapped_column(JSON, nullable=False)
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
