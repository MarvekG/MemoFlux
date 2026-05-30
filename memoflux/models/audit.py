from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class QueryAuditRecord:
    """召回查询审计记录。"""

    query_id: str
    session: str
    original_query: str
    query_type: str
    rewritten_queries: list[str]
    candidate_limit: int
    retrieved: list[dict]
    selected_memory_ids: list[str]
    selection_reasons: dict[str, str]
    final_answer: str | None
    status: str
    error_stage: str | None
    error_message: str | None
    created_at: datetime


@dataclass(frozen=True)
class DeleteAuditRecord:
    """删除操作审计记录。"""

    delete_id: str
    session: str
    target: dict
    dry_run: bool
    matched_memory_ids: list[str]
    affected_memory_ids: list[str]
    status: str
    error_code: str | None
    error_message: str | None
    created_at: datetime
