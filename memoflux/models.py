from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class MemoryRecord:
    """原始记忆记录。"""

    memory_id: str
    scope: str
    content: str
    occurred_at: datetime
    created_at: datetime


@dataclass(frozen=True)
class RecallReference:
    """召回答案引用。"""

    memory_id: str
    occurred_at: datetime
    quote: str
    relevance: str


@dataclass(frozen=True)
class RecallResult:
    """召回答案。"""

    query_id: str
    query_type: str
    answer: str
    references: list[RecallReference] = field(default_factory=list)
    confidence: float = 0.0
    uncertainties: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DeleteResult:
    """删除结果。"""

    delete_id: str
    matched_memory_ids: list[str]
    affected_memory_ids: list[str]
    status: str


@dataclass(frozen=True)
class UsageStats:
    """聚合用量统计。"""

    llm_runs: int
    total_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_miss_tokens: int
    reasoning_tokens: int
    cache_hit_rate: float
    by_operation: dict[str, dict[str, int]]
