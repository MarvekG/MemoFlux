from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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
    answer: str
    references: list[RecallReference] = field(default_factory=list)
    confidence: float = 0.0
    uncertainties: list[str] = field(default_factory=list)
