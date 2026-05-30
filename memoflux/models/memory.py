from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MemoryRecord:
    """原始记忆记录。"""

    memory_id: str
    session: str
    content: str
    occurred_at: datetime
    created_at: datetime
