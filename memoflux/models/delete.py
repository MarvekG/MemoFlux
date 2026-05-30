from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeleteResult:
    """删除结果。"""

    delete_id: str
    matched_memory_ids: list[str]
    affected_memory_ids: list[str]
    status: str
