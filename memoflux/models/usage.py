from __future__ import annotations

from dataclasses import dataclass


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
    by_operation: dict[str, dict[str, int | float]]
