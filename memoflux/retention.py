from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from memoflux.config import MemoFluxSettings
from memoflux.service import MemoFluxService


def start_memory_retention_cleanup(*, service: MemoFluxService, settings: MemoFluxSettings) -> asyncio.Task:
    """启动每日记忆保留期清理后台任务。

    Args:
        service: MemoFlux 业务服务实例。
        settings: 已加载并完成配置校验的 MemoFlux 设置。

    Returns:
        已创建的 asyncio 后台任务。
    """

    return asyncio.create_task(
        _memory_retention_cleanup_loop(
            service,
            retention_days=settings.memory_retention_days,
            schedule_hour=settings.memory_cleanup_hour,
            schedule_minute=settings.memory_cleanup_minute,
        )
    )


async def _memory_retention_cleanup_loop(
    service: MemoFluxService,
    *,
    retention_days: int,
    schedule_hour: int,
    schedule_minute: int,
) -> None:
    """每天按配置时间执行 MemoFlux 记忆保留期清理。

    Args:
        service: MemoFlux 业务服务实例。
        retention_days: 记忆保留天数。
        schedule_hour: 每日执行小时。
        schedule_minute: 每日执行分钟。
    """

    while True:
        await asyncio.sleep(_seconds_until_next_run(schedule_hour=schedule_hour, schedule_minute=schedule_minute))
        try:
            await service.cleanup_expired_memories(retention_days=retention_days)
        except asyncio.CancelledError:
            raise
        except Exception:
            continue


def _seconds_until_next_run(*, schedule_hour: int, schedule_minute: int, now: datetime | None = None) -> float:
    """计算距离下一次每日清理时间的秒数。

    Args:
        schedule_hour: 每日执行小时。
        schedule_minute: 每日执行分钟。
        now: 当前时间；测试可传入固定时间。

    Returns:
        距离下一次执行的秒数。
    """

    current = now or datetime.now(tz=UTC)
    next_run = current.replace(hour=schedule_hour, minute=schedule_minute, second=0, microsecond=0)
    if next_run <= current:
        next_run += timedelta(days=1)
    return (next_run - current).total_seconds()
