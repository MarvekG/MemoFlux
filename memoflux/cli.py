from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from memoflux.config import load_settings

MEMOFLUX_PACKAGE_ROOT = Path(__file__).resolve().parent


def _resolve_reload_dirs() -> list[str]:
    """查找源码运行时可监听的自动重载目录。"""

    candidates = [Path.cwd() / "memoflux"]
    return [str(candidate) for candidate in candidates if candidate.exists()]


def run_server() -> None:
    """按 MemoFlux 配置启动 uvicorn 服务。"""

    settings = load_settings()
    Path(os.getenv("MEMOFLUX_RUNTIME_DIR", "/home/memo/.memoflux/data"), "logs").mkdir(parents=True, exist_ok=True)
    uvicorn_options = {
        "host": "0.0.0.0",
        "port": settings.service_port,
        "reload": settings.app_reload,
        "log_level": "info",
        "log_config": str(MEMOFLUX_PACKAGE_ROOT / "config/log_config.json"),
        "timeout_graceful_shutdown": 0,
    }
    reload_dirs = _resolve_reload_dirs()
    if settings.app_reload and reload_dirs:
        uvicorn_options["reload_dirs"] = reload_dirs
        uvicorn_options["reload_includes"] = ["*.json"]
    uvicorn.run("memoflux.main:app", **uvicorn_options)


def main() -> None:
    """命令行入口。"""

    run_server()
