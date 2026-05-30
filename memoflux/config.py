from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class MemoFluxSettings:
    """MemoFlux 运行配置。"""

    database_url: str | None
    database_schema: str = "memoflux"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "memory/default"
    app_reload: bool = True
    service_port: int = 8020
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_dimension: int = 768
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_timeout_seconds: float = 30.0
    embedding_prewarm_on_startup: bool = True
    embedding_cache_dir: str | None = None
    embedding_local_files_only: bool = False
    hf_endpoint: str = "https://hf-mirror.com"


def load_settings() -> MemoFluxSettings:
    """从环境变量读取 MemoFlux 配置。"""

    return MemoFluxSettings(
        database_url=_first_env("MEMOFLUX_DATABASE_URL"),
        database_schema=_first_env("MEMOFLUX_DATABASE_SCHEMA", default="memoflux"),
        llm_base_url=_first_env("MEMOFLUX_LLM_BASE_URL"),
        llm_api_key=_first_env("MEMOFLUX_LLM_API_KEY"),
        llm_model=getenv("MEMOFLUX_LLM_MODEL", "memory/default"),
        app_reload=_bool_env("MEMOFLUX_APP_RELOAD", default=True),
        service_port=int(_first_env("MEMOFLUX_SERVICE_PORT", default="8020")),
        embedding_provider=_first_env("MEMOFLUX_EMBEDDING_PROVIDER", default="local"),
        embedding_model=_first_env("MEMOFLUX_EMBEDDING_MODEL", default="BAAI/bge-base-zh-v1.5"),
        embedding_dimension=int(_first_env("MEMOFLUX_EMBEDDING_DIM", default="768")),
        embedding_api_key=_first_env("MEMOFLUX_EMBEDDING_API_KEY", "MEMOFLUX_LLM_API_KEY"),
        embedding_base_url=_first_env("MEMOFLUX_EMBEDDING_BASE_URL", "MEMOFLUX_LLM_BASE_URL"),
        embedding_timeout_seconds=float(_first_env("MEMOFLUX_EMBEDDING_TIMEOUT_SECONDS", default="30")),
        embedding_prewarm_on_startup=_bool_env("MEMOFLUX_EMBEDDING_PREWARM_ON_STARTUP", default=True),
        embedding_cache_dir=_first_env("MEMOFLUX_EMBEDDING_CACHE_DIR"),
        embedding_local_files_only=_bool_env("MEMOFLUX_EMBEDDING_LOCAL_FILES_ONLY", default=False),
        hf_endpoint=_first_env("MEMOFLUX_HF_ENDPOINT", default="https://hf-mirror.com"),
    )


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = getenv(name)
        if value not in (None, ""):
            return value
    return default


def _bool_env(*names: str, default: bool) -> bool:
    value = _first_env(*names)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
