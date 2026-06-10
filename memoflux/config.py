from __future__ import annotations

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class MemoFluxSettings(BaseSettings):
    """MemoFlux 运行配置。"""

    database_url: str | None = Field(
        default="postgresql+asyncpg://tradeuser:tradepassword@memo-postgres:5432/memory",
        alias="MEMOFLUX_DATABASE_URL",
    )
    database_schema: str = Field(default="memoflux", alias="MEMOFLUX_DATABASE_SCHEMA")
    llm_base_url: str | None = Field(default=None, alias="MEMOFLUX_LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, alias="MEMOFLUX_LLM_API_KEY")
    llm_model: str = Field(default="memory/default", alias="MEMOFLUX_LLM_MODEL")
    app_reload: bool = Field(default=True, alias="MEMOFLUX_APP_RELOAD")
    service_port: int = Field(default=8020, alias="MEMOFLUX_SERVICE_PORT")
    embedding_provider: str = Field(default="local", alias="MEMOFLUX_EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-base-zh-v1.5", alias="MEMOFLUX_EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=768, alias="MEMOFLUX_EMBEDDING_DIM")
    embedding_api_key: str | None = Field(default=None, alias="MEMOFLUX_EMBEDDING_API_KEY")
    embedding_base_url: str | None = Field(default=None, alias="MEMOFLUX_EMBEDDING_BASE_URL")
    embedding_timeout_seconds: float = Field(default=30.0, alias="MEMOFLUX_EMBEDDING_TIMEOUT_SECONDS")
    embedding_cache_dir: str | None = Field(default="/home/memo/.memoflux/data/models", alias="MEMOFLUX_EMBEDDING_CACHE_DIR")
    embedding_use_local_weights: bool = Field(default=True, alias="MEMOFLUX_EMBEDDING_USE_LOCAL_WEIGHTS")
    memory_cleanup_enabled: bool = Field(default=True, alias="MEMOFLUX_MEMORY_CLEANUP_ENABLED")
    memory_retention_days: int = Field(default=180, alias="MEMOFLUX_MEMORY_RETENTION_DAYS")
    memory_cleanup_hour: int = Field(default=4, alias="MEMOFLUX_MEMORY_CLEANUP_HOUR")
    memory_cleanup_minute: int = Field(default=15, alias="MEMOFLUX_MEMORY_CLEANUP_MINUTE")

    model_config = ConfigDict(
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def apply_embedding_defaults(self) -> "MemoFluxSettings":
        """补齐 embedding 复用 LLM 连接配置的默认值。

        Returns:
            已补齐 embedding API key 和 base URL 的配置对象。
        """

        if self.embedding_api_key in (None, ""):
            self.embedding_api_key = self.llm_api_key
        if self.embedding_base_url in (None, ""):
            self.embedding_base_url = self.llm_base_url
        return self

    @field_validator("memory_retention_days", mode="before")
    @classmethod
    def validate_memory_retention_days(cls, value) -> int:
        """校验记忆保留期配置。

        Args:
            value: 环境变量或默认配置中的原始值。

        Returns:
            限制在允许范围内的记忆保留天数。
        """

        return _coerce_int(
            value,
            default=180,
            min_value=1,
            max_value=3650,
        )

    @field_validator("memory_cleanup_hour", mode="before")
    @classmethod
    def validate_memory_cleanup_hour(cls, value) -> int:
        """校验记忆清理任务的每日执行小时。

        Args:
            value: 环境变量或默认配置中的原始值。

        Returns:
            限制在 0 到 23 之间的小时值。
        """

        return _coerce_int(value, default=4, min_value=0, max_value=23)

    @field_validator("memory_cleanup_minute", mode="before")
    @classmethod
    def validate_memory_cleanup_minute(cls, value) -> int:
        """校验记忆清理任务的每日执行分钟。

        Args:
            value: 环境变量或默认配置中的原始值。

        Returns:
            限制在 0 到 59 之间的分钟值。
        """

        return _coerce_int(value, default=15, min_value=0, max_value=59)


def load_settings() -> MemoFluxSettings:
    """加载 MemoFlux 运行配置。

    Returns:
        当前进程环境变量解析后的 MemoFlux 配置。
    """

    return MemoFluxSettings()


def _coerce_int(value, *, default: int, min_value: int, max_value: int) -> int:
    """将配置值转换为带上下界的整数。

    Args:
        value: 原始配置值。
        default: 无法解析时使用的默认值。
        min_value: 允许的最小值。
        max_value: 允许的最大值。

    Returns:
        解析后限制在上下界内的整数。
    """

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))
