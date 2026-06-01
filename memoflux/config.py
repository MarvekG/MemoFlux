from __future__ import annotations

from pydantic import ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings


class MemoFluxSettings(BaseSettings):
    """MemoFlux 运行配置。"""

    database_url: str | None = Field(default=None, alias="MEMOFLUX_DATABASE_URL")
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
    embedding_prewarm_on_startup: bool = Field(default=True, alias="MEMOFLUX_EMBEDDING_PREWARM_ON_STARTUP")
    embedding_cache_dir: str | None = Field(default=None, alias="MEMOFLUX_EMBEDDING_CACHE_DIR")
    embedding_local_files_only: bool = Field(default=False, alias="MEMOFLUX_EMBEDDING_LOCAL_FILES_ONLY")
    hf_endpoint: str = Field(default="https://hf-mirror.com", alias="MEMOFLUX_HF_ENDPOINT")

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


def load_settings() -> MemoFluxSettings:
    """加载 MemoFlux 运行配置。

    Returns:
        当前进程环境变量解析后的 MemoFlux 配置。
    """

    return MemoFluxSettings()
