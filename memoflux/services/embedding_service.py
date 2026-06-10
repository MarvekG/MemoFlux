from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from memoflux.config import load_settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    """通过本地模型或 OpenAI 兼容接口生成向量。"""

    def __init__(self) -> None:
        settings = load_settings()
        self._provider = settings.embedding_provider.strip().lower()
        self._dimension = int(settings.embedding_dimension)
        self._model = settings.embedding_model
        self._api_key = settings.embedding_api_key
        self._base_url = settings.embedding_base_url
        self._timeout_seconds = float(settings.embedding_timeout_seconds)
        self._cache_dir = settings.embedding_cache_dir
        self._use_local_weights = settings.embedding_use_local_weights
        self._local_model: Any | None = None
        self._local_model_lock = Lock()

    @property
    def provider(self) -> str:
        """返回当前配置的向量服务提供方。"""

        return self._provider

    @property
    def dimension(self) -> int:
        """返回当前配置的向量维度。"""

        return self._dimension

    @property
    def model(self) -> str:
        """返回当前配置的向量模型名称。"""

        return self._model

    async def embed_text(self, text: str) -> list[float]:
        """异步生成单条文本的向量。

        Args:
            text: 待向量化的文本。

        Returns:
            归一化后的浮点向量。
        """

        vectors = await self.embed_texts([text])
        return vectors[0] if vectors else []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """异步使用配置的提供方批量生成文本向量。

        Args:
            texts: 待向量化的文本列表。

        Returns:
            与输入顺序一致的向量列表。

        Raises:
            ValueError: 配置了不支持的向量提供方。
        """

        normalized = [str(text or "") for text in texts]
        if not normalized:
            return []
        if self._provider == "local":
            return await asyncio.to_thread(self._embed_local_texts, normalized)
        if self._provider == "openai_compatible":
            return await self._embed_openai_compatible_texts(normalized)
        raise ValueError(f"Unsupported MEMOFLUX_EMBEDDING_PROVIDER: {self._provider}")

    def _embed_local_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load_local_model()
        vectors = model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        coerced = [[float(value) for value in vector] for vector in vectors]
        for vector in coerced:
            self._validate_dimension(vector)
        return coerced

    def _load_local_model(self) -> Any:
        """延迟加载并复用本地向量模型。

        Returns:
            已加载的 SentenceTransformer 模型实例。
        """

        if self._local_model is not None:
            return self._local_model
        with self._local_model_lock:
            if self._local_model is not None:
                return self._local_model
            started = perf_counter()
            logger.info(
                "Loading local embedding model",
                extra={
                    "embedding_model": self._model,
                    "use_local_weights": self._use_local_weights,
                    "cache_dir_present": bool(self._cache_dir),
                },
            )
            if self._cache_dir:
                cache_path = Path(self._cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
            from sentence_transformers import SentenceTransformer

            model_path = self._resolve_local_model_path(cache_path) if self._use_local_weights and self._cache_dir else self._model
            kwargs: dict[str, Any] = {"local_files_only": self._use_local_weights}
            if self._cache_dir:
                kwargs["cache_folder"] = self._cache_dir
            self._local_model = SentenceTransformer(str(model_path), **kwargs)
            logger.info(
                "Loaded local embedding model",
                extra={
                    "embedding_model": self._model,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                },
            )
            return self._local_model

    def _resolve_local_model_path(self, cache_path: Path) -> str | Path:
        """优先解析构建期缓存的本地模型目录。

        Args:
            cache_path: SentenceTransformer 缓存根目录。

        Returns:
            可直接传给 SentenceTransformer 的本地 snapshot 目录；如果缓存目录还未初始化，则返回模型名。
        """

        snapshot_root = cache_path / f"models--{self._model.replace('/', '--')}" / "snapshots"
        if not snapshot_root.exists():
            return self._model
        snapshots = sorted(
            path
            for path in snapshot_root.iterdir()
            if path.is_dir() and (path / "config.json").exists() and (path / "modules.json").exists()
        )
        return snapshots[-1] if snapshots else self._model

    async def _embed_openai_compatible_texts(self, texts: list[str]) -> list[list[float]]:
        """调用 OpenAI 兼容接口异步生成向量。

        Args:
            texts: 待向量化的文本列表。

        Returns:
            与输入顺序一致的向量列表。

        Raises:
            ValueError: 缺少 API key 或返回维度不符合配置。
        """

        if not self._api_key:
            raise ValueError("MEMOFLUX_EMBEDDING_API_KEY is required for openai_compatible embedding provider")
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": self._api_key, "timeout": self._timeout_seconds}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        client = AsyncOpenAI(**kwargs)
        response = await client.embeddings.create(model=self._model, input=texts, timeout=self._timeout_seconds)
        ordered = sorted(response.data, key=lambda item: int(item.index))
        vectors = [[float(value) for value in item.embedding] for item in ordered]
        for vector in vectors:
            self._validate_dimension(vector)
        return vectors

    def _validate_dimension(self, vector: list[float]) -> None:
        actual = len(vector)
        if actual != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {actual}. "
                "Check MEMOFLUX_EMBEDDING_MODEL and MEMOFLUX_EMBEDDING_DIM."
            )


embedding_service = EmbeddingService()
