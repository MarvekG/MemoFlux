from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_LANGUAGE = "zh"


class I18nService:
    """加载 MemoFlux 本地 JSON 语言包并提供翻译。"""

    def __init__(self, *, locales_dir: Path | None = None, default_language: str = DEFAULT_LANGUAGE) -> None:
        """初始化语言服务。

        Args:
            locales_dir: 语言包目录，默认使用 `memoflux/locales`。
            default_language: 缺省语言。
        """

        self.locales_dir = locales_dir or Path(__file__).resolve().parent / "locales"
        self.default_language = default_language
        self._locales: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, dict[str, str]] = {}
        self._load_locales()

    def get_locale(self, lang: str | None = None) -> dict[str, Any]:
        """返回指定语言的完整语言包。

        Args:
            lang: 语言代码，支持 `zh`、`zh-CN`、`en`、`en-US` 等形式。

        Returns:
            匹配语言包；未命中时返回默认语言包。
        """

        normalized = normalize_language(lang, default=self.default_language)
        return self._locales.get(normalized) or self._locales.get(self.default_language, {})

    def get(self, key: str, default: Any = None, *, lang: str | None = None) -> str:
        """按 key 获取翻译文本。

        Args:
            key: 点分隔翻译键。
            default: 未命中时返回的默认值。
            lang: 语言代码。

        Returns:
            翻译文本或默认值。
        """

        normalized = normalize_language(lang, default=self.default_language)
        cache = self._cache.get(normalized) or self._cache.get(self.default_language, {})
        return cache.get(key, str(default if default is not None else key))

    def t(self, key: str, *, lang: str | None = None, **kwargs: Any) -> str:
        """翻译并填充模板参数。

        Args:
            key: 点分隔翻译键。
            lang: 语言代码。
            **kwargs: 模板参数。

        Returns:
            翻译后的文本。
        """

        return _format_template(self.get(key, key, lang=lang), kwargs)

    def _load_locales(self) -> None:
        """加载语言包目录中的 JSON 文件。"""

        if not self.locales_dir.exists():
            return
        for path in self.locales_dir.glob("*.json"):
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            lang = normalize_language(path.stem, default=self.default_language)
            self._locales[lang] = data
            self._cache[lang] = _flatten_locale(data)


def normalize_language(lang: str | None, *, default: str = DEFAULT_LANGUAGE) -> str:
    """规范化语言代码。

    Args:
        lang: 请求语言，可能来自路径、query 或 Accept-Language。
        default: 缺省语言。

    Returns:
        当前支持的短语言代码。
    """

    if not lang:
        return default
    primary = lang.split(",", maxsplit=1)[0].split(";", maxsplit=1)[0].strip().lower()
    if primary.startswith("en"):
        return "en"
    if primary.startswith("zh"):
        return "zh"
    return default


def _flatten_locale(data: dict[str, Any]) -> dict[str, str]:
    """把嵌套语言包转换为点分隔缓存。"""

    cache: dict[str, str] = {}

    def recurse(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                recurse(child, f"{prefix}.{key}" if prefix else key)
            return
        cache[prefix] = str(value)
        leaf_key = prefix.split(".")[-1]
        cache.setdefault(leaf_key, str(value))

    recurse(data)
    return cache


def _format_template(template: str, params: dict[str, Any]) -> str:
    """填充 `{name}` 或 `{{name}}` 风格模板参数。"""

    if not params:
        return template
    normalized = re.sub(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", r"{\1}", template)

    class SafeFormatDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return normalized.format_map(SafeFormatDict(params))


i18n_service = I18nService()
