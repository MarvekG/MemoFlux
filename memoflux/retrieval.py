from __future__ import annotations

import re

_SYNONYMS = {
    "上线": {"发布", "部署", "投产"},
    "发布": {"上线", "部署", "投产"},
    "卡住": {"延期", "阻塞", "失败"},
    "延期": {"卡住", "阻塞", "延迟"},
    "为什么": {"原因", "因为"},
    "原因": {"为什么", "因为"},
    "变化": {"后来", "之前", "确认", "补充"},
}


def build_terms(text: str) -> set[str]:
    """构建文本/时间检索使用的原型检索词。"""

    ascii_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", text)}
    cjk_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    known_terms = {term for term in _SYNONYMS if term in text}
    terms = ascii_terms | cjk_terms | known_terms
    for term in list(terms):
        terms.update(_SYNONYMS.get(term, set()))
    return terms


def score_content(content: str, terms: set[str]) -> int:
    """计算文本匹配分数。"""

    lower_content = content.lower()
    return sum(1 for term in terms if term and term.lower() in lower_content)


def is_history_query(query: str) -> bool:
    """判断查询是否明显需要按时间顺序组织。"""

    return any(marker in query for marker in ("历史", "之前", "后来", "变化", "过程", "时间线"))
