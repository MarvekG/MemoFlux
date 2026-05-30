from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryPlanInput(BaseModel):
    """Query Planner 的结构化输入。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)


class QueryPlanOutput(BaseModel):
    """Query Planner 的结构化输出。"""

    model_config = ConfigDict(extra="ignore")

    query_type: str = "direct"
    rewritten_queries: list[str] = Field(default_factory=list)

    @field_validator("rewritten_queries", mode="before")
    @classmethod
    def coerce_rewritten_queries(cls, value):
        """兼容 LLM 返回字符串或缺失列表的情况。"""

        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class AnswerMemoryCandidate(BaseModel):
    """Answer Synthesizer 可引用的候选记忆。"""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    content: str
    occurred_at: datetime


class AnswerSynthesisInput(BaseModel):
    """Answer Synthesizer 的结构化输入。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    memories: list[AnswerMemoryCandidate]


class AnswerSynthesisOutput(BaseModel):
    """Answer Synthesizer 的结构化输出。"""

    model_config = ConfigDict(extra="ignore")

    answer: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    used_memory_ids: list[str] = Field(default_factory=list)
    relevance_by_id: dict[str, str] = Field(default_factory=dict)
    uncertainties: list[str] = Field(default_factory=list)

    @field_validator("relevance_by_id", mode="before")
    @classmethod
    def coerce_relevance_by_id(cls, value):
        """兼容 LLM 缺失或返回非字典引用理由的情况。"""

        if value is None:
            return {}
        if not isinstance(value, dict):
            return {}
        return {str(key): str(reason) for key, reason in value.items()}
