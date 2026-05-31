from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from memoflux.llm_schemas import AnswerSynthesisInput, AnswerSynthesisOutput, QueryPlanInput, QueryPlanOutput


@dataclass(frozen=True)
class LLMResult:
    """LLM 调用结果。"""

    output: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    model: str | None = None


class LocalLLMClient:
    """无外部 LLM 配置时使用的本地占位 LLM。"""

    def plan_query(self, *, query: str) -> LLMResult:
        """生成本地 query plan。"""

        output = QueryPlanOutput(query_type="direct", rewritten_queries=[query])
        return LLMResult(output=output.model_dump(), input_tokens=_estimate_tokens(query), output_tokens=4, model="local")

    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        """基于候选记忆生成本地答案。"""

        answer = "\n".join(memory.content for memory in memories)
        output = AnswerSynthesisOutput(
            answer=answer,
            confidence=0.6,
            used_memory_ids=[memory.memory_id for memory in memories],
            relevance_by_id={memory.memory_id: "本地模式将该候选记忆用于答案。" for memory in memories},
            uncertainties=[],
        )
        return LLMResult(output=output.model_dump(), input_tokens=_estimate_tokens(query), output_tokens=_estimate_tokens(answer), model="local")

    def run_prompt(self, *, prompt_key: str, payload: dict) -> LLMResult:
        """返回本地 prompt 调试结果。"""

        return LLMResult(output={"prompt_key": prompt_key, "payload": payload}, input_tokens=1, output_tokens=1, model="local")


@dataclass(frozen=True)
class OpenAICompatibleLLMClient:
    """OpenAI-compatible / LiteLLM Chat Completions 客户端。"""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60

    def plan_query(self, *, query: str) -> LLMResult:
        """调用真实 LLM 生成 query plan。"""

        prompt_input = QueryPlanInput(query=query)
        messages = [
            {
                "role": "system",
                "content": _with_output_schema(
                    "你是 MemoFlux 查询规划器。只输出 JSON：query_type 和 rewritten_queries。",
                    QueryPlanOutput,
                ),
            },
            {"role": "user", "content": prompt_input.model_dump_json()},
        ]
        result = self._chat(messages)
        raw_output = _parse_json_object(result.output.get("content", ""), fallback={"query_type": "direct", "rewritten_queries": [query]})
        output = QueryPlanOutput.model_validate(raw_output)
        if not output.rewritten_queries:
            output = QueryPlanOutput(query_type=output.query_type, rewritten_queries=[query])
        return LLMResult(
            output=output.model_dump(),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            model=result.model,
        )

    def synthesize_answer(self, *, query: str, memories: list, query_type: str = "direct") -> LLMResult:
        """调用真实 LLM 基于候选记忆整合答案。"""

        prompt_input = AnswerSynthesisInput(
            query=query,
            query_type=query_type,
            memories=[
                {"memory_id": memory.memory_id, "content": memory.content, "occurred_at": memory.occurred_at}
                for memory in memories
            ],
        )
        messages = [
            {
                "role": "system",
                "content": _with_output_schema(
                    "你是 MemoFlux 答案整合器。只能基于给定记忆回答，不要猜测，不要用其他主体的候选回答当前主体的问题。请按以下顺序判断：1. 识别问题主体和问题要问的事实。2. 在候选中找主体一致、能直接回答该事实的记忆。3. 如果存在这样的记忆，即使只有一条，也必须回答，并把 answerability 设为 answerable。4. 只有在候选中没有任何主体一致且能直接回答问题的记忆时，才允许拒答，并把 answerability 设为 not_answerable。当前/最新事实规则：如果问题询问当前或最新事实，候选中表达“纠正为/更新为/改为/之前记录不再作为当前判断依据”的记忆就是当前事实证据，必须优先使用。当前依赖规则：dependency/dependency_risk/dependency_analysis/association/current_association 类问题询问当前依赖或依赖风险时，候选中明确写出“当前依赖 ... 服务，依赖风险是 ...，之前依赖记录不再作为当前判断依据”的记忆就是当前依赖证据；如果有多条主体一致的当前依赖证据，必须使用 occurred_at 最新的一条，不要使用更早的当前依赖记录。历史依赖规则：association_history 或询问“历史上依赖过哪些服务”的问题，必须汇总候选中所有主体一致且描述依赖/当前依赖的记忆，不能因为其中写了“当前依赖”就排除在历史之外；答案应列全这些服务并引用对应记忆。历史规则：history/summary/temporal/association_history 类问题需要总结候选中的历史事件，不要按当前事实查询的标准拒答。输出必须是 JSON，字段包括 answer、confidence、used_memory_ids、relevance_by_id、answerability、answerability_reason、uncertainties。confidence 必须是 0 到 1 的数字。used_memory_ids 必须来自候选 memory_id。relevance_by_id 的 key 必须来自候选 memory_id，value 用一句话说明为什么该记忆支持答案。answerability_reason 用一句话说明为什么可答或不可答。如果不可答，answer 必须是：当前 session 中没有足够记忆支持回答该问题。confidence 设为 0 到 0.3，used_memory_ids 返回空数组。",
                    AnswerSynthesisOutput,
                ),
            },
            {"role": "user", "content": prompt_input.model_dump_json()},
        ]
        result = self._chat(messages)
        safe_fallback = _safe_answer_fallback()
        raw_output = _parse_json_object(
            result.output.get("content", ""),
            fallback=safe_fallback,
        )
        try:
            output = AnswerSynthesisOutput.model_validate(raw_output)
        except ValidationError:
            output = AnswerSynthesisOutput.model_validate(safe_fallback)
        return LLMResult(
            output=output.model_dump(),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            model=result.model,
        )

    def run_prompt(self, *, prompt_key: str, payload: dict) -> LLMResult:
        """调用真实 LLM 执行 prompt 调试。"""

        messages = [
            {"role": "system", "content": f"执行 MemoFlux prompt：{prompt_key}，输出 JSON。"},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        result = self._chat(messages)
        output = _parse_json_object(result.output.get("content", ""), fallback={"content": result.output.get("content", "")})
        return LLMResult(
            output=output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            model=result.model,
        )

    def _chat(self, messages: list[dict[str, str]]) -> LLMResult:
        payload_data: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.0}
        payload = json.dumps(payload_data, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return LLMResult(
            output={"content": content},
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=_cached_tokens_from_usage(usage),
            reasoning_tokens=_reasoning_tokens_from_usage(usage),
            model=data.get("model") or self.model,
        )


def _cached_tokens_from_usage(usage: dict[str, Any]) -> int:
    """从 OpenAI-compatible usage 中提取缓存命中 token 数。

    Args:
        usage: Chat Completions 返回的 usage 对象。

    Returns:
        已缓存 prompt token 数；字段缺失时返回 0。
    """

    prompt_details = usage.get("prompt_tokens_details") or {}
    return int(prompt_details.get("cached_tokens") or usage.get("cached_tokens") or 0)


def _reasoning_tokens_from_usage(usage: dict[str, Any]) -> int:
    """从 OpenAI-compatible usage 中提取 reasoning token 数。

    Args:
        usage: Chat Completions 返回的 usage 对象。

    Returns:
        reasoning token 数；字段缺失时返回 0。
    """

    completion_details = usage.get("completion_tokens_details") or {}
    return int(completion_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)


def _parse_json_object(text: str, *, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return fallback
    return parsed if isinstance(parsed, dict) else fallback


def _with_output_schema(system_prompt: str, model: type[BaseModel]) -> str:
    """把 Pydantic 输出模型的 JSON Schema 注入系统提示词。

    Args:
        system_prompt: 原始系统提示词。
        model: 用于约束 LLM 输出的 Pydantic 模型类。

    Returns:
        追加结构化输出 schema 要求后的系统提示词。
    """

    schema = json.dumps(model.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    return (
        f"{system_prompt}\n\n"
        f"必须直接输出符合 {model.__name__} 的 JSON 对象，不要输出 Markdown 或额外解释。\n"
        f"JSON Schema：{schema}"
    )


def _safe_answer_fallback() -> dict[str, Any]:
    """构造 Answer Synthesizer 输出异常时的安全拒答结果。

    Returns:
        不引用任何候选记忆的固定拒答结构。
    """

    return {
        "answer": "当前 session 中没有足够记忆支持回答该问题。",
        "confidence": 0.1,
        "used_memory_ids": [],
        "relevance_by_id": {},
        "answerability": "not_answerable",
        "answerability_reason": "Answer Synthesizer 输出无法解析或不符合结构化 schema。",
        "uncertainties": ["invalid_llm_output"],
    }


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)
