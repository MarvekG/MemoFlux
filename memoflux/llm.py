from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from memoflux.llm_schemas import AnswerSynthesisInput, AnswerSynthesisOutput, QueryPlanInput, QueryPlanOutput


@dataclass(frozen=True)
class LLMResult:
    """LLM 调用结果。"""

    output: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
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
            {"role": "system", "content": "你是 MemoFlux 查询规划器。只输出 JSON：query_type 和 rewritten_queries。"},
            {"role": "user", "content": prompt_input.model_dump_json()},
        ]
        result = self._chat(messages)
        raw_output = _parse_json_object(result.output.get("content", ""), fallback={"query_type": "direct", "rewritten_queries": [query]})
        output = QueryPlanOutput.model_validate(raw_output)
        if not output.rewritten_queries:
            output = QueryPlanOutput(query_type=output.query_type, rewritten_queries=[query])
        return LLMResult(output=output.model_dump(), input_tokens=result.input_tokens, output_tokens=result.output_tokens, model=result.model)

    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        """调用真实 LLM 基于候选记忆整合答案。"""

        memory_text = "\n".join(f"[{index + 1}] {memory.content}" for index, memory in enumerate(memories))
        prompt_input = AnswerSynthesisInput(
            query=query,
            memories=[
                {"memory_id": memory.memory_id, "content": memory.content, "occurred_at": memory.occurred_at}
                for memory in memories
            ],
        )
        messages = [
            {
                "role": "system",
                "content": "你是 MemoFlux 答案整合器。只能基于给定记忆回答。只输出 JSON：answer、confidence、used_memory_ids、uncertainties。confidence 必须是 0 到 1 的数字，used_memory_ids 必须来自候选 memory_id。",
            },
            {"role": "user", "content": prompt_input.model_dump_json()},
        ]
        result = self._chat(messages)
        raw_output = _parse_json_object(result.output.get("content", ""), fallback={"answer": memory_text, "confidence": 0.4, "used_memory_ids": [memory.memory_id for memory in memories], "uncertainties": []})
        try:
            output = AnswerSynthesisOutput.model_validate(raw_output)
        except ValidationError:
            output = AnswerSynthesisOutput(
                answer=str(raw_output.get("answer") or memory_text),
                confidence=0.6,
                used_memory_ids=[],
                uncertainties=["invalid_llm_output"],
            )
        return LLMResult(output=output.model_dump(), input_tokens=result.input_tokens, output_tokens=result.output_tokens, model=result.model)

    def run_prompt(self, *, prompt_key: str, payload: dict) -> LLMResult:
        """调用真实 LLM 执行 prompt 调试。"""

        messages = [
            {"role": "system", "content": f"执行 MemoFlux prompt：{prompt_key}，输出 JSON。"},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        result = self._chat(messages)
        output = _parse_json_object(result.output.get("content", ""), fallback={"content": result.output.get("content", "")})
        return LLMResult(output=output, input_tokens=result.input_tokens, output_tokens=result.output_tokens, model=result.model)

    def _chat(self, messages: list[dict[str, str]]) -> LLMResult:
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0.1}, ensure_ascii=False).encode("utf-8")
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
            model=data.get("model") or self.model,
        )

def _parse_json_object(text: str, *, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return fallback
    return parsed if isinstance(parsed, dict) else fallback


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)
