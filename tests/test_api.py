from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from memoflux.api import create_app
from memoflux.config import load_settings

from memoflux.llm import LLMResult, OpenAICompatibleLLMClient
from memoflux.models import MemoryRecord
from memoflux.service import MemoFluxService
from memoflux.storage.memory import MemoryRepository


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = []

    def plan_query(self, *, query: str) -> LLMResult:
        self.calls.append(("plan_query", query))
        return LLMResult(
            output={"rewritten_queries": [query]},
            input_tokens=10,
            output_tokens=5,
            model="fake",
        )

    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        self.calls.append(("synthesize_answer", query, len(memories)))
        memory_ids = [memory.memory_id for memory in memories]
        return LLMResult(
            output={
                "answer": "LLM 整合答案：" + "；".join(memory.content for memory in memories),
                "confidence": 0.8,
                "used_memory_ids": memory_ids,
                "relevance_by_id": {memory.memory_id: "该记忆被用于生成答案。" for memory in memories},
                "uncertainties": [],
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )

    def run_prompt(self, *, prompt_key: str, payload: dict) -> LLMResult:
        self.calls.append(("run_prompt", prompt_key))
        return LLMResult(output={"prompt_key": prompt_key, "payload": payload}, input_tokens=3, output_tokens=4, model="fake")


class StringConfidenceLLMClient(FakeLLMClient):
    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        self.calls.append(("synthesize_answer", query, len(memories)))
        return LLMResult(
            output={
                "answer": "字符串置信度答案",
                "confidence": "high",
                "used_memory_ids": [memory.memory_id for memory in memories],
                "relevance_by_id": {memory.memory_id: "该记忆被用于生成答案。" for memory in memories},
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )


class SelectiveReferenceLLMClient(FakeLLMClient):
    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        self.calls.append(("synthesize_answer", query, len(memories)))
        return LLMResult(
            output={
                "answer": "只引用第一条记忆",
                "confidence": 0.7,
                "used_memory_ids": [memories[0].memory_id],
                "relevance_by_id": {memories[0].memory_id: "只引用第一条记忆。"},
                "uncertainties": [],
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )


class DiagnosticReferenceLLMClient(FakeLLMClient):
    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        self.calls.append(("synthesize_answer", query, len(memories)))
        return LLMResult(
            output={
                "answer": "Atlas 当前依赖 AuthGate 服务。",
                "confidence": 0.9,
                "used_memory_ids": [memories[0].memory_id],
                "relevance_by_id": {memories[0].memory_id: "该记忆直接说明当前依赖。"},
                "answerability": "answerable",
                "answerability_reason": "候选记忆主体一致，并明确给出当前依赖。",
                "uncertainties": [],
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )


class NoEvidenceLLMClient(FakeLLMClient):
    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        self.calls.append(("synthesize_answer", query, len(memories)))
        return LLMResult(
            output={
                "answer": "当前 session 中没有足够记忆支持回答该问题。",
                "confidence": 0.0,
                "used_memory_ids": [],
                "relevance_by_id": {},
                "uncertainties": ["候选记忆未包含问题所需证据"],
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )


class RewrittenQueryLLMClient(FakeLLMClient):
    def plan_query(self, *, query: str) -> LLMResult:
        self.calls.append(("plan_query", query))
        return LLMResult(
            output={"rewritten_queries": ["Atlas 延期原因", "Atlas 数据库回滚风险"]},
            input_tokens=10,
            output_tokens=5,
            model="fake",
        )


class CountingLLMClient(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.synthesis_candidate_count = 0

    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        self.synthesis_candidate_count = len(memories)
        return LLMResult(
            output={
                "answer": "只引用第三条记忆",
                "confidence": 0.8,
                "used_memory_ids": [memories[2].memory_id],
                "relevance_by_id": {memories[2].memory_id: "第三条记忆提供答案。"},
                "uncertainties": [],
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )


class AuditCountingLLMClient(CountingLLMClient):
    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        self.synthesis_candidate_count = len(memories)
        return LLMResult(
            output={
                "answer": "只引用第一条记忆",
                "confidence": 0.8,
                "used_memory_ids": [memories[0].memory_id],
                "relevance_by_id": {memories[0].memory_id: "第一条记忆提供答案。"},
                "uncertainties": [],
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )


class ExpandedCandidateRepository(MemoryRepository):
    def search_memories(self, *, session: str, terms: set[str], limit: int, query_embedding=None):
        return [
            MemoryRecord(
                memory_id=f"m{index}",
                session=session,
                content=f"第 {index} 条 Atlas 记忆。",
                occurred_at=datetime(2026, 5, index, 10, tzinfo=UTC),
                created_at=datetime(2026, 5, index, 10, tzinfo=UTC),
            )
            for index in range(1, limit + 1)
        ]


class DependencyCandidateRepository(MemoryRepository):
    def search_memories(self, *, session: str, terms: set[str], limit: int, query_embedding=None):
        return [
            MemoryRecord(
                memory_id="older-current",
                session=session,
                content="Falcon 项目当前依赖 AuthGate 服务，依赖风险是容器健康检查误判，之前依赖记录不再作为当前判断依据。",
                occurred_at=datetime(2026, 5, 10, 10, tzinfo=UTC),
                created_at=datetime(2026, 5, 10, 10, tzinfo=UTC),
            ),
            MemoryRecord(
                memory_id="newer-current",
                session=session,
                content="Falcon 项目当前依赖 ReportLab 服务，依赖风险是发布审批链路超时，之前依赖记录不再作为当前判断依据。",
                occurred_at=datetime(2026, 5, 20, 10, tzinfo=UTC),
                created_at=datetime(2026, 5, 20, 10, tzinfo=UTC),
            ),
            MemoryRecord(
                memory_id="older-history",
                session=session,
                content="Falcon 项目依赖 SignalBox 服务，依赖风险是供应商接口限流。",
                occurred_at=datetime(2026, 5, 3, 10, tzinfo=UTC),
                created_at=datetime(2026, 5, 3, 10, tzinfo=UTC),
            ),
        ]


class HistoryOnlyListRepository(MemoryRepository):
    def __init__(self) -> None:
        super().__init__()
        self.list_limits = []

    def search_memories(self, *, session: str, terms: set[str], limit: int, query_embedding=None):
        return [
            MemoryRecord(
                memory_id="vector-noise",
                session=session,
                content="Zephyr 项目延期，原因是前端构建缓存策略错误。",
                occurred_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
                created_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
            )
        ]

    def list_memories(self, *, session: str, limit: int, offset: int = 0):
        self.list_limits.append(limit)
        memories = [
            MemoryRecord(
                memory_id="history-hit",
                session=session,
                content="Atlas 项目发布历史：第 1 次发布因订单索引回填校验失败暂停。",
                occurred_at=datetime(2026, 5, 2, 10, tzinfo=UTC),
                created_at=datetime(2026, 5, 2, 10, tzinfo=UTC),
            )
        ]
        return memories[offset:offset + limit], len(memories)


class HistoryQueryLLMClient(FakeLLMClient):
    def plan_query(self, *, query: str) -> LLMResult:
        self.calls.append(("plan_query", query))
        return LLMResult(
            output={"rewritten_queries": [query]},
            input_tokens=10,
            output_tokens=5,
            model="fake",
        )

    def synthesize_answer(self, *, query: str, memories: list) -> LLMResult:
        self.calls.append(("synthesize_answer", query, [memory.memory_id for memory in memories]))
        memory_ids = [memory.memory_id for memory in memories if memory.memory_id == "history-hit"]
        return LLMResult(
            output={
                "answer": "订单索引回填校验失败。",
                "confidence": 0.9,
                "used_memory_ids": memory_ids,
                "relevance_by_id": {memory_id: "该记忆提供 Atlas 发布暂停历史原因。" for memory_id in memory_ids},
                "uncertainties": [],
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )


class FakeChatLLMClient(OpenAICompatibleLLMClient):
    def __init__(self, content: str) -> None:
        super().__init__(base_url="http://unused", api_key="unused", model="fake")
        self.content = content

    def _chat(self, messages: list[dict[str, str]], **_) -> LLMResult:
        return LLMResult(output={"content": self.content}, input_tokens=10, output_tokens=5, model="fake")


class CapturePayloadLLMClient(OpenAICompatibleLLMClient):
    def __init__(self) -> None:
        super().__init__(base_url="http://unused", api_key="unused", model="fake")
        self.payload = None
        self.response_usage = {"prompt_tokens": 1, "completion_tokens": 1}

    def capture_urlopen(self, request, timeout):
        self.payload = __import__("json").loads(request.data.decode("utf-8"))
        response_usage = self.response_usage

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return __import__("json").dumps(
                    {
                        "model": "fake",
                        "choices": [{"message": {"content": '{"rewritten_queries":["q"]}'}}],
                        "usage": response_usage,
                    }
                ).encode("utf-8")

        return _Response()



class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls = []
        self.batch_calls = []
        self.prewarm_calls = 0

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        self.calls.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    def prewarm_local_model(self) -> None:
        if not load_settings().embedding_prewarm_on_startup:
            return
        self.prewarm_calls += 1


class VectorAwareMemoryRepository(MemoryRepository):
    def __init__(self) -> None:
        super().__init__()
        self.insert_embeddings = []
        self.search_embeddings = []

    def insert_memory(self, *, session: str, content: str, occurred_at, embedding=None):
        self.insert_embeddings.append(embedding)
        return super().insert_memory(session=session, content=content, occurred_at=occurred_at)

    def search_memories(self, *, session: str, terms: set[str], limit: int, query_embedding=None):
        self.search_embeddings.append(query_embedding)
        return super().search_memories(session=session, terms=terms, limit=limit)


def test_ingest_recall_delete_and_preview_flow():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)

    ingest_response = client.post(
        "/v1/ingest",
        json={
            "session": "project:atlas",
            "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )
    assert ingest_response.status_code == 200
    memory_id = ingest_response.json()["data"]["memory_id"]

    recall_response = client.post(
        "/v1/recall",
        json={"session": "project:atlas", "query": "为什么 Atlas 上线被卡住？"},
    )
    assert recall_response.status_code == 200
    recall_data = recall_response.json()["data"]
    assert "数据库迁移回滚方案不完整" in recall_data["answer"]
    assert recall_data["references"][0]["memory_id"] == memory_id
    assert "usage" not in recall_data

    other_session_response = client.post(
        "/v1/recall",
        json={"session": "project:zephyr", "query": "为什么 Atlas 上线被卡住？"},
    )
    assert other_session_response.status_code == 200
    assert other_session_response.json()["data"]["answer"] == "未找到可用于回答该问题的记忆。"

    preview_response = client.get("/v1/preview", params={"session": "project:atlas"})
    assert preview_response.status_code == 200
    assert preview_response.json()["data"]["items"][0]["memory_id"] == memory_id

    delete_response = client.post(
        "/v1/delete",
        json={"session": "project:atlas", "memory_ids": [memory_id], "dry_run": False},
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["data"]["affected_memory_ids"] == [memory_id]

    after_delete_response = client.post(
        "/v1/recall",
        json={"session": "project:atlas", "query": "为什么 Atlas 上线被卡住？"},
    )
    assert after_delete_response.status_code == 200
    assert after_delete_response.json()["data"]["references"] == []


def test_preview_without_session_returns_all_sessions():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={
            "session": "project:atlas",
            "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )
    client.post(
        "/v1/ingest",
        json={
            "session": "project:zephyr",
            "content": "Zephyr 复盘发现浏览器缓存导致旧资源残留。",
            "occurred_at": "2026-05-02T10:00:00Z",
        },
    )

    preview_response = client.get("/v1/preview")

    assert preview_response.status_code == 200
    items = preview_response.json()["data"]["items"]
    assert [item["session"] for item in items] == ["project:atlas", "project:zephyr"]


def test_preview_uses_offset_and_returns_total():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    for index in range(1, 4):
        client.post(
            "/v1/ingest",
            json={
                "session": "project:atlas",
                "content": f"Atlas 第 {index} 条复盘记忆。",
                "occurred_at": f"2026-05-0{index}T10:00:00Z",
            },
        )

    preview_response = client.get("/v1/preview", params={"session": "project:atlas", "limit": 1, "offset": 1})

    assert preview_response.status_code == 200
    data = preview_response.json()["data"]
    assert data["total"] == 3
    assert data["limit"] == 1
    assert data["offset"] == 1
    assert [item["content"] for item in data["items"]] == ["Atlas 第 2 条复盘记忆。"]


def test_history_recall_orders_candidates_by_occurred_at():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={
            "session": "project:atlas",
            "content": "5 月 3 日确认需要补充回滚演练记录。",
            "occurred_at": "2026-05-03T09:00:00Z",
        },
    )
    client.post(
        "/v1/ingest",
        json={
            "session": "project:atlas",
            "content": "5 月 1 日 Atlas 发布延期，因为回滚方案不完整。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )

    response = client.post(
        "/v1/recall",
        json={"session": "project:atlas", "query": "Atlas 计划之前是怎么变化的？"},
    )

    assert response.status_code == 200
    answer = response.json()["data"]["answer"]
    assert answer.index("5 月 1 日") < answer.index("5 月 3 日")


def test_delete_query_mode_requires_dry_run():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/delete",
        json={"session": "project:atlas", "query": "删除发布延期细节", "dry_run": False},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


def test_ingest_rejects_legacy_isolation_field():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/ingest",
        json={"sco" + "pe": "legacy", "content": "旧字段应被拒绝。", "occurred_at": "2026-05-01T10:00:00Z"},
    )

    assert response.status_code == 422


def test_recall_requires_session_field():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/v1/recall", json={"sco" + "pe": "legacy", "query": "旧字段能查吗？"})

    assert response.status_code == 422


def test_i18n_endpoint_returns_locale_bundle():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/v1/i18n/en")

    data = response.json()["data"]
    assert response.status_code == 200
    assert data["common"]["success"] == "Success"
    assert data["errors"]["validation_error"] == "Validation error"


def test_general_i18n_endpoint_is_not_exposed():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/v1/general/i18n/en")

    assert response.status_code == 404


def test_http_error_uses_accept_language():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/delete",
        json={"session": "s", "query": "delete something", "dry_run": False},
        headers={"Accept-Language": "en-US,en;q=0.9"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Validation error"
    assert response.json()["error"]["details"]["reason"] == "query delete requires dry_run=true"


def test_no_memory_recall_uses_accept_language():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/recall",
        json={"session": "s", "query": "Atlas 为什么延期？"},
        headers={"Accept-Language": "en"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["answer"] == "No memory is available to answer this question."


def test_usage_stats_are_aggregate_only():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={"session": "s", "content": "Atlas 发布延期。", "occurred_at": "2026-05-01T10:00:00Z"},
    )
    client.post("/v1/recall", json={"session": "s", "query": "Atlas 为什么延期？"})

    stats_response = client.get("/v1/usage/stats")
    assert stats_response.status_code == 200
    stats = stats_response.json()["data"]
    assert stats["llm_runs"] >= 1
    assert "runs" not in stats

    clear_response = client.delete("/v1/usage/stats")
    assert clear_response.status_code == 200
    assert clear_response.json()["data"]["deleted"] >= 1


def test_usage_stats_include_cached_tokens_and_hit_rate():
    repository = MemoryRepository()
    repository.record_usage(operation="query_planner", input_tokens=100, output_tokens=20, cached_tokens=30)
    repository.record_usage(operation="answer_synthesizer", input_tokens=50, output_tokens=10, cached_tokens=10)

    stats = repository.usage_stats()

    assert stats.input_tokens == 150
    assert stats.cached_tokens == 40
    assert stats.cache_miss_tokens == 110
    assert stats.cache_hit_rate == 40 / 150
    assert stats.by_operation["query_planner"]["cached_tokens"] == 30
    assert stats.by_operation["query_planner"]["cache_miss_tokens"] == 70


def test_retention_repository_deletes_by_occurred_at():
    repository = MemoryRepository()
    old_at = datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
    cutoff = datetime(2025, 7, 1, 0, 0, tzinfo=UTC)
    recent_at = datetime(2025, 7, 2, 10, 0, tzinfo=UTC)
    old_a = repository.insert_memory(session="session:a", content="A 旧记忆。", occurred_at=old_at)
    old_b = repository.insert_memory(session="session:b", content="B 旧记忆。", occurred_at=old_at)
    recent_a = repository.insert_memory(session="session:a", content="A 新记忆。", occurred_at=recent_at)

    deleted = repository.delete_memories_before_occurred_at(cutoff)

    assert deleted == {
        "session:a": [old_a.memory_id],
        "session:b": [old_b.memory_id],
    }
    session_a_memories, _ = repository.list_memories(session="session:a", limit=10)
    session_b_memories, _ = repository.list_memories(session="session:b", limit=10)
    assert [memory.memory_id for memory in session_a_memories] == [
        recent_a.memory_id
    ]
    assert session_b_memories == []


def test_cleanup_expired_memories_records_delete_audits_and_scrubs_queries():
    repository = MemoryRepository()
    service = MemoFluxService(repository, llm_client=FakeLLMClient(), embedding_client=FakeEmbeddingService())
    app = create_app(repository=repository, llm_client=FakeLLMClient(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    old_at = "2025-01-01T10:00:00Z"
    recent_at = "2026-05-01T10:00:00Z"
    old_response = client.post(
        "/v1/ingest",
        json={"session": "session:a", "content": "Atlas 旧记忆。", "occurred_at": old_at},
    )
    old_memory_id = old_response.json()["data"]["memory_id"]
    client.post(
        "/v1/ingest",
        json={"session": "session:a", "content": "Atlas 新记忆。", "occurred_at": recent_at},
    )
    other_old_response = client.post(
        "/v1/ingest",
        json={"session": "session:b", "content": "Zephyr 旧记忆。", "occurred_at": old_at},
    )
    other_old_memory_id = other_old_response.json()["data"]["memory_id"]
    client.post("/v1/recall", json={"session": "session:a", "query": "Atlas 有哪些记忆？", "top_k": 2})

    result = service.cleanup_expired_memories(retention_days=180)

    assert result["status"] == "ok"
    assert result["deleted"] == 2
    assert result["sessions"] == 2
    assert result["retention_days"] == 180
    assert "cutoff_occurred_at" in result
    session_a_audits, _ = repository.list_audits(session="session:a", limit=10)
    session_b_audits, _ = repository.list_audits(session="session:b", limit=10)
    session_a_delete = next(item for item in session_a_audits if item.delete_id)
    session_b_delete = next(item for item in session_b_audits if item.delete_id)
    assert session_a_delete.target["mode"] == "retention_cleanup"
    assert session_a_delete.affected_memory_ids == [old_memory_id]
    assert session_b_delete.affected_memory_ids == [other_old_memory_id]
    query_audit = next(item for item in session_a_audits if getattr(item, "query_id", None))
    old_items = [item for item in query_audit.retrieved if item["memory_id"] == old_memory_id]
    assert old_items[0]["content_preview"] is None
    assert query_audit.final_answer is None
    assert old_memory_id not in query_audit.selection_reasons
    session_a_memories, _ = repository.list_memories(session="session:a", limit=10)
    assert [memory.content for memory in session_a_memories] == ["Atlas 新记忆。"]


def test_recall_uses_configured_llm_without_returning_usage():
    llm_client = FakeLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={"session": "s", "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。", "occurred_at": "2026-05-01T10:00:00Z"},
    )
    response = client.post("/v1/recall", json={"session": "s", "query": "Atlas 为什么延期？"})

    data = response.json()["data"]
    assert response.status_code == 200
    assert data["answer"].startswith("LLM 整合答案")
    assert "usage" not in data
    assert [call[0] for call in llm_client.calls] == ["plan_query", "synthesize_answer"]


def test_recall_handles_non_numeric_llm_confidence():
    llm_client = StringConfidenceLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={"session": "s", "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。", "occurred_at": "2026-05-01T10:00:00Z"},
    )
    response = client.post("/v1/recall", json={"session": "s", "query": "Atlas 为什么延期？"})

    assert response.status_code == 200
    assert response.json()["data"]["confidence"] == 0.6


def test_prompt_eval_uses_configured_llm_without_returning_usage():
    llm_client = FakeLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/v1/prompt-eval", json={"prompt_key": "query_planner", "payload": {"query": "hello"}})

    data = response.json()["data"]
    assert response.status_code == 200
    assert data["model"] == "fake"
    assert "usage" not in data
    assert llm_client.calls == [("run_prompt", "query_planner")]


def test_ingest_and_recall_use_embeddings_for_vector_retrieval():
    llm_client = FakeLLMClient()
    fake_embedding_service = FakeEmbeddingService()
    repository = VectorAwareMemoryRepository()
    app = create_app(repository=repository, llm_client=llm_client, embedding_client=fake_embedding_service)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={"session": "s", "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。", "occurred_at": "2026-05-01T10:00:00Z"},
    )
    response = client.post("/v1/recall", json={"session": "s", "query": "Atlas 为什么延期？"})

    assert response.status_code == 200
    assert repository.insert_embeddings == [[0.1, 0.2, 0.3]]
    assert repository.search_embeddings == [[0.1, 0.2, 0.3]]
    assert fake_embedding_service.calls == ["Atlas 发布延期，因为数据库迁移回滚方案不完整。", "Atlas 为什么延期？"]
    assert [call[0] for call in llm_client.calls] == ["plan_query", "synthesize_answer"]


def test_recall_searches_original_and_rewritten_queries():
    llm_client = RewrittenQueryLLMClient()
    fake_embedding_service = FakeEmbeddingService()
    repository = VectorAwareMemoryRepository()
    app = create_app(repository=repository, llm_client=llm_client, embedding_client=fake_embedding_service)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post("/v1/ingest", json={"session": "s", "content": "Atlas 延期原因是数据库回滚风险。", "occurred_at": "2026-05-01T10:00:00Z"})

    response = client.post("/v1/recall", json={"session": "s", "query": "Atlas 为什么延期？", "top_k": 3})

    assert response.status_code == 200
    assert fake_embedding_service.calls == [
        "Atlas 延期原因是数据库回滚风险。",
        "Atlas 为什么延期？",
        "Atlas 延期原因",
        "Atlas 数据库回滚风险",
    ]
    assert fake_embedding_service.batch_calls == [["Atlas 为什么延期？", "Atlas 延期原因", "Atlas 数据库回滚风险"]]
    assert repository.search_embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_recall_references_only_include_used_memory_ids():
    llm_client = SelectiveReferenceLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post("/v1/ingest", json={"session": "s", "content": "第一条 Atlas 记忆。", "occurred_at": "2026-05-01T10:00:00Z"})
    client.post("/v1/ingest", json={"session": "s", "content": "第二条 Atlas 记忆。", "occurred_at": "2026-05-02T10:00:00Z"})

    response = client.post("/v1/recall", json={"session": "s", "query": "Atlas?", "top_k": 2})

    assert response.status_code == 200
    references = response.json()["data"]["references"]
    assert len(references) == 1
    assert references[0]["quote"] == "第一条 Atlas 记忆。"
    assert references[0]["relevance"] == "只引用第一条记忆。"


def test_recall_returns_empty_references_when_llm_uses_no_memories():
    llm_client = NoEvidenceLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post("/v1/ingest", json={"session": "s", "content": "Zephyr 延期原因是构建缓存错误。", "occurred_at": "2026-05-01T10:00:00Z"})

    response = client.post("/v1/recall", json={"session": "s", "query": "Atlas 延期原因是什么？"})

    data = response.json()["data"]
    assert response.status_code == 200
    assert data["answer"] == "当前 session 中没有足够记忆支持回答该问题。"
    assert data["references"] == []
    assert data["confidence"] == 0.0
    assert data["uncertainties"] == ["候选记忆未包含问题所需证据"]


def test_openai_synthesizer_invalid_json_fails_closed():
    llm_client = FakeChatLLMClient(content="not-json")
    memory = MemoryRecord(
        memory_id="m1",
        session="s",
        content="Hydra 项目外的候选记忆不应泄漏。",
        occurred_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
        created_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
    )

    result = llm_client.synthesize_answer(query="Hydra 当前延期原因是什么？", memories=[memory])

    assert result.output == {
        "answer": "当前 session 中没有足够记忆支持回答该问题。",
        "confidence": 0.1,
        "used_memory_ids": [],
        "relevance_by_id": {},
        "answerability": "not_answerable",
        "answerability_reason": "Answer Synthesizer 输出无法解析或不符合结构化 schema。",
        "uncertainties": ["invalid_llm_output"],
    }


def test_openai_synthesizer_invalid_schema_fails_closed():
    llm_client = FakeChatLLMClient(content='{"answer":"候选泄漏","confidence":"high","used_memory_ids":["m1"]}')
    memory = MemoryRecord(
        memory_id="m1",
        session="s",
        content="Hydra 项目外的候选记忆不应泄漏。",
        occurred_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
        created_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
    )

    result = llm_client.synthesize_answer(query="Hydra 当前延期原因是什么？", memories=[memory])

    assert result.output == {
        "answer": "当前 session 中没有足够记忆支持回答该问题。",
        "confidence": 0.1,
        "used_memory_ids": [],
        "relevance_by_id": {},
        "answerability": "not_answerable",
        "answerability_reason": "Answer Synthesizer 输出无法解析或不符合结构化 schema。",
        "uncertainties": ["invalid_llm_output"],
    }


def test_openai_client_uses_deterministic_temperature():
    llm_client = CapturePayloadLLMClient()

    with patch("urllib.request.urlopen", llm_client.capture_urlopen):
        llm_client.plan_query(query="q")

    assert llm_client.payload is not None
    assert llm_client.payload["temperature"] == 0.0


def test_openai_client_extracts_cached_and_reasoning_tokens():
    llm_client = CapturePayloadLLMClient()
    llm_client.response_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 35},
        "completion_tokens_details": {"reasoning_tokens": 7},
    }

    with patch("urllib.request.urlopen", llm_client.capture_urlopen):
        result = llm_client.plan_query(query="q")

    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.cached_tokens == 35
    assert result.reasoning_tokens == 7


def test_openai_client_injects_pydantic_schema_into_answer_synthesis_prompt():
    llm_client = CapturePayloadLLMClient()
    memory = MemoryRecord(
        memory_id="m1",
        session="s",
        content="Atlas 项目当前依赖 AuthGate 服务。",
        occurred_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
        created_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
    )

    with patch("urllib.request.urlopen", llm_client.capture_urlopen):
        llm_client.synthesize_answer(query="Atlas 当前依赖哪个服务？", memories=[memory])

    assert llm_client.payload is not None
    assert "response_format" not in llm_client.payload
    system_prompt = llm_client.payload["messages"][0]["content"]
    assert "AnswerSynthesisOutput" in system_prompt
    assert "answerability" in system_prompt
    assert "JSON Schema" in system_prompt


def test_openai_client_injects_rewrite_only_schema_into_query_plan_prompt():
    llm_client = CapturePayloadLLMClient()

    with patch("urllib.request.urlopen", llm_client.capture_urlopen):
        llm_client.plan_query(query="Atlas 当前依赖哪个服务？")

    assert llm_client.payload is not None
    assert "response_format" not in llm_client.payload
    system_prompt = llm_client.payload["messages"][0]["content"]
    assert "QueryPlanOutput" in system_prompt
    assert "rewritten_queries" in system_prompt
    assert "query_type" not in system_prompt
    assert "JSON Schema" in system_prompt


def test_recall_uses_expanded_pool_for_audit_and_synthesis():
    llm_client = AuditCountingLLMClient()
    repository = ExpandedCandidateRepository()
    app = create_app(
        repository=repository,
        llm_client=llm_client,
        embedding_client=FakeEmbeddingService(),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/v1/recall", json={"session": "s", "query": "Atlas?", "top_k": 2})

    data = response.json()["data"]
    audits_response = client.get("/v1/audits", params={"session": "s"})
    audits = audits_response.json()["data"]["items"]
    assert response.status_code == 200
    assert llm_client.synthesis_candidate_count == 2
    assert len(audits[0]["retrieved"]) == 6
    assert data["references"] == [
        {
            "memory_id": "m1",
            "occurred_at": "2026-05-01T10:00:00+00:00",
            "quote": "第 1 条 Atlas 记忆。",
            "relevance": "第一条记忆提供答案。",
        }
    ]


def test_history_recall_includes_list_memory_pool_when_vector_misses():
    llm_client = HistoryQueryLLMClient()
    repository = HistoryOnlyListRepository()
    app = create_app(repository=repository, llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/v1/recall", json={"session": "s", "query": "按历史记录总结 Atlas 发布暂停原因。", "top_k": 2})

    data = response.json()["data"]
    assert response.status_code == 200
    assert data["references"][0]["memory_id"] == "history-hit"
    assert repository.list_limits == [6]
    synthesis_call = llm_client.calls[-1]
    assert synthesis_call == ("synthesize_answer", "按历史记录总结 Atlas 发布暂停原因。", ["vector-noise", "history-hit"])
def test_audits_return_persisted_recall_records():
    llm_client = FakeLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={
            "session": "project:atlas",
            "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )
    recall_response = client.post("/v1/recall", json={"session": "project:atlas", "query": "Atlas 为什么延期？"})
    query_id = recall_response.json()["data"]["query_id"]

    audits_response = client.get("/v1/audits", params={"session": "project:atlas"})

    assert audits_response.status_code == 200
    items = audits_response.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["audit_type"] == "query"
    assert items[0]["query_id"] == query_id
    assert items[0]["session"] == "project:atlas"
    assert items[0]["original_query"] == "Atlas 为什么延期？"
    assert "query_type" not in items[0]
    assert items[0]["selected_memory_ids"]
    assert items[0]["final_answer"].startswith("LLM 整合答案")
    assert "llm_usage" not in items[0]


def test_audits_return_selection_reasons():
    llm_client = SelectiveReferenceLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post("/v1/ingest", json={"session": "s", "content": "第一条 Atlas 记忆。", "occurred_at": "2026-05-01T10:00:00Z"})
    recall_response = client.post("/v1/recall", json={"session": "s", "query": "Atlas?"})
    query_id = recall_response.json()["data"]["query_id"]

    audits_response = client.get("/v1/audits", params={"session": "s", "query_id": query_id})

    item = audits_response.json()["data"]["items"][0]
    selected_id = item["selected_memory_ids"][0]
    assert item["selection_reasons"] == {selected_id: "只引用第一条记忆。"}


def test_audits_return_query_plan_fields():
    llm_client = RewrittenQueryLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={"session": "s", "content": "Atlas 延期原因是数据库回滚风险。", "occurred_at": "2026-05-01T10:00:00Z"},
    )
    recall_response = client.post("/v1/recall", json={"session": "s", "query": "Atlas 为什么延期？"})
    query_id = recall_response.json()["data"]["query_id"]

    audits_response = client.get("/v1/audits", params={"session": "s", "query_id": query_id})

    item = audits_response.json()["data"]["items"][0]
    assert item["query"] == "Atlas 为什么延期？"
    assert item["original_query"] == "Atlas 为什么延期？"
    assert item["query_plan"] == {
        "rewritten_queries": ["Atlas 延期原因", "Atlas 数据库回滚风险"],
    }


def test_audits_return_answerability_diagnostics():
    llm_client = DiagnosticReferenceLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={
            "session": "s",
            "content": "Atlas 项目当前依赖 AuthGate 服务，依赖风险是容器健康检查误判导致发布暂停。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )
    recall_response = client.post("/v1/recall", json={"session": "s", "query": "Atlas 当前依赖哪个服务？"})
    query_id = recall_response.json()["data"]["query_id"]

    audits_response = client.get("/v1/audits", params={"session": "s", "query_id": query_id})

    item = audits_response.json()["data"]["items"][0]
    assert item["answerability"] == "answerable"
    assert item["answerability_reason"] == "候选记忆主体一致，并明确给出当前依赖。"


def test_delete_records_audit_and_scrubs_deleted_content_from_query_audits():
    llm_client = FakeLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    ingest_response = client.post(
        "/v1/ingest",
        json={
            "session": "project:atlas",
            "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )
    memory_id = ingest_response.json()["data"]["memory_id"]
    client.post("/v1/recall", json={"session": "project:atlas", "query": "Atlas 为什么延期？"})

    delete_response = client.post(
        "/v1/delete",
        json={"session": "project:atlas", "memory_ids": [memory_id], "dry_run": False},
    )
    delete_id = delete_response.json()["data"]["delete_id"]

    audits_response = client.get("/v1/audits", params={"session": "project:atlas", "limit": 10})
    items = audits_response.json()["data"]["items"]
    delete_audit = next(item for item in items if item["audit_type"] == "delete")
    query_audit = next(item for item in items if item["audit_type"] == "query")

    assert delete_audit["delete_id"] == delete_id
    assert delete_audit["target"] == {"memory_ids": [memory_id]}
    assert delete_audit["affected_memory_ids"] == [memory_id]
    assert query_audit["final_answer"] is None
    assert query_audit["retrieved"][0]["content_preview"] is None


def test_audits_are_session_isolated_and_support_query_id_lookup():
    llm_client = FakeLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post("/v1/ingest", json={"session": "session:a", "content": "A 项目延期。", "occurred_at": "2026-05-01T10:00:00Z"})
    client.post("/v1/ingest", json={"session": "session:b", "content": "B 项目延期。", "occurred_at": "2026-05-01T10:00:00Z"})
    query_a = client.post("/v1/recall", json={"session": "session:a", "query": "A 为什么延期？"}).json()["data"]["query_id"]
    client.post("/v1/recall", json={"session": "session:b", "query": "B 为什么延期？"})

    session_a_response = client.get("/v1/audits", params={"session": "session:a"})
    assert [item["session"] for item in session_a_response.json()["data"]["items"]] == ["session:a"]

    detail_response = client.get("/v1/audits", params={"session": "session:a", "query_id": query_a})
    assert len(detail_response.json()["data"]["items"]) == 1
    assert detail_response.json()["data"]["items"][0]["query_id"] == query_a

    cross_session_detail = client.get("/v1/audits", params={"session": "session:b", "query_id": query_a})
    assert cross_session_detail.json()["data"]["items"] == []


def test_audits_without_session_returns_all_sessions():
    llm_client = FakeLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post("/v1/ingest", json={"session": "session:a", "content": "A 项目延期。", "occurred_at": "2026-05-01T10:00:00Z"})
    client.post("/v1/ingest", json={"session": "session:b", "content": "B 项目延期。", "occurred_at": "2026-05-02T10:00:00Z"})
    client.post("/v1/recall", json={"session": "session:a", "query": "A 为什么延期？"})
    client.post("/v1/recall", json={"session": "session:b", "query": "B 为什么延期？"})

    audits_response = client.get("/v1/audits")

    assert audits_response.status_code == 200
    sessions = {item["session"] for item in audits_response.json()["data"]["items"]}
    assert sessions == {"session:a", "session:b"}


def test_audits_use_offset_and_return_total():
    llm_client = FakeLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    for index in range(1, 4):
        client.post(
            "/v1/ingest",
            json={
                "session": "session:a",
                "content": f"A 项目第 {index} 条延期复盘。",
                "occurred_at": f"2026-05-0{index}T10:00:00Z",
            },
        )
        client.post("/v1/recall", json={"session": "session:a", "query": f"A 第 {index} 次为什么延期？"})

    audits_response = client.get("/v1/audits", params={"session": "session:a", "limit": 1, "offset": 1})

    assert audits_response.status_code == 200
    data = audits_response.json()["data"]
    assert data["total"] == 3
    assert data["limit"] == 1
    assert data["offset"] == 1
    assert len(data["items"]) == 1


def test_settings_use_memoflux_embedding_configuration(monkeypatch):
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_DIM", "2048")
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_BASE_URL", "http://litellm:4000/v1")

    settings = load_settings()

    assert settings.embedding_provider == "openai_compatible"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 2048
    assert settings.embedding_api_key == "embedding-key"
    assert settings.embedding_base_url == "http://litellm:4000/v1"


def test_retention_cleanup_settings_defaults(monkeypatch):
    monkeypatch.delenv("MEMOFLUX_MEMORY_CLEANUP_ENABLED", raising=False)
    monkeypatch.delenv("MEMOFLUX_MEMORY_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("MEMOFLUX_MEMORY_CLEANUP_HOUR", raising=False)
    monkeypatch.delenv("MEMOFLUX_MEMORY_CLEANUP_MINUTE", raising=False)

    settings = load_settings()

    assert settings.memory_cleanup_enabled is True
    assert settings.memory_retention_days == 180
    assert settings.memory_cleanup_hour == 4
    assert settings.memory_cleanup_minute == 15


def test_app_lifespan_schedules_embedding_prewarm(monkeypatch):
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_PREWARM_ON_STARTUP", "true")
    fake_embedding_service = FakeEmbeddingService()
    app = create_app(repository=MemoryRepository(), embedding_client=fake_embedding_service)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert fake_embedding_service.prewarm_calls == 1


def test_create_app_does_not_start_retention_cleanup_when_disabled(monkeypatch):
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_PREWARM_ON_STARTUP", "false")
    monkeypatch.setenv("MEMOFLUX_MEMORY_CLEANUP_ENABLED", "false")
    created_tasks = []

    async def fake_loop(*args, **kwargs):
        return None

    def fake_create_task(coro):
        created_tasks.append(coro)
        coro.close()

        class FakeTask:
            def cancel(self):
                return None

        return FakeTask()

    monkeypatch.setattr("memoflux.retention._memory_retention_cleanup_loop", fake_loop)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert created_tasks == []


def test_app_lifespan_skips_embedding_prewarm_when_disabled(monkeypatch):
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_PREWARM_ON_STARTUP", "false")
    fake_embedding_service = FakeEmbeddingService()
    app = create_app(repository=MemoryRepository(), embedding_client=fake_embedding_service)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert fake_embedding_service.prewarm_calls == 0


def test_embedding_prewarm_skips_non_local_provider(monkeypatch):
    from memoflux.services.embedding_service import EmbeddingService

    monkeypatch.setenv("MEMOFLUX_EMBEDDING_PREWARM_ON_STARTUP", "true")
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_PROVIDER", "openai_compatible")
    service = EmbeddingService()
    load_calls = []
    monkeypatch.setattr(service, "_load_local_model", lambda: load_calls.append("loaded"))

    service.prewarm_local_model()

    assert load_calls == []


def test_default_embedding_provider_is_local_with_configured_dimension(monkeypatch):
    monkeypatch.delenv("MEMOFLUX_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_DIM", "768")

    settings = load_settings()

    assert settings.embedding_provider == "local"
    assert settings.embedding_dimension == 768


def test_settings_fallback_to_llm_values_for_embedding_client(monkeypatch):
    monkeypatch.delenv("MEMOFLUX_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("MEMOFLUX_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.setenv("MEMOFLUX_LLM_API_KEY", "llm-key")
    monkeypatch.setenv("MEMOFLUX_LLM_BASE_URL", "http://litellm:4000/v1")

    settings = load_settings()

    assert settings.embedding_api_key == "llm-key"
    assert settings.embedding_base_url == "http://litellm:4000/v1"


def test_settings_ignore_configured_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MEMOFLUX_SERVICE_PORT=9123\n", encoding="utf-8")
    monkeypatch.setenv("MEMOFLUX_ENV", str(env_file))
    monkeypatch.delenv("MEMOFLUX_SERVICE_PORT", raising=False)

    settings = load_settings()

    assert settings.service_port == 8020


def test_local_embedding_uses_configured_dimension():
    from memoflux.services.embedding_service import EmbeddingService
    import os
    import types

    os.environ["MEMOFLUX_EMBEDDING_DIM"] = "768"
    service = EmbeddingService()
    service._local_model = types.SimpleNamespace(
        encode=lambda texts, **_: [[0.0] * service.dimension for _ in texts]
    )
    result = service.embed_text("hello")

    assert len(result) == 768


def test_answer_synthesis_output_requires_numeric_confidence():
    from pydantic import ValidationError
    from memoflux.llm_schemas import AnswerSynthesisOutput

    try:
        AnswerSynthesisOutput.model_validate({"answer": "x", "confidence": "high", "used_memory_ids": []})
    except ValidationError:
        return
    raise AssertionError("string confidence should be invalid")


def test_answer_synthesis_output_accepts_used_memory_ids():
    from memoflux.llm_schemas import AnswerSynthesisOutput

    output = AnswerSynthesisOutput.model_validate(
        {"answer": "x", "confidence": 0.8, "used_memory_ids": ["m1"], "uncertainties": ["u"]}
    )

    assert output.confidence == 0.8
    assert output.used_memory_ids == ["m1"]
    assert output.uncertainties == ["u"]


def test_answer_synthesis_output_accepts_relevance_by_id():
    from memoflux.llm_schemas import AnswerSynthesisOutput

    output = AnswerSynthesisOutput.model_validate(
        {
            "answer": "Atlas 延期原因是数据库迁移回滚方案不完整。",
            "confidence": 0.9,
            "used_memory_ids": ["m1"],
            "relevance_by_id": {"m1": "该记忆直接说明 Atlas 延期原因。"},
            "uncertainties": [],
        }
    )

    assert output.relevance_by_id == {"m1": "该记忆直接说明 Atlas 延期原因。"}


def test_answer_synthesis_output_accepts_answerability_diagnostics():
    from memoflux.llm_schemas import AnswerSynthesisOutput

    output = AnswerSynthesisOutput.model_validate(
        {
            "answer": "Atlas 当前依赖 AuthGate 服务。",
            "confidence": 0.9,
            "used_memory_ids": ["m1"],
            "relevance_by_id": {"m1": "该记忆直接说明当前依赖。"},
            "answerability": "answerable",
            "answerability_reason": "候选记忆主体一致，并明确给出当前依赖。",
            "uncertainties": [],
        }
    )

    assert output.answerability == "answerable"
    assert output.answerability_reason == "候选记忆主体一致，并明确给出当前依赖。"


def test_postgres_query_audit_mapping_does_not_require_query_type():
    from memoflux.storage.postgres import _query_audit_from_row

    row = SimpleNamespace(
        query_id="q1",
        session="s",
        original_query="Atlas 为什么延期？",
        rewritten_queries=["Atlas 延期原因"],
        candidate_limit=12,
        retrieved=[],
        selected_memory_ids=[],
        selection_reasons={},
        final_answer="answer",
        status="ok",
        error_stage=None,
        error_message=None,
        created_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
    )

    audit = _query_audit_from_row(row)

    assert audit.original_query == "Atlas 为什么延期？"
    assert audit.rewritten_queries == ["Atlas 延期原因"]
    assert not hasattr(audit, "query_type")


def test_postgres_table_definitions_are_extracted_from_repository_module():
    from memoflux.storage import schema

    assert schema.MemoryRow.__tablename__ == "memories"
    assert schema.UsageRunRow.__tablename__ == "usage_runs"
    assert schema.QueryAuditRow.__tablename__ == "memory_query_audits"
    assert schema.DeleteAuditRow.__tablename__ == "memory_delete_audits"
    assert "memoflux.memories" in schema.Base.metadata.tables


def test_postgres_scrub_uses_session_value_in_query(monkeypatch):
    from memoflux.storage import postgres

    captured = {}

    class FakeScalarResult:
        def all(self):
            return []

    class FakeDbSession:
        def __init__(self, engine):
            self.engine = engine

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def scalars(self, statement):
            captured.update(statement.compile().params)
            return FakeScalarResult()

        def commit(self):
            return None

    repository = postgres.PostgresMemoryRepository.__new__(postgres.PostgresMemoryRepository)
    repository.engine = object()
    monkeypatch.setattr(postgres, "Session", FakeDbSession)

    repository.scrub_deleted_memory_from_audits(session="session:test", memory_ids=["m1"])

    assert captured == {"session_1": "session:test"}
