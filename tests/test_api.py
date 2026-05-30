from memoflux.api import create_app
from memoflux.config import load_settings
from memoflux.llm import LLMResult
from memoflux.storage.memory import MemoryRepository


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = []

    def plan_query(self, *, query: str) -> LLMResult:
        self.calls.append(("plan_query", query))
        return LLMResult(
            output={"query_type": "direct", "rewritten_queries": [query]},
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
            output={"answer": "字符串置信度答案", "confidence": "high", "used_memory_ids": [memory.memory_id for memory in memories]},
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
                "uncertainties": [],
            },
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )



class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


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


def test_ingest_rejects_legacy_scope_field():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/ingest",
        json={"scope": "legacy", "content": "旧字段应被拒绝。", "occurred_at": "2026-05-01T10:00:00Z"},
    )

    assert response.status_code == 422


def test_recall_requires_session_field():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/v1/recall", json={"scope": "legacy", "query": "旧字段能查吗？"})

    assert response.status_code == 422


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
    assert items[0]["query_type"] == "direct"
    assert items[0]["selected_memory_ids"]
    assert items[0]["final_answer"].startswith("LLM 整合答案")
    assert "llm_usage" not in items[0]


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


def test_default_embedding_provider_is_local_with_configured_dimension(monkeypatch):
    monkeypatch.delenv("MEMOFLUX_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("MEMOFLUX_EMBEDDING_DIM", "768")

    settings = load_settings()

    assert settings.embedding_provider == "local"
    assert settings.embedding_dimension == 768


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
