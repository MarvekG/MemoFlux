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
        return LLMResult(
            output={"answer": "LLM 整合答案：" + "；".join(memory.content for memory in memories), "confidence": 0.8},
            input_tokens=20,
            output_tokens=10,
            model="fake",
        )

    def run_prompt(self, *, prompt_key: str, payload: dict) -> LLMResult:
        self.calls.append(("run_prompt", prompt_key))
        return LLMResult(output={"prompt_key": prompt_key, "payload": payload}, input_tokens=3, output_tokens=4, model="fake")



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

    def insert_memory(self, *, scope: str, content: str, occurred_at, embedding=None):
        self.insert_embeddings.append(embedding)
        return super().insert_memory(scope=scope, content=content, occurred_at=occurred_at)

    def search_memories(self, *, scope: str, terms: set[str], limit: int, query_embedding=None):
        self.search_embeddings.append(query_embedding)
        return super().search_memories(scope=scope, terms=terms, limit=limit)


def test_ingest_recall_delete_and_preview_flow():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)

    ingest_response = client.post(
        "/v1/ingest",
        json={
            "scope": "project:atlas",
            "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )
    assert ingest_response.status_code == 200
    memory_id = ingest_response.json()["data"]["memory_id"]

    recall_response = client.post(
        "/v1/recall",
        json={"scope": "project:atlas", "query": "为什么 Atlas 上线被卡住？"},
    )
    assert recall_response.status_code == 200
    recall_data = recall_response.json()["data"]
    assert "数据库迁移回滚方案不完整" in recall_data["answer"]
    assert recall_data["references"][0]["memory_id"] == memory_id
    assert "usage" not in recall_data

    other_scope_response = client.post(
        "/v1/recall",
        json={"scope": "project:zephyr", "query": "为什么 Atlas 上线被卡住？"},
    )
    assert other_scope_response.status_code == 200
    assert other_scope_response.json()["data"]["answer"] == "未找到可用于回答该问题的记忆。"

    preview_response = client.get("/v1/preview", params={"scope": "project:atlas"})
    assert preview_response.status_code == 200
    assert preview_response.json()["data"]["items"][0]["memory_id"] == memory_id

    delete_response = client.post(
        "/v1/delete",
        json={"scope": "project:atlas", "memory_ids": [memory_id], "dry_run": False},
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["data"]["affected_memory_ids"] == [memory_id]

    after_delete_response = client.post(
        "/v1/recall",
        json={"scope": "project:atlas", "query": "为什么 Atlas 上线被卡住？"},
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
            "scope": "project:atlas",
            "content": "5 月 3 日确认需要补充回滚演练记录。",
            "occurred_at": "2026-05-03T09:00:00Z",
        },
    )
    client.post(
        "/v1/ingest",
        json={
            "scope": "project:atlas",
            "content": "5 月 1 日 Atlas 发布延期，因为回滚方案不完整。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )

    response = client.post(
        "/v1/recall",
        json={"scope": "project:atlas", "query": "Atlas 计划之前是怎么变化的？"},
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
        json={"scope": "project:atlas", "query": "删除发布延期细节", "dry_run": False},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


def test_usage_stats_are_aggregate_only():
    app = create_app(repository=MemoryRepository(), embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={"scope": "s", "content": "Atlas 发布延期。", "occurred_at": "2026-05-01T10:00:00Z"},
    )
    client.post("/v1/recall", json={"scope": "s", "query": "Atlas 为什么延期？"})

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
        json={"scope": "s", "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。", "occurred_at": "2026-05-01T10:00:00Z"},
    )
    response = client.post("/v1/recall", json={"scope": "s", "query": "Atlas 为什么延期？"})

    data = response.json()["data"]
    assert response.status_code == 200
    assert data["answer"].startswith("LLM 整合答案")
    assert "usage" not in data
    assert [call[0] for call in llm_client.calls] == ["plan_query", "synthesize_answer"]


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
        json={"scope": "s", "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。", "occurred_at": "2026-05-01T10:00:00Z"},
    )
    response = client.post("/v1/recall", json={"scope": "s", "query": "Atlas 为什么延期？"})

    assert response.status_code == 200
    assert repository.insert_embeddings == [[0.1, 0.2, 0.3]]
    assert repository.search_embeddings == [[0.1, 0.2, 0.3]]
    assert fake_embedding_service.calls == ["Atlas 发布延期，因为数据库迁移回滚方案不完整。", "Atlas 为什么延期？"]
    assert [call[0] for call in llm_client.calls] == ["plan_query", "synthesize_answer"]


def test_audits_return_persisted_recall_records():
    llm_client = FakeLLMClient()
    app = create_app(repository=MemoryRepository(), llm_client=llm_client, embedding_client=FakeEmbeddingService())
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post(
        "/v1/ingest",
        json={
            "scope": "project:atlas",
            "content": "Atlas 发布延期，因为数据库迁移回滚方案不完整。",
            "occurred_at": "2026-05-01T10:00:00Z",
        },
    )
    recall_response = client.post("/v1/recall", json={"scope": "project:atlas", "query": "Atlas 为什么延期？"})
    query_id = recall_response.json()["data"]["query_id"]

    audits_response = client.get("/v1/audits", params={"scope": "project:atlas"})

    assert audits_response.status_code == 200
    items = audits_response.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["audit_type"] == "query"
    assert items[0]["query_id"] == query_id
    assert items[0]["scope"] == "project:atlas"
    assert items[0]["original_query"] == "Atlas 为什么延期？"
    assert items[0]["query_type"] == "direct"
    assert items[0]["selected_memory_ids"]
    assert items[0]["final_answer"].startswith("LLM 整合答案")
    assert "llm_usage" not in items[0]


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
