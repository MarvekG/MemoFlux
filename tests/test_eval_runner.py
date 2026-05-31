import json
from pathlib import Path

from evals.scripts.eval_memoflux_recall import build_recall_payload, evaluate_recall, load_suite
from evals.scripts.generate_mixed_1000_suite import build_suite


def test_load_suite_rejects_missing_required_sections(tmp_path):
    suite_path = tmp_path / "invalid.json"
    suite_path.write_text(json.dumps({"suite_id": "invalid"}), encoding="utf-8")

    try:
        load_suite(suite_path)
    except ValueError as exc:
        assert "memories" in str(exc)
        assert "queries" in str(exc)
        return
    raise AssertionError("invalid suite should be rejected")


def test_evaluate_recall_counts_correctness_and_noise():
    memory_id_map = {
        "atlas_corrected_delay": "m-atlas-corrected",
        "zephyr_delay": "m-zephyr",
    }
    query_case = {
        "id": "atlas_latest_delay",
        "kind": "latest_delay",
        "expected_answer_contains": ["权限模型变更未完成安全评审"],
        "expected_reference_memory_ids": ["atlas_corrected_delay"],
        "forbidden_answer_contains": ["Zephyr"],
        "forbidden_reference_memory_ids": ["zephyr_delay"],
    }
    recall_data = {
        "answer": "权限模型变更未完成安全评审。",
        "confidence": 0.95,
        "references": [
            {
                "memory_id": "m-atlas-corrected",
                "quote": "Atlas 项目延期原因纠正为权限模型变更未完成安全评审。",
            }
        ],
    }

    result = evaluate_recall(query_case, recall_data, memory_id_map)

    assert result["correct"] is True
    assert result["answer_correct"] is True
    assert result["answer_noise"] is False
    assert result["ref_noise"] is False
    assert result["missing_answer_terms"] == []
    assert result["missing_reference_case_ids"] == []


def test_evaluate_recall_reports_retrieval_and_selection_hits():
    memory_id_map = {
        "atlas_corrected_delay": "m-atlas-corrected",
        "zephyr_delay": "m-zephyr",
    }
    query_case = {
        "id": "atlas_latest_delay",
        "kind": "latest_delay",
        "expected_reference_memory_ids": ["atlas_corrected_delay"],
    }
    recall_data = {
        "answer": "权限模型变更未完成安全评审。",
        "confidence": 0.95,
        "references": [
            {
                "memory_id": "m-atlas-corrected",
                "quote": "Atlas 项目延期原因纠正为权限模型变更未完成安全评审。",
            }
        ],
    }
    audit = {
        "retrieved": [
            {"memory_id": "m-atlas-corrected", "content_preview": "Atlas 项目延期原因纠正为权限模型变更未完成安全评审。"},
            {"memory_id": "m-zephyr", "content_preview": "Zephyr 项目延期。"},
        ],
        "selected_memory_ids": ["m-atlas-corrected"],
    }

    result = evaluate_recall(query_case, recall_data, memory_id_map, audit=audit)

    assert result["retrieval_hit"] is True
    assert result["selection_hit"] is True
    assert result["missing_retrieved_case_ids"] == []
    assert result["missing_selected_case_ids"] == []


def test_evaluate_recall_keeps_answer_correct_separate_from_selection_hit():
    memory_id_map = {
        "expected_correction": "m-expected",
        "equivalent_correction": "m-equivalent",
    }
    query_case = {
        "id": "latest_delay",
        "kind": "latest_delay",
        "expected_answer_contains": ["行情数据清洗规则冲突"],
        "expected_reference_memory_ids": ["expected_correction"],
    }
    recall_data = {
        "answer": "行情数据清洗规则冲突",
        "confidence": 0.95,
        "references": [{"memory_id": "m-equivalent", "quote": "等价纠错记忆"}],
    }

    result = evaluate_recall(query_case, recall_data, memory_id_map)

    assert result["answer_correct"] is True
    assert result["selection_hit"] is None
    assert result["correct"] is True
    assert result["missing_reference_case_ids"] == ["expected_correction"]


def test_evaluate_recall_validates_no_evidence_response():
    query_case = {
        "id": "orion_no_evidence",
        "kind": "no_evidence",
        "expect_no_evidence": True,
        "forbidden_answer_contains": ["Atlas"],
    }
    recall_data = {
        "answer": "当前 session 中没有足够记忆支持回答该问题。",
        "confidence": 0.1,
        "references": [],
    }

    result = evaluate_recall(query_case, recall_data, {})

    assert result["correct"] is True
    assert result["no_evidence_correct"] is True


def test_build_recall_payload_excludes_eval_assertion_fields():
    query_case = {
        "id": "atlas_latest_delay",
        "session": "{{session}}",
        "kind": "latest_delay",
        "query": "Atlas 当前延期原因是什么？",
        "expected_answer_contains": ["权限模型变更未完成安全评审"],
        "expected_reference_memory_ids": ["atlas_corrected_delay"],
        "forbidden_answer_contains": ["Zephyr"],
        "forbidden_reference_memory_ids": ["zephyr_delay"],
    }

    payload = build_recall_payload(query_case, session="eval:session", top_k=6)

    assert payload == {
        "session": "eval:session",
        "query": "Atlas 当前延期原因是什么？",
        "top_k": 6,
    }


def test_smoke_small_suite_is_valid():
    suite = load_suite(Path("evals/cases/smoke_small.json"))

    assert suite["suite_id"] == "smoke_small"
    assert len(suite["memories"]) == 4
    assert len(suite["queries"]) == 3


def test_build_mixed_1000_suite_is_reproducible():
    suite = build_suite(seed=20260531)

    assert suite["suite_id"] == "mixed_1000"
    assert len(suite["memories"]) == 1000
    assert len(suite["queries"]) == 75
    assert suite == build_suite(seed=20260531)
