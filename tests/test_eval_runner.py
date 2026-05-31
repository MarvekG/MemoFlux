import json
import sys
from pathlib import Path
from unittest.mock import patch

from evals.scripts import eval_memoflux_recall
from evals.scripts.eval_memoflux_recall import build_report_output, build_recall_payload, evaluate_recall, load_suite, write_report
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


def test_evaluate_recall_accepts_equivalent_reference_ids():
    memory_id_map = {
        "expected_correction": "m-expected",
        "equivalent_correction": "m-equivalent",
    }
    query_case = {
        "id": "latest_delay",
        "kind": "latest_delay",
        "expected_answer_contains": ["行情数据清洗规则冲突"],
        "expected_reference_memory_ids": ["expected_correction"],
        "acceptable_reference_memory_ids": ["expected_correction", "equivalent_correction"],
    }
    recall_data = {
        "answer": "行情数据清洗规则冲突",
        "confidence": 0.95,
        "references": [{"memory_id": "m-equivalent", "quote": "等价纠错记忆"}],
    }
    audit = {
        "retrieved": [{"memory_id": "m-equivalent", "content_preview": "等价纠错记忆"}],
        "selected_memory_ids": ["m-equivalent"],
    }

    result = evaluate_recall(query_case, recall_data, memory_id_map, audit=audit)

    assert result["selection_hit"] is True
    assert result["strict_correct"] is False
    assert result["correct"] is True
    assert result["missing_selected_case_ids"] == []
    assert result["missing_strict_reference_case_ids"] == ["expected_correction"]


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


def test_build_report_output_can_filter_to_failures_only():
    result = {
        "suite_id": "suite",
        "session": "session",
        "metrics": {"overall": {"total": 2, "correct": 1}},
        "details": [
            {"query_id": "ok", "correct": True, "answer": "ok"},
            {"query_id": "failed", "correct": False, "answer": "当前 session 中没有足够记忆支持回答该问题。"},
        ],
    }

    output = build_report_output(result, only_failures=True)

    assert output["suite_id"] == "suite"
    assert output["session"] == "session"
    assert output["metrics"] == {"overall": {"total": 2, "correct": 1}}
    assert output["details"] == [
        {"query_id": "failed", "correct": False, "answer": "当前 session 中没有足够记忆支持回答该问题。"}
    ]


def test_write_report_writes_json_file(tmp_path):
    report_path = tmp_path / "report.json"
    result = {"suite_id": "suite", "details": []}

    write_report(result, output_path=report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == result


def test_main_supports_only_failures_and_output_file(tmp_path, capsys):
    report_path = tmp_path / "reports" / "failures.json"
    result = {
        "suite_id": "suite",
        "session": "session",
        "details": [
            {"query_id": "ok", "correct": True},
            {"query_id": "failed", "correct": False},
        ],
    }

    with patch.object(eval_memoflux_recall, "run_suite", return_value=result), patch.object(
        sys,
        "argv",
        [
            "eval_memoflux_recall.py",
            "--suite",
            "evals/cases/smoke_small.json",
            "--only-failures",
            "--output",
            str(report_path),
        ],
    ):
        eval_memoflux_recall.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert printed["details"] == [{"query_id": "failed", "correct": False}]
    assert written == printed


def test_smoke_small_suite_is_valid():
    suite = load_suite(Path("evals/cases/smoke_small.json"))

    assert suite["suite_id"] == "smoke_small"
    assert len(suite["memories"]) == 4
    assert len(suite["queries"]) == 3


def test_build_mixed_1000_suite_is_reproducible():
    suite = build_suite(seed=20260531)

    assert suite["suite_id"] == "mixed_1000"
    assert len(suite["memories"]) == 1000
    assert len(suite["queries"]) == 100
    latest_delay_cases = [query for query in suite["queries"] if query["kind"] == "latest_delay"]
    assert all("acceptable_reference_memory_ids" in query for query in latest_delay_cases)
    assert suite == build_suite(seed=20260531)


def test_build_mixed_1000_suite_splits_current_and_history_association_cases():
    suite = build_suite(seed=20260531)
    memories_by_id = {memory["id"]: memory for memory in suite["memories"]}
    current_cases = [query for query in suite["queries"] if query["kind"] == "current_association"]
    history_cases = [query for query in suite["queries"] if query["kind"] == "association_history"]

    assert len(current_cases) == 15
    assert len(history_cases) == 10
    for query in current_cases:
        assert query["expected_answer_contains"]
        assert len(query["expected_reference_memory_ids"]) == 1
        expected_memory = memories_by_id[query["expected_reference_memory_ids"][0]]
        assert "当前依赖" in query["query"]
        assert "当前判断依据" in expected_memory["content"]

    for query in history_cases:
        assert len(query["expected_reference_memory_ids"]) >= 2
        assert "历史" in query["query"]
        assert all("依赖" in memories_by_id[memory_id]["content"] for memory_id in query["expected_reference_memory_ids"])


def test_build_mixed_1000_latest_delay_timestamps_follow_truth_order():
    suite = build_suite(seed=20260531)
    memories_by_id = {memory["id"]: memory for memory in suite["memories"]}

    for query in suite["queries"]:
        if query["kind"] != "latest_delay":
            continue
        project = query["id"].removeprefix("latest_delay_")
        expected_id = query["expected_reference_memory_ids"][0]
        expected_at = memories_by_id[expected_id]["occurred_at"]
        project_delay_memories = [
            memory
            for memory in suite["memories"]
            if memory["content"].startswith(f"{project} 项目延期，")
            or memory["content"].startswith(f"{project} 项目延期原因纠正为")
        ]

        assert all(memory["occurred_at"] <= expected_at for memory in project_delay_memories)
