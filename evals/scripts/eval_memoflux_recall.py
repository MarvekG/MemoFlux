from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

NO_EVIDENCE_ANSWER = "当前 session 中没有足够记忆支持回答该问题。"


def load_suite(path: Path) -> dict[str, Any]:
    """加载并校验 MemoFlux recall JSON 评测套。

    Args:
        path: JSON 评测套路径。

    Returns:
        校验后的评测套字典。

    Raises:
        ValueError: 评测套缺少必要字段或字段类型不正确。
    """

    suite = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("suite_id", "memories", "queries") if key not in suite]
    if missing:
        raise ValueError(f"suite missing required sections: {', '.join(missing)}")
    if not isinstance(suite["memories"], list):
        raise ValueError("suite memories must be a list")
    if not isinstance(suite["queries"], list):
        raise ValueError("suite queries must be a list")
    return suite


def evaluate_recall(
    query_case: dict[str, Any],
    recall_data: dict[str, Any],
    memory_id_map: dict[str, str],
    *,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """评估单条 recall 结果。

    Args:
        query_case: JSON 评测套中的查询用例。
        recall_data: `/v1/recall` 返回的 data 字段。
        memory_id_map: 评测 memory case id 到实际 memory_id 的映射。
        audit: 可选 query audit，用于计算 retrieval/selection 分阶段命中。

    Returns:
        包含正确性、噪声和缺失项的评估结果。
    """

    answer = str(recall_data.get("answer") or "")
    references = list(recall_data.get("references") or [])
    reference_ids = {str(reference.get("memory_id")) for reference in references}
    expected_reference_ids = _case_ids_to_memory_ids(query_case.get("expected_reference_memory_ids", []), memory_id_map)
    acceptable_reference_ids = _case_ids_to_memory_ids(
        query_case.get("acceptable_reference_memory_ids") or query_case.get("expected_reference_memory_ids", []),
        memory_id_map,
    )
    forbidden_reference_ids = _case_ids_to_memory_ids(query_case.get("forbidden_reference_memory_ids", []), memory_id_map)
    missing_answer_terms = [term for term in query_case.get("expected_answer_contains", []) if str(term) not in answer]
    forbidden_answer_terms = [term for term in query_case.get("forbidden_answer_contains", []) if str(term) in answer]
    missing_reference_ids = [memory_id for memory_id in expected_reference_ids if memory_id not in reference_ids]
    noisy_reference_ids = [memory_id for memory_id in forbidden_reference_ids if memory_id in reference_ids]
    retrieved_ids = {str(item.get("memory_id")) for item in (audit or {}).get("retrieved", [])}
    selected_ids = {str(memory_id) for memory_id in (audit or {}).get("selected_memory_ids", [])}
    retrieval_hit = _any_expected_hit(acceptable_reference_ids, retrieved_ids) if audit is not None else None
    selection_hit = _any_expected_hit(acceptable_reference_ids, selected_ids) if audit is not None else None
    missing_retrieved_ids = [] if retrieval_hit in {None, True} else acceptable_reference_ids
    missing_selected_ids = [] if selection_hit in {None, True} else acceptable_reference_ids
    strict_correct = not missing_answer_terms and not missing_reference_ids
    no_evidence_correct = None
    if query_case.get("expect_no_evidence"):
        no_evidence_correct = (
            NO_EVIDENCE_ANSWER in answer
            and not references
            and _safe_float(recall_data.get("confidence")) <= 0.3
        )
    answer_correct = bool(no_evidence_correct) if no_evidence_correct is not None else not missing_answer_terms
    answer_noise = bool(forbidden_answer_terms)
    ref_noise = bool(noisy_reference_ids)
    return {
        "query_id": query_case.get("id"),
        "kind": query_case.get("kind", "unknown"),
        "correct": answer_correct and not answer_noise and not ref_noise,
        "strict_correct": strict_correct and not answer_noise and not ref_noise,
        "answer_correct": answer_correct,
        "no_evidence_correct": no_evidence_correct,
        "answer_noise": answer_noise,
        "ref_noise": ref_noise,
        "missing_answer_terms": missing_answer_terms,
        "forbidden_answer_terms": forbidden_answer_terms,
        "missing_reference_case_ids": _memory_ids_to_case_ids(missing_reference_ids, memory_id_map),
        "missing_strict_reference_case_ids": _memory_ids_to_case_ids(missing_reference_ids, memory_id_map),
        "noisy_reference_case_ids": _memory_ids_to_case_ids(noisy_reference_ids, memory_id_map),
        "retrieval_hit": retrieval_hit,
        "selection_hit": selection_hit,
        "missing_retrieved_case_ids": _memory_ids_to_case_ids(missing_retrieved_ids, memory_id_map),
        "missing_selected_case_ids": _memory_ids_to_case_ids(missing_selected_ids, memory_id_map),
        "answer": answer,
        "confidence": recall_data.get("confidence"),
        "reference_count": len(references),
    }


def run_suite(*, suite_path: Path, base_url: str, session: str | None = None) -> dict[str, Any]:
    """执行 MemoFlux recall JSON 评测套。

    Args:
        suite_path: JSON 评测套路径。
        base_url: MemoFlux 服务地址。
        session: 可选 session；未提供时自动生成。

    Returns:
        聚合后的评测结果。
    """

    suite = load_suite(suite_path)
    eval_session = session or f"eval:{suite['suite_id']}:{int(time.time())}"
    memory_id_map: dict[str, str] = {}
    for memory_case in suite["memories"]:
        payload = _render_templates(memory_case, eval_session)
        status, body = _request(base_url, "POST", "/v1/ingest", payload)
        if status != 200:
            raise RuntimeError(f"ingest failed for {memory_case.get('id')}: {body}")
        memory_id_map[str(memory_case["id"])] = str(body["data"]["memory_id"])

    details = []
    for query_case in suite["queries"]:
        payload = build_recall_payload(
            query_case,
            session=eval_session,
            top_k=int(query_case.get("top_k") or suite.get("defaults", {}).get("top_k") or 12),
        )
        status, body = _request(base_url, "POST", "/v1/recall", payload)
        if status != 200:
            raise RuntimeError(f"recall failed for {query_case.get('id')}: {body}")
        query_id = str(body["data"].get("query_id") or "")
        audit = _load_query_audit(base_url=base_url, session=eval_session, query_id=query_id)
        details.append(evaluate_recall(query_case, body["data"], memory_id_map, audit=audit))
    metrics = _aggregate(details)
    return {
        "suite_id": suite["suite_id"],
        "session": eval_session,
        "inserted_count": len(suite["memories"]),
        "query_count": len(suite["queries"]),
        "metrics": metrics,
        "details": details,
    }


def main() -> None:
    """命令行入口，执行 JSON 评测套并打印 JSON 结果。"""

    parser = argparse.ArgumentParser(description="Run MemoFlux recall eval suite")
    parser.add_argument("--suite", required=True, type=Path, help="Path to eval suite JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:8020", help="MemoFlux base URL")
    parser.add_argument("--session", default=None, help="Optional eval session")
    parser.add_argument("--only-failures", action="store_true", help="Only include failed details in the printed report")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the JSON report")
    args = parser.parse_args()
    result = run_suite(suite_path=args.suite, base_url=args.base_url, session=args.session)
    report = build_report_output(result, only_failures=args.only_failures)
    if args.output is not None:
        write_report(report, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_report_output(result: dict[str, Any], *, only_failures: bool) -> dict[str, Any]:
    """构造命令行输出报告，可按需仅保留失败明细。

    Args:
        result: `run_suite` 返回的完整评测结果。
        only_failures: 为 True 时仅保留 `correct` 为 False 的明细。

    Returns:
        用于打印或写文件的 JSON 可序列化报告。
    """

    if not only_failures:
        return result
    report = dict(result)
    report["details"] = [detail for detail in result.get("details", []) if not bool(detail.get("correct"))]
    return report


def write_report(report: dict[str, Any], *, output_path: Path) -> None:
    """将评测报告写入 JSON 文件。

    Args:
        report: JSON 可序列化评测报告。
        output_path: 输出文件路径，父目录不存在时会自动创建。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_recall_payload(query_case: dict[str, Any], *, session: str, top_k: int) -> dict[str, Any]:
    """从评测查询用例构造 `/v1/recall` 请求体。

    Args:
        query_case: JSON 评测套中的查询用例。
        session: 本次评测使用的 session。
        top_k: recall 请求的 top_k。

    Returns:
        只包含 API 允许字段的请求体。
    """

    return {
        "session": str(query_case["session"]).replace("{{session}}", session),
        "query": str(query_case["query"]),
        "top_k": top_k,
    }


def _request(base_url: str, method: str, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = base_url.rstrip("/") + path
    if method == "GET" and payload:
        url += "?" + urllib.parse.urlencode(payload)
        data = None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _load_query_audit(*, base_url: str, session: str, query_id: str) -> dict[str, Any] | None:
    """读取单条 query audit。

    Args:
        base_url: MemoFlux 服务地址。
        session: 当前评测 session。
        query_id: recall 响应中的 query_id。

    Returns:
        找到时返回 query audit，否则返回 None。
    """

    if not query_id:
        return None
    status, body = _request(base_url, "GET", "/v1/audits", {"session": session, "query_id": query_id})
    if status != 200:
        return None
    items = body.get("data", {}).get("items", [])
    return items[0] if items else None


def _render_templates(value: Any, session: str) -> Any:
    if isinstance(value, dict):
        return {key: _render_templates(item, session) for key, item in value.items() if key != "id" and key != "kind"}
    if isinstance(value, list):
        return [_render_templates(item, session) for item in value]
    if isinstance(value, str):
        return value.replace("{{session}}", session)
    return value


def _case_ids_to_memory_ids(case_ids: list[str], memory_id_map: dict[str, str]) -> list[str]:
    return [memory_id_map[case_id] for case_id in case_ids if case_id in memory_id_map]


def _memory_ids_to_case_ids(memory_ids: list[str], memory_id_map: dict[str, str]) -> list[str]:
    by_memory_id = {memory_id: case_id for case_id, memory_id in memory_id_map.items()}
    return [by_memory_id.get(memory_id, memory_id) for memory_id in memory_ids]


def _any_expected_hit(expected_ids: list[str], actual_ids: set[str]) -> bool:
    """判断期望证据集合是否被命中。

    Args:
        expected_ids: 可接受的期望 memory_id 列表。
        actual_ids: 实际召回或选择的 memory_id 集合。

    Returns:
        没有配置期望证据时返回 True；否则任意一个期望证据命中即返回 True。
    """

    return True if not expected_ids else bool(set(expected_ids) & actual_ids)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _aggregate(details: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, int]] = {}
    for detail in details:
        kind = str(detail["kind"])
        by_kind.setdefault(kind, {"total": 0, "correct": 0, "answer_correct": 0, "strict_correct": 0})
        by_kind[kind]["total"] += 1
        by_kind[kind]["correct"] += int(bool(detail["correct"]))
        by_kind[kind]["answer_correct"] += int(bool(detail["answer_correct"]))
        by_kind[kind]["strict_correct"] += int(bool(detail["strict_correct"]))
    return {
        "overall": {
            "total": len(details),
            "correct": sum(int(bool(detail["correct"])) for detail in details),
            "answer_correct": sum(int(bool(detail["answer_correct"])) for detail in details),
            "strict_correct": sum(int(bool(detail["strict_correct"])) for detail in details),
        },
        "by_kind": by_kind,
        "answer_noise_count": sum(int(bool(detail["answer_noise"])) for detail in details),
        "ref_noise_count": sum(int(bool(detail["ref_noise"])) for detail in details),
        "no_evidence_correct": sum(int(detail["no_evidence_correct"] is True) for detail in details),
        "retrieval_hit_count": sum(int(detail["retrieval_hit"] is True) for detail in details),
        "selection_hit_count": sum(int(detail["selection_hit"] is True) for detail in details),
    }


if __name__ == "__main__":
    main()
