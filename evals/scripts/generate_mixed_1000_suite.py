from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECTS = [
    "Atlas", "Zephyr", "Nova", "Orion", "Helios", "Lyra", "Vega", "Aster", "Nimbus", "Quartz",
    "Falcon", "Harbor", "Iris", "Juno", "Kite", "Lumen", "Mira", "Nexus", "Opal", "Pioneer",
    "Raven", "Solace", "Triton", "Umbra", "Vertex", "Willow", "Xeno", "Yarrow", "Zenith", "Beryl",
    "Cedar", "Delta", "Ember", "Flora", "Garnet", "Horizon", "Ivory", "Jasper", "Kepler", "Lotus",
    "Matrix", "Neon", "Oasis", "Praxis", "Quasar", "Ripple", "Sierra", "Tempest", "Union", "Voyager",
]
REASONS = [
    "数据库迁移回滚方案不完整", "前端构建缓存策略错误", "供应商接口限流", "订单索引回填校验失败",
    "灰度发布监控指标缺失", "权限模型变更未完成安全评审", "消息队列消费延迟超过阈值", "风控参数同步任务失败",
    "行情数据清洗规则冲突", "容器健康检查误判导致发布暂停", "异步任务重试策略过于激进", "缓存预热脚本缺少幂等保护",
    "报表口径与财务确认口径不一致", "跨区网络抖动影响数据同步", "依赖服务版本锁定失败", "特征计算窗口配置错误",
    "审计日志字段缺失", "任务队列积压未及时告警", "加密证书轮换流程未验证", "发布审批链路超时",
]
OWNERS = ["王晨", "李然", "赵宁", "陈雅", "周航", "林越", "沈澈", "韩序", "顾远", "唐清"]
DEPS = ["RiskCore", "QuoteHub", "OrderFlow", "AuthGate", "DataBridge", "ReportLab", "SignalBox", "TradeSync"]


def build_suite(seed: int = 20260531) -> dict:
    """构造可复现的 1000 条 MemoFlux mixed recall 评测套。

    Args:
        seed: 控制 query 抽样和顺序的随机种子。

    Returns:
        可写入 JSON 文件的评测套。
    """

    random.seed(seed)
    truth = {
        project: {
            "latest_delay_id": None,
            "latest_delay_reason": None,
            "latest_delay_ids_by_reason": {},
            "owner_id": None,
            "owner": None,
            "history_ids": [],
            "current_association_id": None,
            "current_association_dep": None,
            "current_association_reason": None,
            "association_history_ids": [],
            "association_history_deps": [],
        }
        for project in PROJECTS
    }
    memories = []
    for round_index in range(20):
        for project_index, project in enumerate(PROJECTS):
            index = round_index * len(PROJECTS) + project_index
            reason = REASONS[(index + round_index) % len(REASONS)]
            owner = OWNERS[(project_index + round_index) % len(OWNERS)]
            dep = DEPS[(project_index + round_index) % len(DEPS)]
            day = round_index + 1
            hour = 8 + (index % 10)
            kind = round_index % 10
            memory_id = f"m_{index:04d}"
            if kind == 0:
                content = f"{project} 项目延期，原因是{reason}。"
                truth[project]["latest_delay_id"] = memory_id
                truth[project]["latest_delay_reason"] = reason
                truth[project]["latest_delay_ids_by_reason"].setdefault(reason, []).append(memory_id)
            elif kind == 1:
                content = f"{project} 项目负责人是{owner}。"
                truth[project]["owner_id"] = memory_id
                truth[project]["owner"] = owner
            elif kind == 2:
                content = f"{project} 项目依赖 {dep} 服务，依赖风险是{reason}。"
                truth[project]["association_history_ids"].append(memory_id)
                truth[project]["association_history_deps"].append(dep)
                truth[project]["current_association_id"] = memory_id
                truth[project]["current_association_dep"] = dep
                truth[project]["current_association_reason"] = reason
            elif kind == 3:
                content = f"{project} 项目发布历史：第 {len(truth[project]['history_ids']) + 1} 次发布因{reason}暂停。"
                truth[project]["history_ids"].append(memory_id)
            elif kind == 4:
                content = f"{project} 项目恢复推进，前置条件是{reason}。"
            elif kind == 5:
                content = f"{project} 项目完成发布前检查，风险点是{reason}。"
            elif kind == 6:
                content = f"{project} 项目上线后复盘，主要改进项是{reason}。"
            elif kind == 7:
                content = f"{project} 项目暂停发布，阻塞项是{reason}。"
            elif kind == 8:
                corrected = REASONS[(index + 3) % len(REASONS)]
                content = f"{project} 项目延期原因纠正为{corrected}，之前记录的{reason}不再作为当前判断依据。"
                truth[project]["latest_delay_id"] = memory_id
                truth[project]["latest_delay_reason"] = corrected
                truth[project]["latest_delay_ids_by_reason"].setdefault(corrected, []).append(memory_id)
            else:
                content = f"{project} 项目当前依赖 {dep} 服务，依赖风险是{reason}，之前依赖记录不再作为当前判断依据。"
                truth[project]["association_history_ids"].append(memory_id)
                truth[project]["association_history_deps"].append(dep)
                truth[project]["current_association_id"] = memory_id
                truth[project]["current_association_dep"] = dep
                truth[project]["current_association_reason"] = reason
            memories.append(
                {
                    "id": memory_id,
                    "session": "{{session}}",
                    "content": content,
                    "occurred_at": f"2026-05-{day:02d}T{hour:02d}:00:00Z",
                }
            )
    queries = []
    for project in random.sample(PROJECTS, 30):
        latest_reason = truth[project]["latest_delay_reason"]
        queries.append(
            {
                "id": f"latest_delay_{project}",
                "session": "{{session}}",
                "kind": "latest_delay",
                "query": f"{project} 当前延期原因是什么？",
                "expected_answer_contains": [latest_reason],
                "expected_reference_memory_ids": [truth[project]["latest_delay_id"]],
                "acceptable_reference_memory_ids": truth[project]["latest_delay_ids_by_reason"].get(latest_reason, []),
            }
        )
    for project in random.sample(PROJECTS, 20):
        expected_ids = truth[project]["history_ids"]
        queries.append(
            {
                "id": f"history_{project}",
                "session": "{{session}}",
                "kind": "history",
                "query": f"按历史记录总结 {project} 发布暂停原因。",
                "expected_reference_memory_ids": expected_ids[:1],
            }
        )
    for project in random.sample(PROJECTS, 15):
        queries.append(
            {
                "id": f"owner_{project}",
                "session": "{{session}}",
                "kind": "owner",
                "query": f"{project} 项目负责人是谁？",
                "expected_answer_contains": [truth[project]["owner"]],
                "expected_reference_memory_ids": [truth[project]["owner_id"]],
            }
        )
    for project in random.sample(PROJECTS, 15):
        queries.append(
            {
                "id": f"current_association_{project}",
                "session": "{{session}}",
                "kind": "current_association",
                "query": f"{project} 当前依赖哪个服务，当前依赖风险是什么？",
                "expected_answer_contains": [
                    truth[project]["current_association_dep"],
                    truth[project]["current_association_reason"],
                ],
                "expected_reference_memory_ids": [truth[project]["current_association_id"]],
            }
        )
    for project in random.sample(PROJECTS, 10):
        queries.append(
            {
                "id": f"association_history_{project}",
                "session": "{{session}}",
                "kind": "association_history",
                "query": f"按历史记录总结 {project} 依赖过哪些服务。",
                "expected_answer_contains": sorted(set(truth[project]["association_history_deps"])),
                "expected_reference_memory_ids": truth[project]["association_history_ids"],
            }
        )
    for project in ["Apollo", "Borealis", "Cygnus", "Draco", "Equinox", "Fenix", "Gemini", "Hydra", "Ion", "Janus"]:
        queries.append(
            {
                "id": f"no_evidence_{project}",
                "session": "{{session}}",
                "kind": "no_evidence",
                "query": f"{project} 当前延期原因是什么？",
                "expect_no_evidence": True,
                "forbidden_answer_contains": PROJECTS,
            }
        )
    random.shuffle(queries)
    return {
        "suite_id": "mixed_1000",
        "description": "Single-session 1000-memory MemoFlux recall suite covering latest/correction, history, owner, current association, association history, and no-evidence cases.",
        "defaults": {"top_k": 12},
        "memories": memories,
        "queries": queries,
    }


def main() -> None:
    """命令行入口，生成 mixed_1000 JSON 评测套。"""

    parser = argparse.ArgumentParser(description="Generate MemoFlux mixed_1000 eval suite")
    parser.add_argument("--output", type=Path, default=Path("evals/cases/mixed_1000.json"))
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()
    suite = build_suite(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
