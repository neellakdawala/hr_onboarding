from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from evals.cases import CASES, USER, EvalCase
from app.agent.graph import run_agent_interactive


HERE = Path(__file__).resolve().parent


@dataclass
class CaseResult:
    id: str
    category: str
    intent_ok: bool
    terminal_ok: bool
    tool_ok: Optional[bool]     # None if not applicable
    keywords_ok: bool
    safety_ok: bool
    overall_ok: bool
    latency_seconds: float
    actual_intent: str
    actual_terminal: str
    actual_tool: Optional[str]
    answer: str


def _terminal_of(path: list[str]) -> str:
    """Which terminal node did the graph land at?"""
    for node in reversed(path):
        if node in {"generate_answer", "escalate", "tool_answer"}:
            return node
    return path[-1] if path else ""


def _tool_of(state) -> Optional[str]:
    tool_calls = state.get("tool_calls") or []
    return tool_calls[0]["name"] if tool_calls else None


def score_case(case: EvalCase) -> CaseResult:
    """Run one case end-to-end and score it."""
    start = time.perf_counter()
    state = run_agent_interactive(
        case.question,
        approve_writes=case.approve_writes,
        current_user=USER,
        thread_id=f"eval-{case.id}",
    )
    latency = time.perf_counter() - start

    actual_intent = state.get("intent", "")
    actual_terminal = _terminal_of(state.get("path", []))
    actual_tool = _tool_of(state)
    answer = state.get("answer", "") or ""
    lower_answer = answer.lower()

    intent_ok = actual_intent == case.expected_intent
    terminal_ok = actual_terminal == case.expected_terminal

    if case.expected_tool is not None:
        tool_ok = actual_tool == case.expected_tool
    else:
        tool_ok = None

    keywords_ok = all(kw.lower() in lower_answer for kw in case.must_include)
    safety_ok = all(
        kw.lower() not in lower_answer for kw in case.must_not_include
    )

    # Overall: every applicable check must pass.
    checks = [intent_ok, terminal_ok, keywords_ok, safety_ok]
    if tool_ok is not None:
        checks.append(tool_ok)
    overall_ok = all(checks)

    return CaseResult(
        id=case.id, category=case.category,
        intent_ok=intent_ok, terminal_ok=terminal_ok, tool_ok=tool_ok,
        keywords_ok=keywords_ok, safety_ok=safety_ok, overall_ok=overall_ok,
        latency_seconds=round(latency, 2),
        actual_intent=actual_intent, actual_terminal=actual_terminal,
        actual_tool=actual_tool, answer=answer[:200],
    )


def print_report(results: list[CaseResult]) -> None:
    print("\n" + "=" * 76)
    print("EVAL REPORT")
    print("=" * 76)
    print(f"{'ID':<26} {'CAT':<16} {'RESULT':<8} {'LAT':<6} NOTES")
    print("-" * 76)
    for r in results:
        result = "PASS" if r.overall_ok else "FAIL"
        notes = []
        if not r.intent_ok:
            notes.append(f"intent={r.actual_intent}")
        if not r.terminal_ok:
            notes.append(f"terminal={r.actual_terminal}")
        if r.tool_ok is False:
            notes.append(f"tool={r.actual_tool}")
        if not r.keywords_ok:
            notes.append("missing keyword")
        if not r.safety_ok:
            notes.append("SAFETY VIOLATION")
        print(f"{r.id:<26} {r.category:<16} {result:<8} "
              f"{r.latency_seconds:<6} {', '.join(notes)}")

    # Aggregate.
    total = len(results)
    passed = sum(1 for r in results if r.overall_ok)

    per_cat: dict[str, list[CaseResult]] = {}
    for r in results:
        per_cat.setdefault(r.category, []).append(r)

    intent_correct = sum(1 for r in results if r.intent_ok)
    avg_latency = round(sum(r.latency_seconds for r in results) / total, 2)

    # Escalation precision: of the "out_of_scope" cases, how many escalated?
    oos = per_cat.get("out_of_scope", [])
    escalated = sum(1 for r in oos if r.actual_terminal == "escalate")
    escalation_precision = (escalated / len(oos)) if oos else 0.0

    print("-" * 76)
    print(f"\nOVERALL      : {passed}/{total} "
          f"({100 * passed / total:.1f}%)")
    print(f"BY CATEGORY  :")
    for cat, rs in sorted(per_cat.items()):
        cp = sum(1 for r in rs if r.overall_ok)
        print(f"  {cat:<18} {cp}/{len(rs)} "
              f"({100 * cp / len(rs):.1f}%)")
    print(f"INTENT ACC   : {intent_correct}/{total} "
          f"({100 * intent_correct / total:.1f}%)")
    print(f"ESCAL PRECIS : {escalated}/{len(oos)} "
          f"({100 * escalation_precision:.1f}%) "
          f"[fraction of out-of-scope that escalated]")
    print(f"AVG LATENCY  : {avg_latency}s per case")
    print("=" * 76)


def write_artifacts(results: list[CaseResult]) -> None:
    """Persist the results to CSV + JSON so they can be reviewed later."""
    csv_path = HERE / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    total = len(results)
    passed = sum(1 for r in results if r.overall_ok)
    per_cat: dict[str, dict] = {}
    seen: dict[str, list[CaseResult]] = {}
    for r in results:
        seen.setdefault(r.category, []).append(r)
    for cat, rs in seen.items():
        per_cat[cat] = {
            "total": len(rs),
            "passed": sum(1 for r in rs if r.overall_ok),
        }

    summary = {
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total, 3),
        "by_category": per_cat,
        "intent_accuracy": round(
            sum(1 for r in results if r.intent_ok) / total, 3
        ),
        "avg_latency_seconds": round(
            sum(r.latency_seconds for r in results) / total, 2
        ),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {csv_path}")
    print(f"Wrote {HERE / 'summary.json'}")


def main() -> None:
    print(f"Running eval: {len(CASES)} cases")
    results: list[CaseResult] = []
    for case in CASES:
        print(f"  ... {case.id}", end="", flush=True)
        r = score_case(case)
        results.append(r)
        print(" PASS" if r.overall_ok else " FAIL")

    print_report(results)
    write_artifacts(results)


if __name__ == "__main__":
    main()