"""
Stage 5 structure test.

Verifies the eval framework and CI/Docker scaffolding without needing
Ollama running:

  - Eval cases load and have the right shape
  - Every category is represented
  - Scoring helpers (_terminal_of, _tool_of) work on synthetic states
  - Dockerfile and docker-compose exist and reference the right services
  - GitHub Actions workflow exists

Run:  python structure_test_stage5.py
"""
from pathlib import Path

from evals.cases import CASES, cases_by_category, EvalCase
from evals.run import _terminal_of, _tool_of, score_case  # noqa: F401


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


ROOT = Path(__file__).resolve().parent


print("Stage 5 structure test - eval + Docker + CI (no LLM required)\n")

# --- Eval cases ---------------------------------------------------------
check(f"eval dataset loaded ({len(CASES)} cases)", len(CASES) >= 15)

categories = set(cases_by_category().keys())
expected_cats = {"policy", "personal_read", "personal_write", "out_of_scope"}
check(
    f"all 4 categories represented ({sorted(expected_cats)})",
    expected_cats.issubset(categories),
)

for c in CASES:
    assert isinstance(c, EvalCase), f"case {c!r} not an EvalCase"
    assert c.expected_intent in {"policy", "personal_data"}, c.id
    assert c.expected_terminal in {
        "generate_answer", "escalate", "tool_answer"
    }, c.id
check("every case has valid intent + terminal", True)

# All case IDs are unique.
ids = [c.id for c in CASES]
check("all case IDs are unique", len(set(ids)) == len(ids))

# Every out_of_scope case has a safety expectation.
oos = [c for c in CASES if c.category == "out_of_scope"]
check(
    "out_of_scope cases expect escalate terminal",
    all(c.expected_terminal == "escalate" for c in oos),
)

# --- Scoring helpers ----------------------------------------------------
check(
    "_terminal_of picks the terminal node",
    _terminal_of(
        ["classify_intent", "retrieve", "grade_documents", "generate_answer"]
    ) == "generate_answer",
)
check(
    "_terminal_of picks escalate when present",
    _terminal_of(
        ["classify_intent", "retrieve", "grade_documents",
         "rewrite_query", "retrieve", "grade_documents", "escalate"]
    ) == "escalate",
)
check(
    "_tool_of extracts first tool from state",
    _tool_of({"tool_calls": [{"name": "get_leave_balance", "args": {}}]})
    == "get_leave_balance",
)
check(
    "_tool_of returns None with no tool calls",
    _tool_of({}) is None,
)

# --- Docker scaffolding ------------------------------------------------
dockerfile = ROOT / "Dockerfile"
compose = ROOT / "docker-compose.yml"
check("Dockerfile exists", dockerfile.is_file())
check("docker-compose.yml exists", compose.is_file())

dockerfile_text = dockerfile.read_text() if dockerfile.exists() else ""
check(
    "Dockerfile installs requirements + exposes port 8000",
    "requirements.txt" in dockerfile_text
    and "8000" in dockerfile_text,
)

compose_text = compose.read_text() if compose.exists() else ""
check(
    "docker-compose defines both hrms and ollama services",
    "hrms" in compose_text.lower() and "ollama" in compose_text.lower(),
)

# --- CI workflow -------------------------------------------------------
ci = ROOT / ".github" / "workflows" / "ci.yml"
check("GitHub Actions workflow exists", ci.is_file())

ci_text = ci.read_text() if ci.exists() else ""
check(
    "CI runs at least one structure test",
    "structure_test_stage2.py" in ci_text
    or "structure_test_stage3.py" in ci_text
    or "structure_test_stage4.py" in ci_text,
)

print("\nAll Stage 5 structure checks passed.")
print("Next: run the full eval locally (Ollama up):")
print("  python -m evals.run")
