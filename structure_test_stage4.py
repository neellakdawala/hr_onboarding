"""
Stage 4 structure test.

Verifies what can be checked without hitting Ollama:
  - Graph now has 10 nodes (Stage 3 + human_approval + tool_execute)
  - Write tool is registered and detectable via WRITE_TOOL_NAMES
  - Tracing module gracefully returns [] when Langfuse isn't configured
  - Human-approval node behavior:
      - no writes pending: passthrough, approves trivially
  - Tool-execute node correctly SKIPS writes when approval == "rejected"
  - Tool-execute node correctly RUNS writes when approval == "approved"

The full HITL interrupt + resume dance is checked in
smoke_test_stage4.py using the real graph.

Run:  python structure_test_stage4.py
"""
from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.agent.tools import (
    ALL_TOOLS, TOOLS_BY_NAME, WRITE_TOOL_NAMES,
    submit_leave_request,
)
from app.agent.nodes import (
    human_approval_node, tool_execute_node,
)
from app.agent.tracing import get_callbacks, is_tracing_enabled


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


print("Stage 4 structure test - HITL + tracing (no LLM required)\n")

# --- Graph shape ---------------------------------------------------------
graph = build_graph()
node_names = set(graph.get_graph().nodes.keys())
expected = {
    "classify_intent", "retrieve", "grade", "generate",
    "rewrite", "escalate",
    "tool_call", "human_approval", "tool_execute", "tool_answer",
}
check(f"graph has all 10 nodes", expected.issubset(node_names))

# --- Write tool registration --------------------------------------------
check(
    "submit_leave_request is registered",
    "submit_leave_request" in TOOLS_BY_NAME,
)
check(
    "submit_leave_request is marked as a WRITE",
    "submit_leave_request" in WRITE_TOOL_NAMES,
)
check(
    "read tools are NOT in WRITE_TOOL_NAMES",
    "get_leave_balance" not in WRITE_TOOL_NAMES
    and "get_attendance" not in WRITE_TOOL_NAMES
    and "get_leave_requests" not in WRITE_TOOL_NAMES,
)

# --- Tracing gracefully degrades ----------------------------------------
# (No Langfuse env vars set in this test context.)
check(
    "get_callbacks returns [] when Langfuse not configured",
    get_callbacks() == [],
)
check(
    "is_tracing_enabled is False when Langfuse not configured",
    is_tracing_enabled() is False,
)

# --- Human approval node: no writes -> trivially approved ---------------
state: AgentState = {"pending_writes": []}
out = human_approval_node(state)
check(
    "human_approval passthrough when no writes pending",
    out["approval"] == "approved",
)

# --- Tool execute node: read + rejected write -> only read runs --------
# Set up state where the LLM planned 1 read + 1 write, and human rejected.
state = {
    "tool_calls": [
        {"name": "get_leave_balance", "args": {"employee_code": "EMP-001"}},
        {"name": "submit_leave_request", "args": {
            "employee_code": "EMP-001", "leave_type": "annual",
            "start_date": "2026-12-20", "end_date": "2026-12-22",
            "days": 3.0, "reason": "test",
        }},
    ],
    "approval": "rejected",
    "approval_note": "user changed their mind",
}
out = tool_execute_node(state)
results = out["tool_results"]
check("tool_execute returns 2 results (one per planned call)",
      len(results) == 2)
check(
    "read tool RAN even with write rejected",
    results[0]["tool"] == "get_leave_balance"
    and "balances" in results[0].get("result", {}),
)
check(
    "write tool SKIPPED with skipped=True and reason",
    results[1]["tool"] == "submit_leave_request"
    and results[1]["result"].get("skipped") is True
    and "rejected" in results[1]["result"].get("reason", "").lower(),
)

# --- Tool execute node: approved write actually runs --------------------
state = {
    "tool_calls": [
        {"name": "submit_leave_request", "args": {
            "employee_code": "EMP-001", "leave_type": "annual",
            "start_date": "2026-12-24", "end_date": "2026-12-24",
            "days": 1.0, "reason": "structure test approved write",
        }},
    ],
    "approval": "approved",
}
out = tool_execute_node(state)
result = out["tool_results"][0]["result"]
check(
    "approved write actually created a leave request",
    result.get("created") is True and "request_id" in result,
)

print("\nAll Stage 4 structure checks passed.")
print("Next: run smoke_test_stage4.py locally (Ollama running) to test the")
print("real interrupt + resume flow with a write request.")
