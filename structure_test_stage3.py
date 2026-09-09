from app.agent.graph import build_graph, _route_after_intent
from app.agent.state import AgentState
from app.agent.tools import (
    ALL_TOOLS,
    TOOLS_BY_NAME,
    get_leave_balance,
    get_attendance,
    get_leave_requests,
)


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


print("Stage 3 structure test - agent + tools (no LLM required)\n")

# --- Graph now has 8 nodes -----------------------------------------------
graph = build_graph()
node_names = set(graph.get_graph().nodes.keys())
expected = {
    "classify_intent", "retrieve", "grade", "generate",
    "rewrite", "escalate", "tool_call", "tool_answer",
}
check(
    f"graph has all 8 nodes",
    expected.issubset(node_names),
)

# --- Intent routing ------------------------------------------------------
state: AgentState = {"intent": "personal_data"}
check(
    "personal_data -> tool_call",
    _route_after_intent(state) == "tool_call",
)

state = {"intent": "policy"}
check(
    "policy -> retrieve",
    _route_after_intent(state) == "retrieve",
)

state = {}  # missing intent
check(
    "missing intent defaults to retrieve",
    _route_after_intent(state) == "retrieve",
)

# --- Tool registry -------------------------------------------------------
check(
    "3 tools registered",
    len(ALL_TOOLS) == 3
    and set(TOOLS_BY_NAME.keys())
    == {"get_leave_balance", "get_leave_requests", "get_attendance"},
)

# --- Tool schemas are exposed correctly ----------------------------------
schema = get_leave_balance.args_schema.model_json_schema()
props = schema.get("properties", {})
check(
    "get_leave_balance schema advertises employee_code + leave_type",
    "employee_code" in props and "leave_type" in props,
)

# --- Tools work against real seeded DB -----------------------------------
# (You must have run `python -m app.seed` at least once.)
result = get_leave_balance.invoke({"employee_code": "EMP-001"})
check(
    "get_leave_balance returns balances for EMP-001",
    "balances" in result and len(result["balances"]) >= 1,
)

result = get_leave_balance.invoke(
    {"employee_code": "EMP-001", "leave_type": "annual"}
)
check(
    "get_leave_balance filters by leave_type",
    len(result.get("balances", [])) == 1
    and result["balances"][0]["leave_type"] == "annual",
)

result = get_leave_balance.invoke({"employee_code": "EMP-XXX"})
check(
    "unknown employee returns error dict, does not raise",
    "error" in result,
)

result = get_attendance.invoke({"employee_code": "EMP-002"})
check(
    "get_attendance returns summary + records",
    "summary" in result and "records" in result,
)

result = get_leave_requests.invoke({"employee_code": "EMP-003"})
check(
    "get_leave_requests returns a requests list",
    "requests" in result,
)

print("\nAll Stage 3 structure checks passed.")
print("Next: run smoke_test_stage3.py locally (Ollama running) for the")
print("full end-to-end test with real intent classification + tool calls.")
