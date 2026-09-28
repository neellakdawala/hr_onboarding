"""
Feature 4 structure test.

Verifies the multi-agent supervisor architecture WITHOUT the LLM:
  - The graph has 11 nodes (Feature 3 + supervisor + reroute)
  - The old classify_intent alias still resolves (backward-compat)
  - Routing decisions for all four grade outcomes are correct
  - Reroute path respects MAX_SUPERVISOR_REROUTES
  - Each specialist has the right tool subset
  - Config maps for department + ticket category are consistent

Run:  python structure_test_feature4.py
"""
from app.agent.graph import (
    build_graph, _route_after_supervisor, _route_after_grading,
)
from app.agent.nodes import (
    supervisor_node, classify_intent_node,
)
from app.agent import config
from app.agent.tools import ALL_TOOLS


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


print("Feature 4 structure test - multi-agent supervisor\n")

# --- Graph shape ---
graph = build_graph()
node_names = set(graph.get_graph().nodes.keys())
expected = {
    "supervisor", "reroute", "retrieve", "grade", "generate",
    "rewrite", "escalate",
    "tool_call", "human_approval", "tool_execute", "tool_answer",
}
check("graph has all 11 nodes", expected.issubset(node_names))

# --- Backward compat alias ---
check(
    "classify_intent_node still resolves (to supervisor_node)",
    classify_intent_node is supervisor_node,
)

# --- Config maps ---
check(
    "3 specialists configured",
    set(config.KNOWN_SPECIALISTS) == {"hr", "it", "security"},
)
check(
    "every specialist has a department",
    all(s in config.SPECIALIST_TO_DEPARTMENT
        for s in config.KNOWN_SPECIALISTS),
)
check(
    "every specialist has a tools subset",
    all(s in config.SPECIALIST_TO_TOOLS
        for s in config.KNOWN_SPECIALISTS),
)
check(
    "every specialist has a ticket category",
    all(s in config.SPECIALIST_TICKET_CATEGORY
        for s in config.KNOWN_SPECIALISTS),
)

# --- Tool subsets ---
tool_names = {t.name for t in ALL_TOOLS}
check(
    "HR specialist can call most write tools",
    "submit_leave_request" in config.SPECIALIST_TO_TOOLS["hr"]
    and "approve_leave_request" in config.SPECIALIST_TO_TOOLS["hr"]
    and "create_hr_ticket" in config.SPECIALIST_TO_TOOLS["hr"],
)
check(
    "IT specialist CANNOT call leave tools",
    "submit_leave_request" not in config.SPECIALIST_TO_TOOLS["it"]
    and "approve_leave_request" not in config.SPECIALIST_TO_TOOLS["it"],
)
check(
    "Security specialist CANNOT call leave tools",
    "submit_leave_request" not in config.SPECIALIST_TO_TOOLS["security"]
    and "approve_leave_request" not in config.SPECIALIST_TO_TOOLS["security"],
)
check(
    "every specialist can file tickets",
    all("create_hr_ticket" in config.SPECIALIST_TO_TOOLS[s]
        for s in config.KNOWN_SPECIALISTS),
)
check(
    "every allowed name is a real tool",
    all(n in tool_names
        for s in config.KNOWN_SPECIALISTS
        for n in config.SPECIALIST_TO_TOOLS[s]),
)

# --- Routing after supervisor ---
check(
    "policy -> retrieve",
    _route_after_supervisor({"intent": "policy"}) == "retrieve",
)
check(
    "personal_data -> tool_call",
    _route_after_supervisor({"intent": "personal_data"}) == "tool_call",
)

# --- Routing after grade (four outcomes) ---
check(
    "good -> generate",
    _route_after_grading({"grade": "good"}) == "generate",
)
check(
    "weak with retries left -> rewrite",
    _route_after_grading(
        {"grade": "weak", "retry_count": 0}
    ) == "rewrite",
)
check(
    "weak exhausted -> escalate",
    _route_after_grading(
        {"grade": "weak", "retry_count": config.MAX_RETRIES}
    ) == "escalate",
)
check(
    "none -> escalate",
    _route_after_grading({"grade": "none"}) == "escalate",
)
check(
    "out_of_scope with reroutes left -> reroute",
    _route_after_grading(
        {"grade": "out_of_scope", "supervisor_retries": 0}
    ) == "reroute",
)
check(
    "out_of_scope exhausted -> escalate",
    _route_after_grading({
        "grade": "out_of_scope",
        "supervisor_retries": config.MAX_SUPERVISOR_REROUTES,
    }) == "escalate",
)

print("\nAll Feature 4 structure checks passed.")
