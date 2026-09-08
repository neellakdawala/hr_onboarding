"""
Stage 2 structure test.

Verifies what CAN be verified without a running Ollama:
  - All modules import cleanly
  - The graph compiles
  - The conditional routing logic is correct for every case
  - The state helpers work

This runs in CI. The full end-to-end test that actually calls the LLM
and vector store lives in smoke_test_stage2.py and is run on the
developer's machine.
"""
from app.agent.graph import build_graph, _route_after_grading
from app.agent.state import AgentState
from app.agent.nodes import _extract_json, _append_path


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


print("Stage 2 structure test - agent graph (no LLM required)\n")

# --- Graph compiles ---
graph = build_graph()
check("graph compiles into a runnable", graph is not None)

# --- Node names present ---
nodes = graph.get_graph().nodes
expected = {"retrieve", "grade", "generate", "rewrite", "escalate"}
check(
    f"graph has all 5 nodes ({sorted(expected)})",
    expected.issubset(set(nodes.keys())),
)

# --- Routing logic: good -> generate ---
state: AgentState = {"grade": "good", "retry_count": 0}
check("good -> generate", _route_after_grading(state) == "generate")

# --- Routing logic: none -> escalate immediately ---
state = {"grade": "none", "retry_count": 0}
check("none -> escalate", _route_after_grading(state) == "escalate")

# --- Routing logic: weak with retries left -> rewrite ---
state = {"grade": "weak", "retry_count": 0}
check("weak + 0 retries -> rewrite", _route_after_grading(state) == "rewrite")

state = {"grade": "weak", "retry_count": 1}
check("weak + 1 retry -> rewrite", _route_after_grading(state) == "rewrite")

# --- Routing logic: weak, retries exhausted -> escalate ---
state = {"grade": "weak", "retry_count": 2}
check(
    "weak + max retries -> escalate",
    _route_after_grading(state) == "escalate",
)

# --- JSON extraction helpers ---
check(
    "extract_json handles clean JSON",
    _extract_json('{"grade": "good"}') == {"grade": "good"},
)
check(
    "extract_json handles JSON inside markdown fence",
    _extract_json('```json\n{"grade": "weak"}\n```') == {"grade": "weak"},
)
check(
    "extract_json handles preamble + JSON",
    _extract_json('Sure! Here you go: {"grade": "none"}')
    == {"grade": "none"},
)
check("extract_json returns None on junk", _extract_json("no json here") is None)

# --- Path tracing helper ---
state = {"path": ["retrieve"]}
check(
    "append_path appends without mutating original",
    _append_path(state, "grade") == ["retrieve", "grade"]
    and state["path"] == ["retrieve"],
)

# --- Grader early-exit: empty docs -> 'none' without LLM call ---
from app.agent.nodes import grade_documents_node
result = grade_documents_node({"question": "anything", "documents": []})
check(
    "grader shortcircuits to 'none' when no docs (no LLM call)",
    result["grade"] == "none",
)

print("\nAll Stage 2 structure checks passed.")
print("Next: run smoke_test_stage2.py locally (with Ollama running) for the")
print("full end-to-end test that actually queries the LLM and vector store.")
