from langgraph.graph import StateGraph, START, END
from app.agent import config
from app.agent.state import AgentState
from app.agent.nodes import (
    classify_intent_node,
    retrieve_node,
    grade_documents_node,
    generate_answer_node,
    rewrite_query_node,
    escalate_node,
    tool_call_node,
    tool_answer_node,
)


def _route_after_intent(state: AgentState) -> str:
    """First branch: which mode is this question in?"""
    return "tool_call" if state.get("intent") == "personal_data" else "retrieve"


def _route_after_grading(state: AgentState) -> str:
    """Second branch (same as Stage 2)."""
    grade = state.get("grade", "weak")
    retries = state.get("retry_count", 0)
    if grade == "good":
        return "generate"
    if grade == "none":
        return "escalate"
    if retries < config.MAX_RETRIES:
        return "rewrite"
    return "escalate"


def build_graph():
    """Compile and return the runnable LangGraph."""
    workflow = StateGraph(AgentState)

    # Register every node.
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("tool_answer", tool_answer_node)

    # Entry point.
    workflow.add_edge(START, "classify_intent")

    # First branch: intent -> retrieve OR tool_call.
    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_intent,
        {"retrieve": "retrieve", "tool_call": "tool_call"},
    )

    # Tool path.
    workflow.add_edge("tool_call", "tool_answer")
    workflow.add_edge("tool_answer", END)

    # Policy path (unchanged from Stage 2).
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        _route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "escalate": "escalate",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


def run_agent(question: str, current_user: str = "EMP-001") -> AgentState:
    """
    Convenience entry point.

    `current_user` is the employee_code the tools will act on when the
    question is about personal data. In a real deployment this comes from
    the authenticated session; for local development we pass it directly.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "question": question,
        "query": question,
        "current_user": current_user,
        "retry_count": 0,
        "path": [],
    }
    return graph.invoke(initial_state)