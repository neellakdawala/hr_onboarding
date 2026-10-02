from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.agent import config
from app.agent.state import AgentState
from app.agent.tracing import get_callbacks
from app.agent.nodes import (
    supervisor_node,
    retrieve_node,
    grade_documents_node,
    generate_answer_node,
    rewrite_query_node,
    escalate_node,
    tool_call_node,
    human_approval_node,
    tool_execute_node,
    tool_answer_node,
)


def _route_after_supervisor(state: AgentState) -> str:
    return "tool_call" if state.get("intent") == "personal_data" else "retrieve"


# Backward-compat aliases: Stage 3/4 tests import these names.
_route_after_intent = _route_after_supervisor


def _route_after_grading(state: AgentState) -> str:
    """
    Four-way route:
      good          -> generate
      out_of_scope  -> supervisor (reroute)   if reroutes remaining
      weak          -> rewrite                if retries remaining
      none          -> escalate
      exhausted     -> escalate
    """
    grade = state.get("grade", "weak")
    retries = state.get("retry_count", 0)
    reroutes = state.get("supervisor_retries", 0)

    if grade == "good":
        return "generate"
    if grade == "out_of_scope":
        if reroutes < config.MAX_SUPERVISOR_REROUTES:
            return "reroute"
        return "escalate"
    if grade == "none":
        return "escalate"
    if retries < config.MAX_RETRIES:
        return "rewrite"
    return "escalate"


def _reroute_node(state: AgentState) -> dict:
    """
    Between the grader and the supervisor on a reroute. Increments the
    reroute counter, resets query state, then supervisor picks a new
    specialist.
    """
    return {
        "supervisor_retries": state.get("supervisor_retries", 0) + 1,
        # Reset per-run counters that belong to the OLD specialist path.
        "retry_count": 0,
        "query": state["question"],
        "path": _append_path(state, "reroute"),
    }


def _append_path(state: AgentState, label: str) -> list:
    return list(state.get("path", [])) + [label]


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("reroute", _reroute_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("tool_execute", tool_execute_node)
    workflow.add_node("tool_answer", tool_answer_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"retrieve": "retrieve", "tool_call": "tool_call"},
    )

    workflow.add_edge("tool_call", "human_approval")
    workflow.add_edge("human_approval", "tool_execute")
    workflow.add_edge("tool_execute", "tool_answer")
    workflow.add_edge("tool_answer", END)

    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        _route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "reroute": "reroute",
            "escalate": "escalate",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("reroute", "supervisor")   # bounded by MAX_SUPERVISOR_REROUTES
    workflow.add_edge("generate", END)
    workflow.add_edge("escalate", END)

    return workflow.compile(checkpointer=MemorySaver())


def _run_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": get_callbacks(),
    }


def run_agent(
    question: str,
    current_user: str = "EMP-001",
    thread_id: str = "default",
    recent_history: str = "",
) -> AgentState:
    graph = build_graph()
    initial_state: AgentState = {
        "question": question,
        "query": question,
        "current_user": current_user,
        "retry_count": 0,
        "supervisor_retries": 0,
        "recent_history": recent_history,
        "path": [],
    }
    return graph.invoke(initial_state, config=_run_config(thread_id))


def run_agent_interactive(
    question: str,
    approve_writes: bool = True,
    current_user: str = "EMP-001",
    thread_id: str = "default",
    approval_note: str = "",
    recent_history: str = "",
) -> AgentState:
    from langgraph.types import Command

    graph = build_graph()
    cfg = _run_config(thread_id)

    initial_state: AgentState = {
        "question": question,
        "query": question,
        "current_user": current_user,
        "retry_count": 0,
        "supervisor_retries": 0,
        "recent_history": recent_history,
        "path": [],
    }

    result = graph.invoke(initial_state, config=cfg)
    snapshot = graph.get_state(cfg)
    if snapshot.interrupts:
        decision = "approved" if approve_writes else "rejected"
        result = graph.invoke(
            Command(resume={"approval": decision, "note": approval_note}),
            config=cfg,
        )
    return result


def start_run(
    question: str,
    current_user: str = "EMP-001",
    thread_id: str = "default",
    graph=None,
    recent_history: str = "",
) -> tuple[AgentState, bool, list[dict]]:
    graph = graph or build_graph()
    cfg = _run_config(thread_id)

    initial_state: AgentState = {
        "question": question,
        "query": question,
        "current_user": current_user,
        "retry_count": 0,
        "supervisor_retries": 0,
        "recent_history": recent_history,
        "path": [],
    }
    state = graph.invoke(initial_state, config=cfg)
    snapshot = graph.get_state(cfg)
    if snapshot.interrupts:
        pending = state.get("pending_writes", []) or []
        return state, True, pending
    return state, False, []


def resume_run(
    thread_id: str,
    approval: str,
    note: str = "",
    graph=None,
) -> AgentState:
    from langgraph.types import Command
    graph = graph or build_graph()
    cfg = _run_config(thread_id)
    return graph.invoke(
        Command(resume={"approval": approval, "note": note}),
        config=cfg,
    )