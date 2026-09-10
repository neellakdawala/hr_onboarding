from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
 
from app.agent import config
from app.agent.state import AgentState
from app.agent.tracing import get_callbacks
from app.agent.nodes import (
    classify_intent_node,
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
 
 
def _route_after_intent(state: AgentState) -> str:
    return "tool_call" if state.get("intent") == "personal_data" else "retrieve"
 
 
def _route_after_grading(state: AgentState) -> str:
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
    """
    Compile and return the runnable LangGraph.
 
    Compiled with an in-memory checkpointer so `interrupt()` works.
    When you call `graph.invoke(...)`, always pass a `config` with a
    unique `thread_id` - each conversation is a separate thread and
    the checkpointer keys state by thread.
    """
    workflow = StateGraph(AgentState)
 
    # Register every node.
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("tool_execute", tool_execute_node)
    workflow.add_node("tool_answer", tool_answer_node)
 
    # Entry.
    workflow.add_edge(START, "classify_intent")
 
    # First branch: intent -> retrieve OR tool_call.
    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_intent,
        {"retrieve": "retrieve", "tool_call": "tool_call"},
    )
 
    # Tool pipeline (Stage 4).
    workflow.add_edge("tool_call", "human_approval")
    workflow.add_edge("human_approval", "tool_execute")
    workflow.add_edge("tool_execute", "tool_answer")
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
 
    # Compile WITH a checkpointer so interrupt/resume works.
    return workflow.compile(checkpointer=MemorySaver())
 
 
def _run_config(thread_id: str) -> dict:
    """Standard run config: thread + tracing callbacks if configured."""
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": get_callbacks(),
    }
 
 
def run_agent(
    question: str,
    current_user: str = "EMP-001",
    thread_id: str = "default",
) -> AgentState:
    """
    Run the agent to completion for a NON-interrupting question.
 
    If the graph pauses on human_approval, use `run_agent_interactive`
    below instead - that handles the pause + resume dance.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "question": question,
        "query": question,
        "current_user": current_user,
        "retry_count": 0,
        "path": [],
    }
    return graph.invoke(initial_state, config=_run_config(thread_id))
 
 
def run_agent_interactive(
    question: str,
    approve_writes: bool = True,
    current_user: str = "EMP-001",
    thread_id: str = "default",
    approval_note: str = "",
) -> AgentState:
    """
    Run the agent AND handle any human-approval pause automatically.
 
    `approve_writes` is the decision the "human" (you, in tests) will
    supply if the agent pauses to ask. Real applications would surface
    the pause to a UI and wait for a user click.
 
    Returns the final state after the graph runs to completion.
    """
    from langgraph.types import Command
 
    graph = build_graph()
    cfg = _run_config(thread_id)
 
    initial_state: AgentState = {
        "question": question,
        "query": question,
        "current_user": current_user,
        "retry_count": 0,
        "path": [],
    }
 
    # First run - may hit an interrupt and pause.
    result = graph.invoke(initial_state, config=cfg)
 
    # After invoke, check if the graph is paused waiting on approval.
    snapshot = graph.get_state(cfg)
    if snapshot.interrupts:
        decision = "approved" if approve_writes else "rejected"
        result = graph.invoke(
            Command(resume={"approval": decision, "note": approval_note}),
            config=cfg,
        )
 
    return result
 