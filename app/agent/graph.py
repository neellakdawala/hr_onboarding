"""
LangGraph state graph for the onboarding agent.

Flow:

    START
      |
      v
    retrieve
      |
      v
    grade_documents
      |
      +----- good  ------> generate_answer ---> END
      |
      +----- weak  ------> (retries left?)
      |                        |
      |                        +-- yes --> rewrite_query --> retrieve (loop)
      |                        +-- no  --> escalate --> END
      |
      +----- none  ------> escalate ---> END

The conditional edge after `grade_documents` is what makes this an
agent: the same input can take different paths based on the graded
quality of what was retrieved. That is control flow decided by the
system's own judgment, not a fixed script.
"""
from langgraph.graph import StateGraph, START, END

from app.agent import config
from app.agent.state import AgentState
from app.agent.nodes import (
    retrieve_node,
    grade_documents_node,
    generate_answer_node,
    rewrite_query_node,
    escalate_node,
)


def _route_after_grading(state: AgentState) -> str:
    """
    Conditional edge: decide which node runs after grading.

    Returns the name of the next node. LangGraph's add_conditional_edges
    maps these string labels to actual node names.
    """
    grade = state.get("grade", "weak")
    retries = state.get("retry_count", 0)

    if grade == "good":
        return "generate"
    if grade == "none":
        return "escalate"
    # grade == "weak"
    if retries < config.MAX_RETRIES:
        return "rewrite"
    return "escalate"      # out of retries -> escalate


def build_graph():
    """Compile and return the runnable LangGraph."""
    workflow = StateGraph(AgentState)

    # Register nodes.
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("escalate", escalate_node)

    # Straight edges.
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")

    # The one conditional edge - the "brain" of the agent.
    workflow.add_conditional_edges(
        "grade",
        _route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "escalate": "escalate",
        },
    )

    # Rewrite loops back to retrieve; generate and escalate are terminal.
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


def run_agent(question: str) -> AgentState:
    """
    Convenience entry point: run one question end-to-end through the
    graph and return the final state.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "question": question,
        "query": question,
        "retry_count": 0,
        "path": [],
    }
    return graph.invoke(initial_state)
