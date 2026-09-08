"""
The shared state that flows through every node in the graph.

In LangGraph, every node reads from this state and returns a partial
update to it. Keeping the fields explicit (rather than a free-form dict)
means we can trust what is in the state at each node.

Key design choice: `question` (the user's original words) and `query`
(the current search string) are separate. `query` gets overwritten by
the Rewrite node during a retry, while `question` never changes. This
is what makes the retry loop clean.
"""
from typing import TypedDict, Annotated
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    # ---- Inputs ----
    question: str                     # user's original question, never changes
    query: str                        # current search query (rewritten on retry)

    # ---- Retrieval ----
    documents: list[Document]         # chunks returned by the last retrieval

    # ---- Grading ----
    grade: str                        # "good" | "weak" | "none"
    grade_reason: str                 # short explanation from the grader

    # ---- Loop control ----
    retry_count: int                  # how many rewrites we have done so far

    # ---- Outputs ----
    answer: str                       # final answer if we generated one
    citations: list[str]              # source docs used for the answer
    escalation_team: str              # HR / IT / Security / Finance if escalated
    escalation_reason: str            # short explanation of why we escalated

    # ---- Bookkeeping (useful for tracing + eval) ----
    path: list[str]                   # ordered names of nodes we ran
