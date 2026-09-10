from typing import TypedDict, Annotated
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    # ---- Inputs ----
    question: str                     # user's original question, never changes
    query: str                        # current search query (rewritten on retry)
    current_user: str                 # employee_code of the person asking
                                      # (in a real app this comes from auth)

    # ---- Intent (Stage 3) ----
    intent: str                       # "policy" | "personal_data"

    # ---- Retrieval (policy path) ----
    documents: list[Document]         # chunks returned by the last retrieval

    # ---- Grading (policy path) ----
    grade: str                        # "good" | "weak" | "none"
    grade_reason: str                 # short explanation from the grader

    # ---- Tools (personal_data path, Stage 3) ----
    tool_calls: list[dict]            # tool invocations the LLM chose
    tool_results: list[dict]          # raw results returned by each tool

    # ---- HITL (Stage 4) ----
    pending_writes: list[dict]        # write tool calls awaiting approval
    approval: str                     # "approved" | "rejected" (set on resume)
    approval_note: str                # optional note from the human

    # ---- Loop control ----
    retry_count: int                  # how many rewrites we have done so far

    # ---- Outputs ----
    answer: str                       # final answer if we generated one
    citations: list[str]              # source docs used for the answer
    escalation_team: str              # HR / IT / Security / Finance if escalated
    escalation_reason: str            # short explanation of why we escalated

    # ---- Bookkeeping (useful for tracing + eval) ----
    path: list[str]                   # ordered names of nodes we ran