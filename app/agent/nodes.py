"""
The five nodes of the onboarding agent.

Each node is a plain function that takes the current AgentState and
returns a partial update (a dict of only the keys it wants to change).
LangGraph merges the update into the state and passes it on.

Node order in the graph:

    retrieve  ->  grade_documents  ->  (conditional edge)
                                       -> generate_answer  (good)
                                       -> rewrite_query   (weak, retries left)
                                       -> escalate        (no docs / retries out)
    rewrite_query -> retrieve (loop)

The intelligence lives in `grade_documents` and the conditional edge:
those are what make this an agent rather than a chain.
"""
from __future__ import annotations

import json
import re
from textwrap import dedent

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent import config
from app.agent.llm import get_llm, get_json_llm
from app.agent.state import AgentState
from app.agent.vector_store import get_retriever


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _append_path(state: AgentState, node_name: str) -> list[str]:
    """Extend the trace of which nodes have run."""
    return list(state.get("path", [])) + [node_name]


def _extract_json(text: str) -> dict | None:
    """
    Best-effort JSON extraction.

    Small local models (like llama3.2:3b) sometimes wrap JSON in
    markdown fences or add commentary. We try direct parse first,
    then fall back to grabbing the first {...} block.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _format_docs_for_prompt(docs: list[Document]) -> str:
    """Render retrieved docs as a numbered, source-labelled block."""
    if not docs:
        return "(no documents)"
    parts = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] source={source}\n{d.page_content}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def retrieve_node(state: AgentState) -> dict:
    """
    Pure data-fetch node. Reads the CURRENT query (which may be the
    rewritten one) and pulls the top-k chunks from the vector store.
    """
    retriever = get_retriever()
    query = state.get("query") or state["question"]
    docs = retriever.invoke(query)
    return {
        "documents": docs,
        "path": _append_path(state, "retrieve"),
    }


def grade_documents_node(state: AgentState) -> dict:
    """
    The decision-making node. An LLM inspects the retrieved chunks
    against the question and returns a JSON verdict:

        {"grade": "good" | "weak" | "none", "reason": "..."}

    - "good": chunks contain the answer to the question asked.
    - "weak": chunks are on-topic but the specific answer is not there.
    - "none": chunks are unrelated to the question.

    Uses Ollama's JSON mode so output is guaranteed to be valid JSON,
    and gives the small model concrete examples so it grades against
    the RIGHT bar: "does this answer the question that was asked",
    not "does this cover everything about the topic".
    """
    llm = get_json_llm()
    docs = state.get("documents", [])
    question = state["question"]

    # If retrieval returned literally nothing, skip the LLM call.
    if not docs:
        return {
            "grade": "none",
            "grade_reason": "retrieval returned no documents",
            "path": _append_path(state, "grade_documents"),
        }

    system = dedent("""\
        You are a document relevance grader for a company onboarding
        assistant.

        Your ONLY job: decide whether the retrieved document chunks
        contain enough information to answer the SPECIFIC question the
        user asked.

        Do NOT require the chunks to cover the entire topic area. If
        the user asked one narrow question and the chunks answer that
        one narrow question, the grade is "good" - even if the chunks
        do not mention related things.

        Reply with a JSON object of the form:
        {"grade": "good" | "weak" | "none", "reason": "<one short sentence>"}

        Grade meanings:
          - "good": the chunks directly answer the specific question asked.
          - "weak": the chunks are on-topic but do NOT contain the answer.
          - "none": the chunks are unrelated to the question.

        Examples:

        Q: "How many days of annual leave do I get?"
        Chunks: "Every full-time employee receives 20 days of paid annual
                 leave per calendar year."
        -> {"grade": "good", "reason": "chunk states 20 days annual leave"}

        Q: "What are the password requirements?"
        Chunks: "All accounts must use a password of at least 12 characters,
                 containing uppercase and lowercase letters, at least one
                 number and one special character."
        -> {"grade": "good", "reason": "chunk lists the password rules"}

        Q: "Can I bring my dog to the office?"
        Chunks: "Every full-time employee receives 20 days of paid annual
                 leave per calendar year."
        -> {"grade": "none", "reason": "chunks are about leave, not pets"}

        Q: "What is the exact bonus percentage for engineers?"
        Chunks: "The company offers competitive compensation and benefits
                 to all employees."
        -> {"grade": "weak", "reason": "on-topic but no specific number"}
    """)

    user = (
        f"Question:\n{question}\n\n"
        f"Retrieved chunks:\n{_format_docs_for_prompt(docs)}"
    )

    response = llm.invoke([SystemMessage(system), HumanMessage(user)])
    parsed = _extract_json(response.content) or {}

    grade = str(parsed.get("grade", "")).lower().strip()
    if grade not in {"good", "weak", "none"}:
        # Defensive default: if the model returned junk, treat as weak
        # so we get one retry rather than blindly answering.
        grade = "weak"
    reason = str(parsed.get("reason", "")).strip() or "no reason given"

    return {
        "grade": grade,
        "grade_reason": reason,
        "path": _append_path(state, "grade_documents"),
    }


def generate_answer_node(state: AgentState) -> dict:
    """
    Produce the final answer, grounded ONLY in the retrieved chunks,
    with source citations. This is called only when the grader said
    the chunks are 'good'.
    """
    llm = get_llm()
    docs = state.get("documents", [])
    question = state["question"]

    system = dedent("""\
        You are a company onboarding assistant.

        Answer the user's question using ONLY the information contained
        in the provided document chunks. Do NOT rely on outside knowledge
        or make up policy details. Keep the answer concise (2-5 sentences).

        End your answer with a "Sources:" line listing the source filenames
        of the chunks you used, comma-separated.
    """)

    user = (
        f"Question:\n{question}\n\n"
        f"Document chunks:\n{_format_docs_for_prompt(docs)}"
    )

    response = llm.invoke([SystemMessage(system), HumanMessage(user)])
    answer_text = response.content.strip()

    citations = sorted({d.metadata.get("source", "unknown") for d in docs})

    return {
        "answer": answer_text,
        "citations": citations,
        "path": _append_path(state, "generate_answer"),
    }


def rewrite_query_node(state: AgentState) -> dict:
    """
    Reformulate the question into a better search query. Called when
    the grader said the retrieval was weak. Increments the retry
    counter so the loop is bounded.
    """
    llm = get_llm()
    question = state["question"]
    previous_query = state.get("query") or question

    system = dedent("""\
        You are a query rewriter for a document retrieval system.
        Given a user question and a previous search query that did not
        retrieve good results, produce a SHORT alternative search query
        (5-15 words) that is more likely to match relevant company
        policy documents. Return ONLY the new query text, no quotes,
        no preamble.
    """)

    user = (
        f"User question: {question}\n"
        f"Previous query that failed: {previous_query}"
    )

    response = llm.invoke([SystemMessage(system), HumanMessage(user)])
    new_query = response.content.strip().splitlines()[0].strip().strip('"')

    # Defensive floor: if the model returned nothing sensible, add a
    # topical prefix so at least SOMETHING changes.
    if not new_query or new_query.lower() == previous_query.lower():
        new_query = f"company policy {question}"

    return {
        "query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "path": _append_path(state, "rewrite_query"),
    }


def escalate_node(state: AgentState) -> dict:
    """
    Route the question to the right human team.

    Strategy: ask the LLM to classify the question into one of the
    known departments. Fall back to the DEFAULT if the classification
    is nonsense.
    """
    llm = get_json_llm()
    question = state["question"]
    depts = ", ".join(config.KNOWN_DEPARTMENTS)

    system = dedent(f"""\
        You are an escalation router. Classify the user's question
        into exactly ONE of these departments: {depts}.

        Reply with ONLY a JSON object of the form:
        {{"team": "<department>", "reason": "<one short sentence>"}}
        Do NOT add any text before or after the JSON.
    """)

    response = llm.invoke([SystemMessage(system), HumanMessage(question)])
    parsed = _extract_json(response.content) or {}

    team = parsed.get("team", "").strip()
    if team not in config.KNOWN_DEPARTMENTS:
        team = config.DEFAULT_ESCALATION
    reason = (parsed.get("reason") or "escalated by agent").strip()

    return {
        "escalation_team": team,
        "escalation_reason": reason,
        "answer": (
            f"I could not confidently answer this from the company documents. "
            f"I have escalated it to the {team} team."
        ),
        "path": _append_path(state, "escalate"),
    }
