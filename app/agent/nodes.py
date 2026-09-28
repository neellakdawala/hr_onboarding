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

    Feature 4: if a specialist has been chosen, we filter retrieval
    by that specialist's department metadata. So the HR specialist
    reads HR docs, the IT specialist reads IT docs, and the Security
    specialist reads Security docs. On a first-pass empty result we
    fall back to unfiltered search so a mildly mis-routed question
    still finds something useful.
    """
    from app.agent.vector_store import build_vector_store

    query = state.get("query") or state["question"]
    specialist = state.get("specialist")
    dept = config.SPECIALIST_TO_DEPARTMENT.get(specialist) if specialist else None

    store = build_vector_store()
    if dept:
        filtered = store.as_retriever(
            search_kwargs={"k": config.RETRIEVAL_K,
                           "filter": {"department": dept}}
        )
        docs = filtered.invoke(query)
        # Fallback if the specialist's own corpus has nothing.
        if not docs:
            docs = store.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K}
            ).invoke(query)
    else:
        docs = store.as_retriever(
            search_kwargs={"k": config.RETRIEVAL_K}
        ).invoke(query)

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
    specialist = state.get("specialist", "hr")

    # If retrieval returned literally nothing, skip the LLM call.
    if not docs:
        return {
            "grade": "none",
            "grade_reason": "retrieval returned no documents",
            "path": _append_path(state, "grade_documents"),
        }

    system = dedent(f"""\
        You are a document relevance grader for the {specialist.upper()}
        specialist of a company onboarding assistant.

        Your job: decide whether the retrieved chunks contain enough
        information to answer the SPECIFIC question the user asked,
        AND whether the question is actually within the {specialist.upper()}
        specialist's scope.

        Scopes:
          - hr:       leave, PTO, benefits, attendance, approvals
          - it:       VPN, laptops, accounts, hardware, software
          - security: passwords, 2FA, data handling, incidents

        Do NOT require the chunks to cover the entire topic area. If
        the user asked one narrow question and the chunks answer it,
        the grade is "good".

        Reply with JSON:
        {{"grade": "good" | "weak" | "none" | "out_of_scope",
          "reason": "<one short sentence>"}}

        Grade meanings:
          - "good": chunks directly answer the specific question.
          - "weak": chunks are on-topic but do not contain the answer.
          - "none": chunks are unrelated to the question.
          - "out_of_scope": question is real but belongs to a DIFFERENT
            specialist (e.g. a password question routed to HR).

        Examples:

        Q: "How many days of annual leave do I get?"  [specialist=hr]
        Chunks: "Every full-time employee receives 20 days..."
        -> {{"grade": "good", "reason": "chunk states 20 days"}}

        Q: "What are the password requirements?"  [specialist=hr]
        Chunks: (HR chunks about leave and benefits)
        -> {{"grade": "out_of_scope", "reason": "password is security scope"}}

        Q: "Can I bring my dog to the office?"  [specialist=hr]
        Chunks: (unrelated HR chunks)
        -> {{"grade": "none", "reason": "not covered in HR docs"}}
    """)

    user = (
        f"Question:\n{question}\n\n"
        f"Retrieved chunks:\n{_format_docs_for_prompt(docs)}"
    )

    response = llm.invoke([SystemMessage(system), HumanMessage(user)])
    parsed = _extract_json(response.content) or {}

    grade = str(parsed.get("grade", "")).lower().strip()
    if grade not in {"good", "weak", "none", "out_of_scope"}:
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


# ==========================================================================
# STAGE 3 NODES - intent classification + tool calling for live HR data
# ==========================================================================

def supervisor_node(state: AgentState) -> dict:
    """
    Multi-agent supervisor (Feature 4).

    Two decisions in one LLM call:
      1. Which specialist should handle this? (hr / it / security)
      2. Is it a policy question or a personal_data / action?

    On a reroute (a specialist said "not my scope"), a hint is added
    so the supervisor picks a different specialist rather than picking
    the same one again.

    Reply: {"specialist": ..., "intent": ..., "reason": "..."}
    """
    llm = get_json_llm()
    question = state["question"]
    retries = state.get("supervisor_retries", 0)
    last_specialist = state.get("specialist")

    reroute_hint = ""
    if retries > 0 and last_specialist:
        reroute_hint = (
            f"\n\nIMPORTANT: The '{last_specialist}' specialist just said "
            f"this is out of their scope. Pick a DIFFERENT specialist."
        )

    system = dedent(f"""\
        You are the supervisor for a multi-agent HR onboarding assistant.

        Your job is to pick the right SPECIALIST and the right MODE for
        each question. Three specialists are available:

          - "hr": leave, PTO, benefits, onboarding, attendance,
                  approvals, manager workflows, HR support tickets
          - "it": VPN, laptops, accounts, hardware, software installs,
                  passwords for accounts, IT support tickets
          - "security": password policy, 2FA, data handling rules,
                        security incidents, security tickets

        Also decide the intent:
          - "policy": general question, answer from documents
          - "personal_data": look up live data OR perform an action
            (submit leave, open ticket, approve request, etc.)

        Reply with JSON:
        {{"specialist": "hr" | "it" | "security",
          "intent": "policy" | "personal_data",
          "reason": "<one sentence>"}}

        Examples:

        Q: "How many days of annual leave do I get?"
        -> {{"specialist": "hr", "intent": "policy", "reason": "general PTO rule"}}

        Q: "How many days do I have left?"
        -> {{"specialist": "hr", "intent": "personal_data", "reason": "own balance"}}

        Q: "What are the password requirements?"
        -> {{"specialist": "security", "intent": "policy", "reason": "security rule"}}

        Q: "Please open an IT ticket - VPN keeps dropping."
        -> {{"specialist": "it", "intent": "personal_data", "reason": "IT support ticket"}}

        Q: "How do I set up two-factor authentication?"
        -> {{"specialist": "security", "intent": "policy", "reason": "2FA setup rule"}}

        Q: "When will my laptop arrive?"
        -> {{"specialist": "it", "intent": "policy", "reason": "IT onboarding"}}

        Q: "Approve leave request 12."
        -> {{"specialist": "hr", "intent": "personal_data", "reason": "manager action"}}

        Q: "What tickets have I filed?"
        -> {{"specialist": "hr", "intent": "personal_data", "reason": "asks about own tickets"}}

        Q: "What is the status of my IT ticket?"
        -> {{"specialist": "it", "intent": "personal_data", "reason": "own ticket status"}}

        Q: "Show me my open tickets."
        -> {{"specialist": "hr", "intent": "personal_data", "reason": "list own tickets"}}
        {reroute_hint}
        {reroute_hint}
    """)

    response = llm.invoke([SystemMessage(system), HumanMessage(question)])
    parsed = _extract_json(response.content) or {}

    specialist = str(parsed.get("specialist", "")).lower().strip()
    if specialist not in set(config.KNOWN_SPECIALISTS):
        specialist = config.DEFAULT_SPECIALIST

    # On reroute, force a change if the LLM ignored the hint.
    if retries > 0 and last_specialist and specialist == last_specialist:
        alternatives = [
            s for s in config.KNOWN_SPECIALISTS if s != last_specialist
        ]
        specialist = alternatives[0]

    intent = str(parsed.get("intent", "")).lower().strip()
    if intent not in {"policy", "personal_data"}:
        intent = "policy"

    reason = str(parsed.get("reason", "")).strip() or "no reason given"

    return {
        "specialist": specialist,
        "intent": intent,
        "supervisor_reason": reason,
        "path": _append_path(state, f"supervisor[{specialist}]"),
    }


# Alias for backward compat with earlier tests / graph wiring that
# reference classify_intent_node. Any code importing the old name
# gets the new supervisor behavior transparently.
classify_intent_node = supervisor_node


def tool_call_node(state: AgentState) -> dict:
    """
    Ask the LLM which tool(s) to call to answer the user's question.

    In Stage 4 this node ONLY plans the calls. Execution happens later
    in `tool_execute_node`, after `human_approval_node` has (if needed)
    gated on human approval. Splitting plan from execution is what
    lets HITL sit between them cleanly.

    Robustness: if the small model fails to emit any tool call, fall
    back to `get_leave_balance` on the current user - the most common
    personal-data question.
    """
    from app.agent.tools import ALL_TOOLS, WRITE_TOOL_NAMES

    specialist = state.get("specialist", config.DEFAULT_SPECIALIST)
    allowed_names = config.SPECIALIST_TO_TOOLS.get(
        specialist, set(t.name for t in ALL_TOOLS)
    )
    specialist_tools = [t for t in ALL_TOOLS if t.name in allowed_names]

    llm = get_llm().bind_tools(specialist_tools)
    question = state["question"]
    user_code = state.get("current_user", "EMP-001")

    system = dedent(f"""\
        You are the {specialist.upper()} specialist of a multi-agent HR
        assistant. You have access to tools that read live data and
        (with human approval) modify records.

        The current user's employee_code is "{user_code}". Always pass
        that employee_code as the argument to tools that need one.

        Tool selection rules:
          - Use get_leave_balance / get_attendance / get_leave_requests
            when the user asks about their OWN data.
          - Use submit_leave_request when they clearly want to FILE
            a new leave request.
          - Use list_pending_approvals when a manager asks about their
            approval queue. Pass their employee_code as manager_code.
          - Use approve_leave_request / reject_leave_request when the
            user says to APPROVE or REJECT a specific request id.
            Pass their employee_code as approver_code. The tool will
            refuse if they are not the direct manager.
          - Use create_hr_ticket when the user wants to FILE a support
            ticket. Choose category="{config.SPECIALIST_TICKET_CATEGORY.get(specialist, "HR")}"
            for tickets you file as the {specialist} specialist.
          - Use list_my_tickets when the user asks about their tickets.

        You may only call tools that are actually bound to you. Do not
        try to call tools you cannot see. Do not answer personal-data
        questions from memory - the tools are the source of truth.
    """)

    response = llm.invoke([SystemMessage(system), HumanMessage(question)])
    raw_calls = getattr(response, "tool_calls", None) or []

    # Fallback if the small model failed to emit a tool call. Choose a
    # sensible default based on what the specialist can actually invoke.
    if not raw_calls:
        if "get_leave_balance" in allowed_names:
            raw_calls = [{
                "name": "get_leave_balance",
                "args": {"employee_code": user_code},
            }]
        else:
            raw_calls = [{
                "name": "list_my_tickets",
                "args": {"employee_code": user_code},
            }]

    # Normalise the shape and split reads/writes.
    tool_calls: list[dict] = []
    pending_writes: list[dict] = []
    for c in raw_calls:
        entry = {"name": c.get("name"), "args": c.get("args", {}) or {}}
        tool_calls.append(entry)
        if entry["name"] in WRITE_TOOL_NAMES:
            pending_writes.append(entry)

    return {
        "tool_calls": tool_calls,
        "pending_writes": pending_writes,
        "path": _append_path(state, "tool_call"),
    }


def human_approval_node(state: AgentState) -> dict:
    """
    Human-in-the-loop gate.

    If there are no writes pending, this is a no-op passthrough.
    If there are writes, we call `interrupt()`, which pauses the
    graph until the caller resumes with a decision:

        Command(resume={"approval": "approved" | "rejected",
                        "note": "<optional>"})

    That decision arrives back to us here on resume, and we record it
    in state for `tool_execute_node` to act on.
    """
    from langgraph.types import interrupt

    pending = state.get("pending_writes", [])
    if not pending:
        return {
            "approval": "approved",   # trivially approved: nothing to write
            "path": _append_path(state, "human_approval"),
        }

    # Ask the caller for approval. This pauses the graph.
    decision = interrupt({
        "kind": "approval_request",
        "message": (
            "The agent wants to perform the following WRITE actions. "
            "Approve or reject."
        ),
        "pending_writes": pending,
    })

    # `decision` is whatever the caller passed via Command(resume=...).
    # Be defensive about shape - a bare string is common.
    if isinstance(decision, dict):
        approval = str(decision.get("approval", "rejected")).lower()
        note = str(decision.get("note", ""))
    else:
        approval = str(decision).lower()
        note = ""

    if approval not in {"approved", "rejected"}:
        approval = "rejected"          # any garbage means "do not write"

    return {
        "approval": approval,
        "approval_note": note,
        "path": _append_path(state, "human_approval"),
    }


def tool_execute_node(state: AgentState) -> dict:
    """
    Execute the tool calls the LLM planned, respecting the approval
    decision for any writes.

    - Read tools always run.
    - Write tools run only if `approval == "approved"`. If rejected,
      we record a skipped-result so `tool_answer_node` can tell the
      user honestly what happened.
    """
    from app.agent.tools import TOOLS_BY_NAME, WRITE_TOOL_NAMES

    tool_calls = state.get("tool_calls", [])
    approval = state.get("approval", "approved")

    results: list[dict] = []
    for call in tool_calls:
        name = call.get("name")
        args = call.get("args", {}) or {}
        is_write = name in WRITE_TOOL_NAMES

        if is_write and approval != "approved":
            results.append({
                "tool": name,
                "args": args,
                "result": {
                    "skipped": True,
                    "reason": "human rejected write action",
                    "note": state.get("approval_note", ""),
                },
            })
            continue

        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            results.append({"tool": name, "error": "unknown tool"})
            continue
        try:
            output = tool.invoke(args)
        except Exception as exc:                               # noqa: BLE001
            output = {"error": f"tool raised: {exc}"}
        results.append({"tool": name, "args": args, "result": output})

    return {
        "tool_results": results,
        "path": _append_path(state, "tool_execute"),
    }


def tool_answer_node(state: AgentState) -> dict:
    """
    Take the raw tool results and turn them into a natural language
    answer for the user. Grounded strictly in what the tools returned.
    """
    llm = get_llm()
    question = state["question"]
    tool_results = state.get("tool_results", [])

    # Render results compactly for the prompt.
    rendered = json.dumps(tool_results, indent=2, default=str)

    system = dedent("""\
        You are a company HR assistant. You have just called tools to
        fetch live data for the user. Below are the tool results.

        Write a short, friendly answer to the user's question, using
        ONLY the numbers and facts in the tool results. Do not invent
        values. If a tool returned an error, apologise briefly and say
        you could not retrieve the data.

        Keep it to 1-3 sentences.
    """)

    user = f"Question:\n{question}\n\nTool results:\n{rendered}"
    response = llm.invoke([SystemMessage(system), HumanMessage(user)])

    return {
        "answer": response.content.strip(),
        "path": _append_path(state, "tool_answer"),
    }