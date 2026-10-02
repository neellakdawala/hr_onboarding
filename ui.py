from __future__ import annotations

import streamlit as st

from app.agent.graph import build_graph, start_run, resume_run
from app.database import SessionLocal
from app.models import Employee
from app.services.chats import (
    load_messages_for_user, append_message, clear_messages_for_user,
    build_recent_history_snippet,
)


# --------------------------------------------------------------------------
# One-time setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Onboarding Agent",
    page_icon="🧭",
    layout="centered",
)


@st.cache_resource
def get_graph():
    """Compile the graph ONCE per session; expensive to rebuild."""
    return build_graph()


@st.cache_data(ttl=60)
def list_employees() -> list[tuple[str, str, str]]:
    """Return (code, name, department) for the sidebar picker."""
    db = SessionLocal()
    try:
        rows = db.query(Employee).order_by(Employee.employee_code).all()
        return [(e.employee_code, e.full_name, e.department) for e in rows]
    finally:
        db.close()


# --------------------------------------------------------------------------
# Session state + persistent-chat helpers
# --------------------------------------------------------------------------
def _init_state() -> None:
    # In-memory cache of DB history, keyed by employee_code. Populated
    # lazily on first read so we don't hit the DB when nothing changes.
    st.session_state.setdefault("chats_by_user", {})
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("employee_code", "EMP-001")


def _current_history() -> list:
    """
    Return the message list for whoever is signed in, loading from DB
    the first time we see this user in the session.
    """
    code = st.session_state["employee_code"]
    if code not in st.session_state["chats_by_user"]:
        db = SessionLocal()
        try:
            st.session_state["chats_by_user"][code] = (
                load_messages_for_user(db, code)
            )
        finally:
            db.close()
    return st.session_state["chats_by_user"][code]


def _persist_message(role: str, content: str, state: dict | None = None) -> dict:
    """
    Append to BOTH the DB (survives restarts) and the in-memory cache
    (immediate UI reflection). Returns the entry dict for chaining.
    """
    code = st.session_state["employee_code"]
    db = SessionLocal()
    try:
        append_message(db, code, role, content, state=state)
    finally:
        db.close()

    entry: dict = {"role": role, "content": content}
    if state:
        entry["state"] = state
    _current_history().append(entry)
    return entry


def _clear_history() -> None:
    """Delete the current user's messages from DB and cache."""
    code = st.session_state["employee_code"]
    db = SessionLocal()
    try:
        clear_messages_for_user(db, code)
    finally:
        db.close()
    st.session_state["chats_by_user"][code] = []


def _thread_id_for(code: str) -> str:
    """Stable thread id per employee so LangGraph state is per-user."""
    return f"ui-{code}"


_init_state()


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def render_path_panel(state: dict) -> None:
    """Collapsible detail panel that makes the agent's work visible."""
    path = state.get("path") or []
    with st.expander("🧠 How the agent got here", expanded=False):
        st.caption("Path through the graph:")
        st.code(" → ".join(path) if path else "(no path recorded)")

        cols = st.columns(2)
        cols[0].metric("Intent", state.get("intent") or "-")
        term = path[-1] if path else "-"
        cols[1].metric("Terminal", term)

        if state.get("grade"):
            st.caption(
                f"Retrieval grade: **{state['grade']}** — "
                f"{state.get('grade_reason', '')}"
            )

        if state.get("tool_calls"):
            st.caption("Tool call(s):")
            for tc in state["tool_calls"]:
                st.code(f"{tc['name']}({tc.get('args', {})})", language="python")

        if state.get("citations"):
            st.caption("Cited sources:")
            for c in state["citations"]:
                st.markdown(f"- `{c}`")

        if state.get("escalation_team"):
            st.warning(
                f"Escalated to **{state['escalation_team']}** — "
                f"{state.get('escalation_reason', '')}"
            )


def render_assistant_turn(entry: dict) -> None:
    """One assistant message: the answer + the details panel."""
    with st.chat_message("assistant"):
        st.markdown(entry["content"])
        if entry.get("state"):
            render_path_panel(entry["state"])


def render_user_turn(entry: dict) -> None:
    with st.chat_message("user"):
        st.markdown(entry["content"])


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("HR Onboarding Agent")
    st.caption("LangGraph · Ollama · FastAPI · Chroma")

    employees = list_employees()
    labels = [f"{code} — {name} ({dept})" for code, name, dept in employees]
    codes = [code for code, _, _ in employees]
    try:
        default_idx = codes.index(st.session_state["employee_code"])
    except ValueError:
        default_idx = 0
    picked = st.selectbox(
        "Signed in as",
        options=range(len(employees)),
        format_func=lambda i: labels[i],
        index=default_idx,
    )
    new_code = codes[picked]
    if new_code != st.session_state["employee_code"]:
        # Switching users silently cancels any pending write from the
        # previous user, so an orphan Approve/Reject can't hang around.
        st.session_state["pending"] = None
    st.session_state["employee_code"] = new_code

    st.divider()
    st.caption("**Try asking:**")
    st.markdown(
        "- How many days of annual leave do I get?\n"
        "- How many days do I have left?\n"
        "- What was my attendance this week?\n"
        "- Please submit an annual leave request from "
        "2026-11-20 to 2026-11-22 for a family event."
    )

    if st.button("🗑️ Clear conversation"):
        _clear_history()
        st.session_state["pending"] = None
        st.rerun()


# --------------------------------------------------------------------------
# Main chat pane
# --------------------------------------------------------------------------
st.title("Ask the onboarding agent")

for entry in _current_history():
    if entry["role"] == "user":
        render_user_turn(entry)
    else:
        render_assistant_turn(entry)


# ---- Pending approval UI (only shown while the graph is paused) ---------
pending = st.session_state.get("pending")
if pending is not None:
    with st.chat_message("assistant"):
        st.warning(
            "🖐️ The agent wants to perform a **write action** and is "
            "waiting for your approval."
        )
        for w in pending["writes"]:
            st.caption(f"Tool: `{w['name']}`")
            st.json(w.get("args", {}))
        note = st.text_input(
            "Optional note",
            key="approval_note",
            placeholder="e.g. 'looks fine' or 'wrong dates'",
        )
        cols = st.columns(2)
        approved = cols[0].button("✅ Approve", use_container_width=True)
        rejected = cols[1].button("❌ Reject", use_container_width=True)

        if approved or rejected:
            decision = "approved" if approved else "rejected"
            final = resume_run(
                thread_id=pending["thread_id"],
                approval=decision,
                note=note,
                graph=get_graph(),
            )
            _persist_message(
                role="assistant",
                content=final.get("answer", "(no answer)"),
                state=final,
            )
            st.session_state["pending"] = None
            st.rerun()


# ---- Chat input (disabled while an approval is pending) -----------------
user_input = st.chat_input(
    "Ask a question…",
    disabled=pending is not None,
)

# Phase 1: user just typed — persist immediately and rerun so their
# message shows up before we start the slow agent call.
if user_input and user_input.strip():
    _persist_message(role="user", content=user_input.strip())
    # Mark the just-persisted entry so phase 2 knows to process it.
    _current_history()[-1]["pending_run"] = True
    st.rerun()

# Phase 2: on the rerun, actually run the agent with a visible spinner
history = _current_history()
if (
    history
    and history[-1].get("pending_run")
    and st.session_state.get("pending") is None
):
    last = history[-1]
    last["pending_run"] = False   # consume the flag (in-memory only)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            thread_id = _thread_id_for(st.session_state["employee_code"])

            # Pull the last few turns BEFORE the current one, so the
            # supervisor and tool_call nodes can resolve short follow-ups
            # like "yes", "approve that", or "make it 4 days instead".
            db_mem = SessionLocal()
            try:
                prior = load_messages_for_user(
                    db_mem, st.session_state["employee_code"]
                )[:-1][-6:]
                recent_history = "\n".join(
                    f"{('User' if m['role'] == 'user' else 'Assistant')}: "
                    f"{(m['content'] or '').strip()[:300]}"
                    for m in prior
                )
            finally:
                db_mem.close()

            state, paused, pending_writes = start_run(
                question=last["content"],
                current_user=st.session_state["employee_code"],
                thread_id=thread_id,
                graph=get_graph(),
                recent_history=recent_history,
            )

    if paused:
        st.session_state["pending"] = {
            "thread_id": thread_id,
            "writes": pending_writes,
            "state": state,
        }
    else:
        _persist_message(
            role="assistant",
            content=state.get("answer", "(no answer)"),
            state=state,
        )
    st.rerun()