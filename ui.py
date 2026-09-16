from __future__ import annotations
 
import uuid
 
import streamlit as st
 
from app.agent.graph import build_graph, start_run, resume_run
from app.database import SessionLocal
from app.models import Employee
 
 
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
# Session state
# --------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("history", [])          # chat log
    st.session_state.setdefault("pending", None)         # {thread_id, writes, state}
    st.session_state.setdefault("employee_code", "EMP-001")
 
 
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
    st.session_state["employee_code"] = codes[picked]
 
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
        st.session_state["history"] = []
        st.session_state["pending"] = None
        st.rerun()
 
 
# --------------------------------------------------------------------------
# Main chat pane
# --------------------------------------------------------------------------
st.title("Ask the onboarding agent")
 
for entry in st.session_state["history"]:
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
            st.session_state["history"].append({
                "role": "assistant",
                "content": final.get("answer", "(no answer)"),
                "state": final,
            })
            st.session_state["pending"] = None
            st.rerun()
 
 
# ---- Chat input (disabled while an approval is pending) -----------------
user_input = st.chat_input(
    "Ask a question…",
    disabled=pending is not None,
)
 
if user_input:
    st.session_state["history"].append({"role": "user", "content": user_input})
 
    thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    state, paused, pending_writes = start_run(
        question=user_input,
        current_user=st.session_state["employee_code"],
        thread_id=thread_id,
        graph=get_graph(),
    )
 
    if paused:
        # Stash and render the approval UI on next rerun.
        st.session_state["pending"] = {
            "thread_id": thread_id,
            "writes": pending_writes,
            "state": state,
        }
    else:
        st.session_state["history"].append({
            "role": "assistant",
            "content": state.get("answer", "(no answer)"),
            "state": state,
        })
 
    st.rerun()
 