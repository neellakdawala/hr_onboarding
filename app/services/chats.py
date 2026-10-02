from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import ChatMessage


# Fields the "How the agent got here" panel reads. Everything else is
# either non-serializable (documents, LangChain messages) or transient
# (retry counters).
_STATE_FIELDS_TO_SAVE = {
    "path",
    "intent",
    "specialist",
    "supervisor_reason",
    "grade",
    "grade_reason",
    "tool_calls",
    "citations",
    "escalation_team",
    "escalation_reason",
}


def _sanitize_state(state: Optional[dict]) -> Optional[dict]:
    """Keep only the panel-relevant, JSON-safe subset."""
    if not state:
        return None
    out: dict[str, Any] = {}
    for key in _STATE_FIELDS_TO_SAVE:
        if key not in state:
            continue
        value = state[key]
        # tool_calls contain args dicts; stringify anything non-obvious.
        try:
            json.dumps(value)   # probe serializability
        except (TypeError, ValueError):
            value = str(value)
        out[key] = value
    return out or None


def load_messages_for_user(
    session: Session, employee_code: str
) -> list[dict]:
    """
    Return every stored message for this user, oldest first, in the
    shape the UI's `history` list expects.
    """
    rows = session.query(ChatMessage).filter(
        ChatMessage.employee_code == employee_code
    ).order_by(ChatMessage.created_at, ChatMessage.id).all()

    history: list[dict] = []
    for row in rows:
        entry: dict = {"role": row.role, "content": row.content}
        if row.state_json:
            try:
                entry["state"] = json.loads(row.state_json)
            except json.JSONDecodeError:
                pass   # skip corrupt row's state, keep the text
        history.append(entry)
    return history


def append_message(
    session: Session,
    employee_code: str,
    role: str,
    content: str,
    state: Optional[dict] = None,
) -> ChatMessage:
    """Persist one turn. Called after every user or assistant message."""
    sanitized = _sanitize_state(state)
    row = ChatMessage(
        employee_code=employee_code,
        role=role,
        content=content or "",
        state_json=json.dumps(sanitized) if sanitized else None,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def clear_messages_for_user(
    session: Session, employee_code: str
) -> int:
    """Delete every message for this user. Returns count deleted."""
    count = session.query(ChatMessage).filter(
        ChatMessage.employee_code == employee_code
    ).delete()
    session.commit()
    return count


# --------------------------------------------------------------------------
# Feature 7: conversation memory for the agent
# --------------------------------------------------------------------------
def build_recent_history_snippet(
    session: Session,
    employee_code: str,
    max_turns: int = 6,
    max_chars_per_turn: int = 300,
) -> str:
    """
    Build a compact text summary of the last few turns, for the
    supervisor and tool_call prompts.

    Returns an empty string when there is no prior history. Each turn
    is truncated to `max_chars_per_turn` so the context stays small
    even if a previous answer was long.

    Format:
        [turn 1]
        User: how many days do i have left?
        Assistant: You have 15 days of annual leave remaining.
        [turn 2]
        User: submit a leave request from 2026-12-15 to 2026-12-20
        Assistant: Your request has been submitted with id 11, pending.
    """
    rows = session.query(ChatMessage).filter(
        ChatMessage.employee_code == employee_code
    ).order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(
        max_turns
    ).all()
    if not rows:
        return ""
    rows = list(reversed(rows))   # oldest first for prompt readability

    lines: list[str] = []
    for i, row in enumerate(rows, start=1):
        content = (row.content or "").strip()
        if len(content) > max_chars_per_turn:
            content = content[:max_chars_per_turn].rstrip() + "…"
        role = "User" if row.role == "user" else "Assistant"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)