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