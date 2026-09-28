"""
Feature 5 structure test.

Verifies the manager ticket workflow WITHOUT the LLM:
  - list_open_tickets registered as READ tool
  - update_ticket_status registered as WRITE tool
  - Role gate: manager can only act on tickets in their scope
  - list_open_tickets scoped by department (IT covers IT+Security)
  - Status transitions validated (legal + illegal)
  - Every specialist has these tools bound

Run:  python structure_test_feature5.py
"""
from datetime import datetime
from app.database import SessionLocal
from app.models import HRTicket
from app.services.tickets import (
    create_ticket, is_ticket_manager,
    list_open_tickets_for_manager, update_ticket_status,
    resolve_ticket, DEPARTMENT_TO_CATEGORIES,
    TicketServiceError, InvalidTicketTransition, TicketNotFound,
)
from app.agent.tools import (
    ALL_TOOLS, TOOLS_BY_NAME, WRITE_TOOL_NAMES,
    list_open_tickets, update_ticket_status as tool_update,
)
from app.agent import config


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


print("Feature 5 structure test - manager ticket workflow\n")

# --- Tool registry ---
check(
    "list_open_tickets registered as READ",
    "list_open_tickets" in TOOLS_BY_NAME
    and "list_open_tickets" not in WRITE_TOOL_NAMES,
)
check(
    "update_ticket_status registered as WRITE (HITL-gated)",
    "update_ticket_status" in TOOLS_BY_NAME
    and "update_ticket_status" in WRITE_TOOL_NAMES,
)

# --- Every specialist can act on tickets ---
for spec in config.KNOWN_SPECIALISTS:
    check(
        f"{spec} specialist can list_open_tickets",
        "list_open_tickets" in config.SPECIALIST_TO_TOOLS[spec],
    )
    check(
        f"{spec} specialist can update_ticket_status",
        "update_ticket_status" in config.SPECIALIST_TO_TOOLS[spec],
    )

db = SessionLocal()

# --- Seed a ticket to work on ---
t = create_ticket(
    db, employee_code="EMP-005",
    subject="Test VPN issue",
    description="Test description",
    category="IT", priority="high",
)
tid = t.id

# --- Role gate ---
check(
    "Nina (IT dept) IS ticket manager for IT ticket",
    is_ticket_manager(db, "EMP-102", tid) is True,
)
check(
    "Ravi (Engineering) is NOT ticket manager for IT ticket",
    is_ticket_manager(db, "EMP-101", tid) is False,
)
check(
    "Unknown code is NOT ticket manager",
    is_ticket_manager(db, "NOBODY", tid) is False,
)
check(
    "Empty code is NOT ticket manager",
    is_ticket_manager(db, "", tid) is False,
)

# --- list_open_tickets scoping ---
nina = list_open_tickets_for_manager(db, "EMP-102")
check(
    "Nina's open-ticket queue includes IT ticket",
    any(x.id == tid for x in nina),
)
ravi = list_open_tickets_for_manager(db, "EMP-101")
check(
    "Ravi's open-ticket queue does NOT include IT ticket",
    all(x.id != tid for x in ravi),
)

# --- Tool wrapper enforces role gate ---
r = tool_update.invoke({
    "ticket_id": tid, "manager_code": "EMP-101",
    "new_status": "resolved",
})
check(
    "tool refuses when manager dept doesn't cover ticket category",
    r.get("error") == "authorization_failed",
)

# Ticket must still be open (nothing was mutated)
db.expire_all()
still_open = db.query(HRTicket).filter(HRTicket.id == tid).one()
check("refused update did NOT mutate the ticket",
      still_open.status == "open")

# --- Tool wrapper succeeds for the right manager ---
r = tool_update.invoke({
    "ticket_id": tid, "manager_code": "EMP-102",
    "new_status": "resolved", "note": "reset client",
})
check("Nina resolves ticket successfully", r.get("updated") is True)

db.expire_all()
resolved_row = db.query(HRTicket).filter(HRTicket.id == tid).one()
check("ticket is now resolved", resolved_row.status == "resolved")
check("resolved_at is stamped", resolved_row.resolved_at is not None)
check("note was appended to description",
      "reset client" in (resolved_row.description or ""))

# --- Status validation ---
# Bogus new_status
r = tool_update.invoke({
    "ticket_id": tid, "manager_code": "EMP-102",
    "new_status": "archived",
})
check("bogus new_status is rejected", "error" in r)

# Cannot resolve an already-resolved ticket
r = tool_update.invoke({
    "ticket_id": tid, "manager_code": "EMP-102",
    "new_status": "resolved",
})
check("double-resolve is rejected", "error" in r)

# --- Unknown ticket ---
r = tool_update.invoke({
    "ticket_id": 999999, "manager_code": "EMP-102",
    "new_status": "resolved",
})
check("unknown ticket id is rejected", "error" in r)

db.close()

print("\nAll Feature 5 structure checks passed.")
