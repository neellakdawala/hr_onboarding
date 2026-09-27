"""
Feature 3 structure test.

Verifies HR-ticket workflow WITHOUT the LLM:
  - hr_tickets table exists
  - create_ticket produces a valid ticket
  - list_tickets_for_employee is scoped (users don't see others' tickets)
  - Invalid inputs (empty subject, bad category, unknown employee)
    return errors, not exceptions
  - create_hr_ticket is registered as a WRITE tool (HITL-gated)
  - list_my_tickets is registered as a READ tool
  - resolve_ticket moves a ticket to 'resolved' with a resolved_at stamp
  - Re-resolving a resolved ticket raises InvalidTicketTransition

Run:  python structure_test_feature3.py
"""
from app.database import SessionLocal
from app.models import HRTicket
from app.services.tickets import (
    create_ticket, list_tickets_for_employee, resolve_ticket,
    TicketServiceError, InvalidTicketTransition, TicketNotFound,
)
from app.agent.tools import (
    ALL_TOOLS, TOOLS_BY_NAME, WRITE_TOOL_NAMES,
    create_hr_ticket, list_my_tickets,
)


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


print("Feature 3 structure test - HR ticket creation + lookup\n")

# --- Tool registry ---
check(
    "create_hr_ticket registered as WRITE tool (HITL-gated)",
    "create_hr_ticket" in WRITE_TOOL_NAMES
    and "create_hr_ticket" in TOOLS_BY_NAME,
)
check(
    "list_my_tickets registered as READ tool",
    "list_my_tickets" in TOOLS_BY_NAME
    and "list_my_tickets" not in WRITE_TOOL_NAMES,
)

db = SessionLocal()

# --- Create + list ---
t = create_ticket(
    db,
    employee_code="EMP-005",
    subject="Test subject",
    description="Test description",
    category="IT",
    priority="high",
)
check("create_ticket returns a persisted row", t.id is not None)
check("assigned_team defaults to category", t.assigned_team == "IT")
check("status starts open", t.status == "open")

tickets_ethan = list_tickets_for_employee(db, "EMP-005")
check("EMP-005 sees their own ticket", len(tickets_ethan) >= 1)

# Ticket scoping
tickets_fatima = list_tickets_for_employee(db, "EMP-006")
check(
    "EMP-006 does NOT see EMP-005's ticket (proper scoping)",
    all(x.employee_id != t.employee_id for x in tickets_fatima),
)

# --- Invalid inputs ---
try:
    create_ticket(db, "EMP-005", "", "some description", "HR", "medium")
    check("empty subject raises", False)
except TicketServiceError:
    check("empty subject raises TicketServiceError", True)

try:
    create_ticket(db, "EMP-005", "sub", "desc", "BOGUS", "medium")
    check("bad category raises", False)
except TicketServiceError:
    check("bad category raises TicketServiceError", True)

try:
    create_ticket(db, "EMP-005", "sub", "desc", "HR", "critical")
    check("bad priority raises", False)
except TicketServiceError:
    check("bad priority raises TicketServiceError", True)

try:
    create_ticket(db, "NOBODY", "sub", "desc")
    check("unknown employee raises", False)
except TicketServiceError:
    check("unknown employee raises TicketServiceError", True)

# --- Tool wrapper returns error dicts, doesn't raise ---
r = create_hr_ticket.invoke({
    "employee_code": "EMP-005", "subject": "", "description": "d",
})
check("tool returns error dict, not exception, for bad input",
      "error" in r)

# --- Resolve workflow ---
db.expire_all()
t_row = db.query(HRTicket).filter(HRTicket.id == t.id).one()
resolved = resolve_ticket(db, t_row.id)
check("resolve moves ticket to 'resolved'", resolved.status == "resolved")
check("resolve stamps resolved_at", resolved.resolved_at is not None)

try:
    resolve_ticket(db, resolved.id)
    check("re-resolve raises", False)
except InvalidTicketTransition:
    check("re-resolve raises InvalidTicketTransition", True)

# Unknown id
try:
    resolve_ticket(db, 999999)
    check("unknown ticket raises", False)
except TicketNotFound:
    check("unknown ticket raises TicketNotFound", True)

db.close()

print("\nAll Feature 3 structure checks passed.")
