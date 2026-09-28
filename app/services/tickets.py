from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.models import Employee, HRTicket


VALID_CATEGORIES = {"HR", "IT", "Security", "Finance", "Other"}
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_STATUSES = {"open", "in_progress", "resolved"}


class TicketServiceError(Exception):
    """Base class for expected ticket-service failures."""


class TicketNotFound(TicketServiceError):
    pass


class InvalidTicketTransition(TicketServiceError):
    pass


def _get_employee_by_code(session: Session, code: str) -> Employee:
    emp = session.query(Employee).filter(Employee.employee_code == code).first()
    if emp is None:
        raise TicketServiceError(f"No employee with code '{code}'.")
    return emp


def _get_ticket(session: Session, ticket_id: int) -> HRTicket:
    t = session.query(HRTicket).filter(HRTicket.id == ticket_id).first()
    if t is None:
        raise TicketNotFound(f"No HR ticket with id {ticket_id}.")
    return t


def create_ticket(
    session: Session,
    employee_code: str,
    subject: str,
    description: str,
    category: str = "HR",
    priority: str = "medium",
    assigned_team: Optional[str] = None,
) -> HRTicket:
    """
    Open a new HR ticket. Starts in status='open' and defaults the
    assigned_team to the ticket's category if not set explicitly.
    """
    if not subject or not subject.strip():
        raise TicketServiceError("subject is required.")
    if not description or not description.strip():
        raise TicketServiceError("description is required.")
    if category not in VALID_CATEGORIES:
        raise TicketServiceError(
            f"category must be one of {sorted(VALID_CATEGORIES)}."
        )
    if priority not in VALID_PRIORITIES:
        raise TicketServiceError(
            f"priority must be one of {sorted(VALID_PRIORITIES)}."
        )

    emp = _get_employee_by_code(session, employee_code)

    ticket = HRTicket(
        employee_id=emp.id,
        subject=subject.strip(),
        description=description.strip(),
        category=category,
        priority=priority,
        status="open",
        assigned_team=assigned_team or category,
    )
    session.add(ticket)
    session.commit()
    session.refresh(ticket)
    return ticket


def list_tickets_for_employee(
    session: Session, employee_code: str
) -> list[HRTicket]:
    """Every ticket ever filed by (or on behalf of) this employee."""
    emp = _get_employee_by_code(session, employee_code)
    return sorted(
        emp_tickets(session, emp.id),
        key=lambda t: t.created_at,
        reverse=True,
    )


def emp_tickets(session: Session, employee_id: int) -> list[HRTicket]:
    return session.query(HRTicket).filter(
        HRTicket.employee_id == employee_id
    ).all()


def resolve_ticket(session: Session, ticket_id: int) -> HRTicket:
    """Close a ticket. Only tickets in open/in_progress can be resolved."""
    t = _get_ticket(session, ticket_id)
    if t.status not in {"open", "in_progress"}:
        raise InvalidTicketTransition(
            f"Cannot resolve a ticket in status '{t.status}'."
        )
    t.status = "resolved"
    t.resolved_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(t)
    return t


# --------------------------------------------------------------------------
# Feature 5: manager-facing ticket workflow
# --------------------------------------------------------------------------

# Ticket categories a department's manager can act on. IT and Security
# categories both fall under the IT department because that's how the
# seed set up Nina Park's role. Real orgs would tune this map.
DEPARTMENT_TO_CATEGORIES = {
    "IT":       {"IT", "Security"},
    "HR":       {"HR"},
    "Finance":  {"Finance"},
    "Security": {"Security"},   # for future dedicated security managers
}


def is_ticket_manager(
    session: Session, manager_code: str, ticket_id: int
) -> bool:
    """
    Role gate: can the given manager act on this ticket?

    True iff the manager's department covers the ticket's category
    (per DEPARTMENT_TO_CATEGORIES). Returns False for unknown managers,
    unknown tickets, or departments that don't cover the category.
    """
    if not manager_code:
        return False
    manager = session.query(Employee).filter(
        Employee.employee_code == manager_code
    ).first()
    if manager is None or not manager.department:
        return False
    ticket = session.query(HRTicket).filter(
        HRTicket.id == ticket_id
    ).first()
    if ticket is None:
        return False
    covered = DEPARTMENT_TO_CATEGORIES.get(manager.department, set())
    return ticket.category in covered


def list_open_tickets_for_manager(
    session: Session, manager_code: str
) -> list[HRTicket]:
    """
    All open or in-progress tickets whose category falls within this
    manager's department scope. Sorted by priority (high first) then
    created_at.
    """
    manager = session.query(Employee).filter(
        Employee.employee_code == manager_code
    ).first()
    if manager is None or not manager.department:
        return []
    covered = DEPARTMENT_TO_CATEGORIES.get(manager.department, set())
    if not covered:
        return []
    tickets = session.query(HRTicket).filter(
        HRTicket.category.in_(covered),
        HRTicket.status.in_({"open", "in_progress"}),
    ).all()
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        tickets,
        key=lambda t: (priority_order.get(t.priority, 3), t.created_at),
    )


def update_ticket_status(
    session: Session,
    ticket_id: int,
    new_status: str,
    note: Optional[str] = None,
) -> HRTicket:
    """
    Move a ticket to open / in_progress / resolved with a legal
    transition. Optional note is appended to the description.

    Legal transitions:
        open         -> in_progress, resolved
        in_progress  -> resolved, open (re-open)
        resolved     -> (nothing)  # closed for good
    """
    if new_status not in VALID_STATUSES:
        raise TicketServiceError(
            f"new_status must be one of {sorted(VALID_STATUSES)}."
        )
    t = _get_ticket(session, ticket_id)

    legal = {
        "open": {"in_progress", "resolved"},
        "in_progress": {"resolved", "open"},
        "resolved": set(),
    }
    if new_status not in legal.get(t.status, set()):
        raise InvalidTicketTransition(
            f"Cannot transition ticket from '{t.status}' to '{new_status}'."
        )

    t.status = new_status
    if note:
        t.description = f"{t.description}\n\n[{new_status.upper()}] {note}"
    if new_status == "resolved":
        t.resolved_at = datetime.now(timezone.utc)

    session.commit()
    session.refresh(t)
    return t