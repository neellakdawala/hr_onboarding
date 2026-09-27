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