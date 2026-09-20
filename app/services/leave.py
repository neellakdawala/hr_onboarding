from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.models import Employee, LeaveBalance, LeaveRequest


# --------------------------------------------------------------------------
# Custom exceptions - so callers can distinguish "bad request" from "500"
# --------------------------------------------------------------------------
class LeaveServiceError(Exception):
    """Base class for expected leave-service failures."""


class EmployeeNotFound(LeaveServiceError):
    pass


class RequestNotFound(LeaveServiceError):
    pass


class InvalidTransition(LeaveServiceError):
    """Trying to transition a request from a state that doesn't allow it."""


class BalanceNotFound(LeaveServiceError):
    """The employee has no balance row for the requested leave_type."""


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _get_employee_by_code(session: Session, code: str) -> Employee:
    emp = session.query(Employee).filter(Employee.employee_code == code).first()
    if emp is None:
        raise EmployeeNotFound(f"No employee with code '{code}'.")
    return emp


def _get_request(session: Session, request_id: int) -> LeaveRequest:
    req = session.query(LeaveRequest).filter(
        LeaveRequest.id == request_id
    ).first()
    if req is None:
        raise RequestNotFound(f"No leave request with id {request_id}.")
    return req


def _get_balance(
    session: Session, employee_id: int, leave_type: str
) -> LeaveBalance:
    bal = session.query(LeaveBalance).filter(
        LeaveBalance.employee_id == employee_id,
        LeaveBalance.leave_type == leave_type,
    ).first()
    if bal is None:
        raise BalanceNotFound(
            f"Employee {employee_id} has no '{leave_type}' balance row."
        )
    return bal


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def create_leave_request(
    session: Session,
    employee_code: str,
    leave_type: str,
    start_date: date,
    end_date: date,
    days: float,
    reason: Optional[str] = None,
) -> LeaveRequest:
    """
    Create a new leave request in status='pending'.

    Balance is NOT touched here - pending requests do not consume days.
    Days are only deducted when a request is approved.
    """
    if end_date < start_date:
        raise LeaveServiceError("end_date cannot be before start_date.")
    if days <= 0:
        raise LeaveServiceError("days must be positive.")

    emp = _get_employee_by_code(session, employee_code)
    # Confirm the balance row exists so we fail early rather than at
    # approval time.
    _get_balance(session, emp.id, leave_type)

    request = LeaveRequest(
        employee_id=emp.id,
        leave_type=leave_type,
        start_date=start_date,
        end_date=end_date,
        days=float(days),
        reason=reason,
        status="pending",
    )
    session.add(request)
    session.commit()
    session.refresh(request)
    return request


def approve_leave_request(
    session: Session, request_id: int
) -> LeaveRequest:
    """
    Transition a pending request to 'approved' and ADD its days to the
    matching balance.used_days - atomically.

    Raises InvalidTransition if the request is not currently pending.
    """
    req = _get_request(session, request_id)
    if req.status != "pending":
        raise InvalidTransition(
            f"Cannot approve a request in status '{req.status}'."
        )

    balance = _get_balance(session, req.employee_id, req.leave_type)
    balance.used_days = float(balance.used_days) + float(req.days)
    req.status = "approved"

    session.commit()
    session.refresh(req)
    return req


def reject_leave_request(
    session: Session, request_id: int, reason: Optional[str] = None
) -> LeaveRequest:
    """
    Transition a pending request to 'rejected'. Balance is not touched.
    """
    req = _get_request(session, request_id)
    if req.status != "pending":
        raise InvalidTransition(
            f"Cannot reject a request in status '{req.status}'."
        )

    req.status = "rejected"
    if reason:
        # Preserve the rejection reason alongside the original.
        req.reason = f"{req.reason or ''} | REJECTED: {reason}".strip(" |")
    session.commit()
    session.refresh(req)
    return req


def cancel_leave_request(
    session: Session, request_id: int
) -> LeaveRequest:
    """
    Cancel a request.

    - If the request was 'pending': just mark as cancelled, no balance change.
    - If the request was 'approved': mark cancelled AND subtract days from
      balance.used_days.
    - Rejected / cancelled requests cannot be cancelled again.
    """
    req = _get_request(session, request_id)
    if req.status not in {"pending", "approved"}:
        raise InvalidTransition(
            f"Cannot cancel a request in status '{req.status}'."
        )

    if req.status == "approved":
        balance = _get_balance(session, req.employee_id, req.leave_type)
        new_used = float(balance.used_days) - float(req.days)
        # Guard against pathological underflow.
        balance.used_days = max(0.0, new_used)

    req.status = "cancelled"
    session.commit()
    session.refresh(req)
    return req


# --------------------------------------------------------------------------
# Consistency check - the invariant the whole module exists to protect
# --------------------------------------------------------------------------
def check_balance_consistency(session: Session) -> list[dict]:
    """
    Walk every (employee, leave_type) balance row and confirm that
    used_days == sum(approved requests' days).

    Returns a list of MISMATCH dicts (empty means all consistent).
    Used by the consistency test and by anyone auditing the DB.
    """
    mismatches: list[dict] = []

    for bal in session.query(LeaveBalance).all():
        approved_days_sum = sum(
            float(r.days) for r in session.query(LeaveRequest).filter(
                LeaveRequest.employee_id == bal.employee_id,
                LeaveRequest.leave_type == bal.leave_type,
                LeaveRequest.status == "approved",
            ).all()
        )
        if abs(float(bal.used_days) - approved_days_sum) > 1e-6:
            mismatches.append({
                "employee_id": bal.employee_id,
                "leave_type": bal.leave_type,
                "stored_used_days": float(bal.used_days),
                "computed_from_requests": approved_days_sum,
            })
    return mismatches