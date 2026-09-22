from __future__ import annotations
 
from datetime import date, timedelta
 
from langchain_core.tools import tool
 
from app.database import SessionLocal
from app.models import Employee, LeaveBalance, LeaveRequest, Attendance
 
 
# --------------------------------------------------------------------------
# Internal helper: session lifecycle
# --------------------------------------------------------------------------
def _with_session(fn):
    """Run `fn(session)` with a fresh SQLAlchemy session, always closed."""
    session = SessionLocal()
    try:
        return fn(session)
    finally:
        session.close()
 
 
def _get_employee(session, employee_code: str) -> Employee | None:
    return (
        session.query(Employee)
        .filter(Employee.employee_code == employee_code)
        .first()
    )
 
 
# --------------------------------------------------------------------------
# Tools exposed to the LLM
# --------------------------------------------------------------------------
@tool
def get_leave_balance(employee_code: str, leave_type: str | None = None) -> dict:
    """
    Look up the remaining leave balance for an employee.
 
    Args:
        employee_code: The employee's code, e.g. "EMP-001".
        leave_type: Optional. One of "annual", "sick", "unpaid". If omitted,
                    balances for all leave types are returned.
 
    Returns:
        A dict with the employee's leave balances.
    """
    def _run(session):
        emp = _get_employee(session, employee_code)
        if emp is None:
            return {"error": f"No employee with code '{employee_code}'."}
 
        balances = emp.leave_balances
        if leave_type:
            balances = [b for b in balances if b.leave_type == leave_type]
            if not balances:
                return {
                    "error": f"No '{leave_type}' balance found for "
                             f"{employee_code}."
                }
 
        return {
            "employee_code": emp.employee_code,
            "employee_name": emp.full_name,
            "balances": [
                {
                    "leave_type": b.leave_type,
                    "total_days": b.total_days,
                    "used_days": b.used_days,
                    "remaining_days": b.remaining_days,
                }
                for b in balances   
            ],
        }
 
    return _with_session(_run)
 
 
@tool
def get_leave_requests(employee_code: str) -> dict:
    """
    Look up the leave requests submitted by an employee, newest first.
 
    Args:
        employee_code: The employee's code, e.g. "EMP-001".
 
    Returns:
        A dict listing the employee's leave requests with status.
    """
    def _run(session):
        emp = _get_employee(session, employee_code)
        if emp is None:
            return {"error": f"No employee with code '{employee_code}'."}
 
        requests = sorted(
            emp.leave_requests,
            key=lambda r: r.submitted_at,
            reverse=True,
        )
        return {
            "employee_code": emp.employee_code,
            "employee_name": emp.full_name,
            "requests": [
                {
                    "leave_type": r.leave_type,
                    "start_date": r.start_date.isoformat(),
                    "end_date": r.end_date.isoformat(),
                    "days": r.days,
                    "reason": r.reason,
                    "status": r.status,
                }
                for r in requests
            ],
        }
 
    return _with_session(_run)
 
 
@tool
def get_attendance(employee_code: str, days: int = 7) -> dict:
    """
    Look up an employee's recent attendance records.
 
    Args:
        employee_code: The employee's code, e.g. "EMP-001".
        days: Number of days back to look (default 7).
 
    Returns:
        A dict listing attendance records with check-in/out and status.
    """
    def _run(session):
        emp = _get_employee(session, employee_code)
        if emp is None:
            return {"error": f"No employee with code '{employee_code}'."}
 
        cutoff = date.today() - timedelta(days=days)
        recent = sorted(
            [r for r in emp.attendance_records if r.work_date >= cutoff],
            key=lambda r: r.work_date,
            reverse=True,
        )
        present = sum(1 for r in recent if r.status == "present")
        late = sum(1 for r in recent if r.status == "late")
        absent = sum(1 for r in recent if r.status == "absent")
 
        return {
            "employee_code": emp.employee_code,
            "employee_name": emp.full_name,
            "days_covered": days,
            "summary": {"present": present, "late": late, "absent": absent},
            "records": [
                {
                    "date": r.work_date.isoformat(),
                    "check_in": r.check_in,
                    "check_out": r.check_out,
                    "status": r.status,
                }
                for r in recent
            ],
        }
 
    return _with_session(_run)
 
 
# The list every tool-calling node imports and binds to the LLM.
ALL_TOOLS = [get_leave_balance, get_leave_requests, get_attendance]
 
# Look-up by name so we can dispatch tool calls from the LLM response.
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
 
 
# ==========================================================================
# STAGE 4 - write tool (gated by human-in-the-loop approval in the graph)
# ==========================================================================
 
@tool
def submit_leave_request(
    employee_code: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    days: float,
    reason: str | None = None,
) -> dict:
    """
    Submit a new leave request on behalf of an employee.
 
    This is a WRITE action. It creates a real row in the leave_requests
    table with status 'pending'. In the graph, this tool cannot be
    invoked without human approval.
 
    Args:
        employee_code: The employee's code, e.g. "EMP-001".
        leave_type: One of "annual", "sick", "unpaid".
        start_date: ISO date "YYYY-MM-DD" for the first day of leave.
        end_date:   ISO date "YYYY-MM-DD" for the last day of leave.
        days: Number of leave days (allow half-days like 1.5).
        reason: Optional short reason.
 
    Returns:
        Dict describing the created request, or an error.
    """
    from datetime import date as _date
    from app.services.leave import (
        create_leave_request, LeaveServiceError,
    )
 
    def _run(session):
        try:
            start = _date.fromisoformat(start_date)
            end = _date.fromisoformat(end_date)
        except ValueError as e:
            return {"error": f"Invalid date format: {e}"}
 
        try:
            request = create_leave_request(
                session=session,
                employee_code=employee_code,
                leave_type=leave_type,
                start_date=start,
                end_date=end,
                days=float(days),
                reason=reason,
            )
        except LeaveServiceError as exc:
            return {"error": str(exc)}
 
        return {
            "created": True,
            "request_id": request.id,
            "employee_code": employee_code,
            "leave_type": request.leave_type,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "days": request.days,
            "status": request.status,
        }
 
    return _with_session(_run)
 
 
# Extend the tool registry. Order matters for LLM presentation only.
ALL_TOOLS = [
    get_leave_balance,
    get_leave_requests,
    get_attendance,
    submit_leave_request,
]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
 
# The set of tools that MUTATE data. The graph interrupts for these.
# Add any future write tools here - it is the single source of truth.
WRITE_TOOL_NAMES = {"submit_leave_request"}
 
 
# ==========================================================================
# STAGE 6 - Feature 2: Manager approval workflow
# ==========================================================================
 
@tool
def list_pending_approvals(manager_code: str) -> dict:
    """
    List all pending leave requests submitted by employees who report
    to the given manager.
 
    Use this ONLY when the user IS a manager and is asking about
    requests awaiting their approval - e.g. "what needs my approval?",
    "any pending leave requests from my team?".
 
    Args:
        manager_code: The manager's employee code, e.g. "EMP-101".
 
    Returns:
        Dict listing pending requests, each with id, employee, dates.
    """
    from app.services.leave import list_pending_for_manager
 
    def _run(session):
        pending = list_pending_for_manager(session, manager_code)
        rows = []
        for req in pending:
            emp = _get_employee(session, None) if False else None  # noqa
            emp = session.query(Employee).filter(
                Employee.id == req.employee_id
            ).first()
            rows.append({
                "request_id": req.id,
                "employee_code": emp.employee_code if emp else "?",
                "employee_name": emp.full_name if emp else "?",
                "leave_type": req.leave_type,
                "start_date": req.start_date.isoformat(),
                "end_date": req.end_date.isoformat(),
                "days": req.days,
                "reason": req.reason,
            })
        return {
            "manager_code": manager_code,
            "pending_count": len(rows),
            "pending_requests": rows,
        }
 
    return _with_session(_run)
 
 
@tool
def approve_leave_request(request_id: int, approver_code: str) -> dict:
    """
    Approve a pending leave request. WRITE action - requires human approval.
 
    Enforces a role check: the approver_code MUST be the manager of the
    employee who submitted the request, otherwise the tool refuses.
 
    Args:
        request_id: The id of the leave request to approve.
        approver_code: The employee code of the person approving
                       (typically the current user, if they are a manager).
 
    Returns:
        Dict describing the approved request, or an error explaining why
        approval was refused.
    """
    from app.services.leave import (
        approve_leave_request as svc_approve,
        is_manager_of, LeaveServiceError, InvalidTransition, RequestNotFound,
    )
 
    def _run(session):
        try:
            req = session.query(LeaveRequest).filter(
                LeaveRequest.id == request_id
            ).first()
            if req is None:
                return {"error": f"No leave request with id {request_id}."}
 
            # Role gate - the whole point of this tool.
            if not is_manager_of(session, approver_code, req.employee_id):
                return {
                    "error": "authorization_failed",
                    "message": (
                        f"'{approver_code}' is not authorized to approve "
                        f"leave request {request_id}. Only the requesting "
                        f"employee's direct manager may approve."
                    ),
                }
 
            result = svc_approve(session, request_id)
            return {
                "approved": True,
                "request_id": result.id,
                "employee_id": result.employee_id,
                "leave_type": result.leave_type,
                "days": result.days,
                "status": result.status,
            }
        except RequestNotFound as exc:
            return {"error": str(exc)}
        except InvalidTransition as exc:
            return {"error": str(exc)}
        except LeaveServiceError as exc:
            return {"error": str(exc)}
 
    return _with_session(_run)
 
 
@tool
def reject_leave_request(
    request_id: int, approver_code: str, reason: str | None = None
) -> dict:
    """
    Reject a pending leave request. WRITE action - requires human approval.
 
    Enforces the same role check as approve_leave_request: only the
    requesting employee's direct manager may reject.
 
    Args:
        request_id: The id of the leave request to reject.
        approver_code: The employee code of the person rejecting.
        reason: Optional short rejection reason attached to the record.
    """
    from app.services.leave import (
        reject_leave_request as svc_reject,
        is_manager_of, LeaveServiceError, InvalidTransition, RequestNotFound,
    )
 
    def _run(session):
        try:
            req = session.query(LeaveRequest).filter(
                LeaveRequest.id == request_id
            ).first()
            if req is None:
                return {"error": f"No leave request with id {request_id}."}
 
            if not is_manager_of(session, approver_code, req.employee_id):
                return {
                    "error": "authorization_failed",
                    "message": (
                        f"'{approver_code}' is not authorized to reject "
                        f"leave request {request_id}."
                    ),
                }
 
            result = svc_reject(session, request_id, reason=reason)
            return {
                "rejected": True,
                "request_id": result.id,
                "status": result.status,
            }
        except RequestNotFound as exc:
            return {"error": str(exc)}
        except InvalidTransition as exc:
            return {"error": str(exc)}
        except LeaveServiceError as exc:
            return {"error": str(exc)}
 
    return _with_session(_run)
 
 
# Register the new tools. Order shown to the LLM matters only for
# discoverability - it doesn't change routing.
ALL_TOOLS = [
    get_leave_balance,
    get_leave_requests,
    get_attendance,
    list_pending_approvals,
    submit_leave_request,
    approve_leave_request,
    reject_leave_request,
]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
 
# Approval and rejection are writes too - they change status + balance.
WRITE_TOOL_NAMES = {
    "submit_leave_request",
    "approve_leave_request",
    "reject_leave_request",
}
 