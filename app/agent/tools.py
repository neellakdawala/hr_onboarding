from __future__ import annotations
from datetime import date, timedelta
from langchain_core.tools import tool
from app.database import SessionLocal
from app.models import LeaveRequest, Employee, LeaveBalance, Attendance

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
         return {"error": f"Employee {employee_code} not found."}
       
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