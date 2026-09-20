from datetime import date, timedelta

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, Base, engine
from app import models, schemas

# Create tables on startup if they don't exist yet. (Seeding is separate.)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Minimal HRMS API",
    description="LangGraph onboarding agent ",
    version="0.1.0",
)


def _get_employee_or_404(employee_code: str, db: Session) -> models.Employee:
    """Shared helper: look up an employee by code or raise 404."""
    emp = (
        db.query(models.Employee)
        .filter(models.Employee.employee_code == employee_code)
        .first()
    )
    if emp is None:
        raise HTTPException(
            status_code=404,
            detail=f"No employee with code '{employee_code}'.",
        )
    return emp


@app.get("/")
def root():
    """Health check / friendly landing response."""
    return {
        "service": "Minimal HRMS API",
        "status": "ok",
        "docs": "/docs",
    }


# ---- Employees ----------------------------------------------------------
@app.get("/employees", response_model=list[schemas.EmployeeOut])
def list_employees(db: Session = Depends(get_db)):
    """List all employees."""
    return db.query(models.Employee).all()


@app.get("/employees/{employee_code}", response_model=schemas.EmployeeOut)
def get_employee(employee_code: str, db: Session = Depends(get_db)):
    """Get a single employee by their code (e.g. EMP-001)."""
    return _get_employee_or_404(employee_code, db)


# ---- Leave balances -----------------------------------------------------
@app.get(
    "/employees/{employee_code}/leave-balances",
    response_model=list[schemas.LeaveBalanceOut],
)
def get_leave_balances(employee_code: str, db: Session = Depends(get_db)):
    """All leave balances for an employee, with remaining days computed."""
    emp = _get_employee_or_404(employee_code, db)
    return emp.leave_balances


# ---- Leave requests -----------------------------------------------------
@app.get(
    "/employees/{employee_code}/leave-requests",
    response_model=list[schemas.LeaveRequestOut],
)
def get_leave_requests(employee_code: str, db: Session = Depends(get_db)):
    """All leave requests for an employee, newest intent first."""
    emp = _get_employee_or_404(employee_code, db)
    return sorted(
        emp.leave_requests, key=lambda r: r.submitted_at, reverse=True
    )


@app.post("/leave-requests", response_model=schemas.LeaveRequestOut)
def create_leave_request(
    payload: schemas.LeaveRequestCreate, db: Session = Depends(get_db)
):
    """
    Submit a new leave request. It starts in 'pending' status.
    """
    from app.services.leave import (
        create_leave_request as svc_create, LeaveServiceError,
    )
    try:
        return svc_create(
            session=db,
            employee_code=payload.employee_code,
            leave_type=payload.leave_type,
            start_date=payload.start_date,
            end_date=payload.end_date,
            days=payload.days,
            reason=payload.reason,
        )
    except LeaveServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/leave-requests/{request_id}/approve",
    response_model=schemas.LeaveRequestOut,
)
def approve_leave_request_endpoint(
    request_id: int, db: Session = Depends(get_db)
):
    """
    Approve a pending leave request.

    Atomically transitions status to 'approved' AND adds the request's
    days to the matching leave_balance.used_days. If either fails, both
    roll back - the invariant is maintained.
    """
    from app.services.leave import (
        approve_leave_request as svc_approve,
        LeaveServiceError, InvalidTransition, RequestNotFound,
    )
    try:
        return svc_approve(db, request_id)
    except RequestNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except LeaveServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/leave-requests/{request_id}/reject",
    response_model=schemas.LeaveRequestOut,
)
def reject_leave_request_endpoint(
    request_id: int,
    reason: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Reject a pending leave request. Balance is not touched.
    """
    from app.services.leave import (
        reject_leave_request as svc_reject,
        LeaveServiceError, InvalidTransition, RequestNotFound,
    )
    try:
        return svc_reject(db, request_id, reason=reason)
    except RequestNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except LeaveServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/leave-requests/{request_id}/cancel",
    response_model=schemas.LeaveRequestOut,
)
def cancel_leave_request_endpoint(
    request_id: int, db: Session = Depends(get_db)
):
    """
    Cancel a request. If it was approved, its days are refunded to
    the balance in the same transaction.
    """
    from app.services.leave import (
        cancel_leave_request as svc_cancel,
        LeaveServiceError, InvalidTransition, RequestNotFound,
    )
    try:
        return svc_cancel(db, request_id)
    except RequestNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except LeaveServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---- Attendance ---------------------------------------------------------
@app.get(
    "/employees/{employee_code}/attendance",
    response_model=list[schemas.AttendanceOut],
)
def get_attendance(
    employee_code: str, days: int = 7, db: Session = Depends(get_db)
):
    """
    Attendance records for an employee over the last `days` days
    (default 7). Useful for questions like 'what was my attendance
    this week'.
    """
    emp = _get_employee_or_404(employee_code, db)
    cutoff = date.today() - timedelta(days=days)
    recent = [r for r in emp.attendance_records if r.work_date >= cutoff]
    return sorted(recent, key=lambda r: r.work_date, reverse=True)