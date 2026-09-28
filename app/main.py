from datetime import date, timedelta

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, Base, engine
from app import models, schemas

# Create tables on startup if they don't exist yet. (Seeding is separate.)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Minimal HRMS API",
    description="Stage 1 foundation. A thin HR data layer for the "
                "LangGraph onboarding agent to query in later stages.",
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


# ==========================================================================
# STAGE 6 - Feature 3: HR ticket endpoints
# ==========================================================================
from pydantic import BaseModel as _BaseModel
from datetime import datetime as _dt


class HRTicketCreate(_BaseModel):
    employee_code: str
    subject: str
    description: str
    category: str = "HR"
    priority: str = "medium"


class HRTicketOut(_BaseModel):
    id: int
    employee_id: int
    subject: str
    description: str
    category: str
    priority: str
    status: str
    assigned_team: str | None = None
    created_at: _dt | None = None
    resolved_at: _dt | None = None

    class Config:
        from_attributes = True


@app.post("/hr-tickets", response_model=HRTicketOut)
def create_hr_ticket_endpoint(
    payload: HRTicketCreate, db: Session = Depends(get_db)
):
    from app.services.tickets import (
        create_ticket as svc_create, TicketServiceError,
    )
    try:
        return svc_create(
            session=db,
            employee_code=payload.employee_code,
            subject=payload.subject,
            description=payload.description,
            category=payload.category,
            priority=payload.priority,
        )
    except TicketServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get(
    "/employees/{employee_code}/hr-tickets",
    response_model=list[HRTicketOut],
)
def list_employee_tickets(
    employee_code: str, db: Session = Depends(get_db)
):
    from app.services.tickets import (
        list_tickets_for_employee, TicketServiceError,
    )
    try:
        return list_tickets_for_employee(db, employee_code)
    except TicketServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/hr-tickets/{ticket_id}/resolve", response_model=HRTicketOut)
def resolve_ticket_endpoint(
    ticket_id: int, db: Session = Depends(get_db)
):
    from app.services.tickets import (
        resolve_ticket as svc_resolve,
        TicketServiceError, TicketNotFound, InvalidTicketTransition,
    )
    try:
        return svc_resolve(db, ticket_id)
    except TicketNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTicketTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except TicketServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ==========================================================================
# STAGE 6 - Feature 5: manager-facing ticket endpoints
# ==========================================================================
class TicketStatusUpdate(_BaseModel):
    new_status: str
    note: str | None = None


@app.get(
    "/managers/{manager_code}/open-tickets",
    response_model=list[HRTicketOut],
)
def list_open_tickets_endpoint(
    manager_code: str, db: Session = Depends(get_db)
):
    from app.services.tickets import list_open_tickets_for_manager
    return list_open_tickets_for_manager(db, manager_code)


@app.post(
    "/hr-tickets/{ticket_id}/update-status",
    response_model=HRTicketOut,
)
def update_ticket_status_endpoint(
    ticket_id: int,
    payload: TicketStatusUpdate,
    db: Session = Depends(get_db),
):
    from app.services.tickets import (
        update_ticket_status as svc_update,
        TicketServiceError, InvalidTicketTransition, TicketNotFound,
    )
    try:
        return svc_update(db, ticket_id, payload.new_status, note=payload.note)
    except TicketNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTicketTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except TicketServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))