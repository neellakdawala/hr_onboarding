"""
FastAPI app for the minimal HRMS.

These endpoints are the surface the LangGraph agent's tools will call in a
later stage. For now they are a normal REST API you can browse yourself.

Run the API:
    uvicorn app.main:app --reload
Then open:
    http://127.0.0.1:8000/docs       (interactive API docs)
"""
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
    This is the one write endpoint the agent's tools may use later
    (e.g. to file a request on a new hire's behalf).
    """
    emp = _get_employee_or_404(payload.employee_code, db)

    if payload.end_date < payload.start_date:
        raise HTTPException(
            status_code=400, detail="end_date cannot be before start_date."
        )

    request = models.LeaveRequest(
        employee_id=emp.id,
        leave_type=payload.leave_type,
        start_date=payload.start_date,
        end_date=payload.end_date,
        days=payload.days,
        reason=payload.reason,
        status="pending",
    )
    db.add(request)
    db.commit()
    db.refresh(request)
    return request


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
