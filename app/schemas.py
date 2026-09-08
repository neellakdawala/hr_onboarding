"""
Pydantic schemas: the shape of data going in and out of the API.

Kept separate from the ORM models (models.py) on purpose. The ORM models
are the database; these schemas are the public API contract. Keeping them
separate means we can change one without breaking the other.
"""
from datetime import date, datetime
from pydantic import BaseModel, ConfigDict


# ---- Employee ----
class EmployeeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employee_code: str
    full_name: str
    email: str
    department: str
    designation: str
    manager_name: str | None = None
    date_joined: date
    status: str


# ---- Leave balance ----
class LeaveBalanceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    leave_type: str
    total_days: float
    used_days: float
    remaining_days: float


# ---- Leave request ----
class LeaveRequestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    leave_type: str
    start_date: date
    end_date: date
    days: float
    reason: str | None = None
    status: str
    submitted_at: datetime


class LeaveRequestCreate(BaseModel):
    """Body for submitting a new leave request."""
    employee_code: str
    leave_type: str
    start_date: date
    end_date: date
    days: float
    reason: str | None = None


# ---- Attendance ----
class AttendanceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    work_date: date
    check_in: str | None = None
    check_out: str | None = None
    status: str
