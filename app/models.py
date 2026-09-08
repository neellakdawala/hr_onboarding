"""
ORM models for the minimal HRMS.

Four tables, deliberately kept minimal. They exist to give the LangGraph
agent (built in later stages) real structured data to query:

  employees       - who works here
  leave_balances  - how much leave each employee has left
  leave_requests  - leave requests and their approval status
  attendance      - daily check-in / check-out records

We intentionally do NOT model payroll, roles/permissions, budgeting, etc.
Those are future scope. The agent does not need them to be interesting.
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship

from app.database import Base


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (replaces deprecated datetime.utcnow)."""
    return datetime.now(timezone.utc)


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_code = Column(String, unique=True, index=True)  # e.g. "EMP-001"
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    department = Column(String, nullable=False)   # HR, IT, Security, Finance...
    designation = Column(String, nullable=False)
    manager_name = Column(String)
    date_joined = Column(Date, nullable=False)
    status = Column(String, default="active")     # active, on_leave, exited

    # Relationships let us navigate from an employee to their records.
    leave_balances = relationship("LeaveBalance", back_populates="employee")
    leave_requests = relationship("LeaveRequest", back_populates="employee")
    attendance_records = relationship("Attendance", back_populates="employee")


class LeaveBalance(Base):
    __tablename__ = "leave_balances"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    leave_type = Column(String, nullable=False)   # annual, sick, unpaid...
    total_days = Column(Float, nullable=False)
    used_days = Column(Float, default=0.0)

    employee = relationship("Employee", back_populates="leave_balances")

    @property
    def remaining_days(self) -> float:
        """Convenience: days still available."""
        return self.total_days - self.used_days


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    leave_type = Column(String, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    days = Column(Float, nullable=False)
    reason = Column(String)
    status = Column(String, default="pending")    # pending, approved, rejected
    submitted_at = Column(DateTime, default=_utcnow)

    employee = relationship("Employee", back_populates="leave_requests")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    work_date = Column(Date, nullable=False)
    check_in = Column(String)    # stored as "HH:MM" for simplicity
    check_out = Column(String)
    status = Column(String, default="present")    # present, late, absent

    employee = relationship("Employee", back_populates="attendance_records")
