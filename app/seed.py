"""
Seed the HRMS database with synthetic data.

Run this once before starting the API:  python -m app.seed

All data here is fake. That is completely fine for a portfolio project;
the point is that the *system* is real, not the data. Run this again any
time to reset the database to a known clean state.
"""
from datetime import date, datetime, timedelta, timezone
import random

from app.database import Base, engine, SessionLocal
from app.models import Employee, LeaveBalance, LeaveRequest, Attendance


def reset_schema():
    """Drop everything and recreate empty tables."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


# --- Synthetic employees -------------------------------------------------
EMPLOYEES = [
    # code, name, email, dept, designation, manager
    ("EMP-001", "Aisha Khan",      "aisha.khan@acme.test",   "Engineering", "Software Engineer",   "Ravi Menon"),
    ("EMP-002", "Ben Carter",      "ben.carter@acme.test",   "Engineering", "Senior Engineer",     "Ravi Menon"),
    ("EMP-003", "Carlos Diaz",     "carlos.diaz@acme.test",  "IT",          "IT Support Lead",     "Nina Park"),
    ("EMP-004", "Deepa Nair",      "deepa.nair@acme.test",   "HR",          "HR Generalist",       "Sara Lewis"),
    ("EMP-005", "Ethan Wright",    "ethan.wright@acme.test", "Finance",     "Financial Analyst",   "Omar Said"),
    ("EMP-006", "Fatima Noor",     "fatima.noor@acme.test",  "Security",    "Security Analyst",    "Nina Park"),
    ("EMP-007", "Grace Liu",       "grace.liu@acme.test",    "Engineering", "Junior Engineer",     "Ravi Menon"),
    ("EMP-008", "Hassan Ali",      "hassan.ali@acme.test",   "IT",          "Systems Admin",       "Nina Park"),
]

LEAVE_TYPES = {
    "annual": 20.0,
    "sick": 10.0,
    "unpaid": 0.0,
}


def seed():
    reset_schema()
    db = SessionLocal()
    try:
        employees = []
        for i, (code, name, email, dept, desig, mgr) in enumerate(EMPLOYEES):
            emp = Employee(
                employee_code=code,
                full_name=name,
                email=email,
                department=dept,
                designation=desig,
                manager_name=mgr,
                # Joined sometime in the last ~2 years.
                date_joined=date.today() - timedelta(days=random.randint(60, 730)),
                status="active",
            )
            db.add(emp)
            employees.append(emp)
        db.flush()  # assigns primary keys without committing yet

        # Leave balances: every employee gets each leave type, partly used.
        for emp in employees:
            for ltype, total in LEAVE_TYPES.items():
                used = round(random.uniform(0, total * 0.6), 1) if total else 0.0
                db.add(LeaveBalance(
                    employee_id=emp.id,
                    leave_type=ltype,
                    total_days=total,
                    used_days=used,
                ))

        # A few leave requests in different states.
        sample_requests = [
            (0, "annual",  5,  3, "approved", "Family vacation"),
            (1, "sick",    2,  1, "approved", "Flu"),
            (2, "annual",  10, 2, "pending",  "Trip abroad"),
            (6, "unpaid",  20, 4, "pending",  "Personal matter"),
            (4, "sick",    1,  1, "rejected", "Insufficient notice"),
        ]
        for emp_idx, ltype, days_ahead, length, status, reason in sample_requests:
            start = date.today() + timedelta(days=days_ahead)
            db.add(LeaveRequest(
                employee_id=employees[emp_idx].id,
                leave_type=ltype,
                start_date=start,
                end_date=start + timedelta(days=length - 1),
                days=float(length),
                reason=reason,
                status=status,
                submitted_at=datetime.now(timezone.utc) - timedelta(days=random.randint(1, 20)),
            ))

        # Attendance: last 7 working days for everyone.
        for emp in employees:
            for d in range(1, 8):
                day = date.today() - timedelta(days=d)
                if day.weekday() >= 5:  # skip Sat/Sun
                    continue
                roll = random.random()
                if roll < 0.08:
                    db.add(Attendance(
                        employee_id=emp.id, work_date=day,
                        check_in=None, check_out=None, status="absent",
                    ))
                elif roll < 0.25:
                    db.add(Attendance(
                        employee_id=emp.id, work_date=day,
                        check_in="09:%02d" % random.randint(31, 58),
                        check_out="18:%02d" % random.randint(0, 30),
                        status="late",
                    ))
                else:
                    db.add(Attendance(
                        employee_id=emp.id, work_date=day,
                        check_in="08:%02d" % random.randint(45, 59),
                        check_out="17:%02d" % random.randint(30, 59),
                        status="present",
                    ))

        db.commit()
        print(f"Seeded {len(employees)} employees with leave balances, "
              f"requests, and attendance.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
