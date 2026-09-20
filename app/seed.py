from datetime import date, datetime, timedelta, timezone
import random

from app.database import Base, engine, SessionLocal
from app.models import Employee, LeaveBalance, LeaveRequest, Attendance
from app.services.leave import check_balance_consistency


# Fixed seed so the same eval / demo produces the same data every time.
_RNG = random.Random(20260916)


def reset_schema():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


# --- Synthetic employees ------------------------------------------------
# Columns: code, name, email, dept, designation, manager
EMPLOYEES = [
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
    "sick":   10.0,
    "unpaid": 0.0,
}


def seed():
    reset_schema()
    db = SessionLocal()
    try:
        # ---- Employees ----
        employees: list[Employee] = []
        for i, (code, name, email, dept, desig, mgr) in enumerate(EMPLOYEES):
            emp = Employee(
                employee_code=code,
                full_name=name,
                email=email,
                department=dept,
                designation=desig,
                manager_name=mgr,
                date_joined=date.today() - timedelta(
                    days=_RNG.randint(60, 730)
                ),
                status="active",
            )
            db.add(emp)
            employees.append(emp)
        db.flush()

        # ---- Balances - all start at 0 used, we'll fill from approvals ----
        for emp in employees:
            for ltype, total in LEAVE_TYPES.items():
                db.add(LeaveBalance(
                    employee_id=emp.id,
                    leave_type=ltype,
                    total_days=total,
                    used_days=0.0,          # will be updated by approvals
                ))
        db.flush()

        # ---- Leave requests (a mix of statuses) ----
        # (emp_idx, leave_type, days_ahead, length, initial_status, reason)
        sample_requests = [
            # These will be APPROVED - they will move the balance.
            (0, "annual",  5,  3, "approved", "Family vacation"),
            (0, "sick",    1,  1, "approved", "Migraine"),
            (1, "sick",    2,  1, "approved", "Flu"),
            (3, "annual",  20, 2, "approved", "Long weekend"),
            (4, "annual",  40, 5, "approved", "Wedding"),
            (7, "sick",    3,  2, "approved", "Recovery"),
            # These stay PENDING - balance is unaffected.
            (2, "annual",  10, 2, "pending",  "Trip abroad"),
            (6, "unpaid",  20, 4, "pending",  "Personal matter"),
            # This one is REJECTED - balance is unaffected.
            (4, "sick",    1,  1, "rejected", "Insufficient notice"),
        ]

        for emp_idx, ltype, days_ahead, length, status, reason in sample_requests:
            start = date.today() + timedelta(days=days_ahead)
            end = start + timedelta(days=length - 1)
            req = LeaveRequest(
                employee_id=employees[emp_idx].id,
                leave_type=ltype,
                start_date=start,
                end_date=end,
                days=float(length),
                reason=reason,
                status="pending",  # start pending; will approve below
                submitted_at=datetime.now(timezone.utc)
                             - timedelta(days=_RNG.randint(1, 20)),
            )
            db.add(req)
            db.flush()

            # If the seed says approved, transition it AND update balance
            # in the same transaction, so the invariant holds.
            if status == "approved":
                bal = db.query(LeaveBalance).filter(
                    LeaveBalance.employee_id == req.employee_id,
                    LeaveBalance.leave_type == req.leave_type,
                ).one()
                bal.used_days = float(bal.used_days) + float(req.days)
                req.status = "approved"
            elif status == "rejected":
                req.status = "rejected"
            # pending: leave as-is

        db.commit()

        # ---- Attendance: last 7 working days for everyone ----
        for emp in employees:
            for d in range(1, 8):
                day = date.today() - timedelta(days=d)
                if day.weekday() >= 5:
                    continue
                roll = _RNG.random()
                if roll < 0.08:
                    db.add(Attendance(
                        employee_id=emp.id, work_date=day,
                        check_in=None, check_out=None, status="absent",
                    ))
                elif roll < 0.25:
                    db.add(Attendance(
                        employee_id=emp.id, work_date=day,
                        check_in="09:%02d" % _RNG.randint(31, 58),
                        check_out="18:%02d" % _RNG.randint(0, 30),
                        status="late",
                    ))
                else:
                    db.add(Attendance(
                        employee_id=emp.id, work_date=day,
                        check_in="08:%02d" % _RNG.randint(45, 59),
                        check_out="17:%02d" % _RNG.randint(30, 59),
                        status="present",
                    ))
        db.commit()

        # ---- Sanity check the invariant before we finish ----
        mismatches = check_balance_consistency(db)
        if mismatches:
            raise RuntimeError(
                f"Seed produced inconsistent balances: {mismatches}"
            )

        print(f"Seeded {len(employees)} employees, "
              f"{len(sample_requests)} leave requests, "
              f"balances consistent.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()