from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


print("Stage 1 smoke test - minimal HRMS API\n")

# Root
r = client.get("/")
check("root returns ok", r.status_code == 200 and r.json()["status"] == "ok")

# List employees
r = client.get("/employees")
employees = r.json()
check("list employees returns 8", r.status_code == 200 and len(employees) == 8)

# Get one employee
r = client.get("/employees/EMP-003")
emp = r.json()
check("get EMP-003 -> Carlos Diaz in IT",
      emp["full_name"] == "Carlos Diaz" and emp["department"] == "IT")

# Unknown employee -> 404
r = client.get("/employees/EMP-999")
check("unknown employee returns 404", r.status_code == 404)

# Leave balances
r = client.get("/employees/EMP-001/leave-balances")
balances = r.json()
has_remaining = all("remaining_days" in b for b in balances)
check("leave balances include computed remaining_days",
      r.status_code == 200 and len(balances) == 3 and has_remaining)

# Leave requests
r = client.get("/employees/EMP-001/leave-requests")
check("leave requests returns a list", r.status_code == 200)

# Create a leave request
new_req = {
    "employee_code": "EMP-007",
    "leave_type": "annual",
    "start_date": "2026-07-01",
    "end_date": "2026-07-03",
    "days": 3,
    "reason": "Smoke test request",
}
r = client.post("/leave-requests", json=new_req)
created = r.json()
check("create leave request -> pending status",
      r.status_code == 200 and created["status"] == "pending")

# Bad date range -> 400
bad_req = {**new_req, "start_date": "2026-07-10", "end_date": "2026-07-01"}
r = client.post("/leave-requests", json=bad_req)
check("invalid date range returns 400", r.status_code == 400)

# Attendance
r = client.get("/employees/EMP-002/attendance")
records = r.json()
valid_status = all(rec["status"] in ("present", "late", "absent")
                    for rec in records)
check("attendance returns records with valid status",
      r.status_code == 200 and valid_status)

# --- State transition endpoints ---
# Create -> get its id
new_req = {
    "employee_code": "EMP-008", "leave_type": "annual",
    "start_date": "2026-10-05", "end_date": "2026-10-07",
    "days": 3, "reason": "smoke transition test",
}
r = client.post("/leave-requests", json=new_req)
req_id = r.json()["id"]
check("create returned an id", isinstance(req_id, int))

# Approve should shift the balance
before = client.get("/employees/EMP-008/leave-balances").json()
annual_before = next(b["remaining_days"] for b in before if b["leave_type"] == "annual")
r = client.post(f"/leave-requests/{req_id}/approve")
check("approve returns 200", r.status_code == 200 and r.json()["status"] == "approved")
after = client.get("/employees/EMP-008/leave-balances").json()
annual_after = next(b["remaining_days"] for b in after if b["leave_type"] == "annual")
check(
    f"balance dropped by 3 on approve ({annual_before} -> {annual_after})",
    abs((annual_before - annual_after) - 3.0) < 1e-6,
)

# Approving twice is a conflict
r = client.post(f"/leave-requests/{req_id}/approve")
check("re-approve returns 409", r.status_code == 409)

# Cancel refunds the balance
r = client.post(f"/leave-requests/{req_id}/cancel")
check("cancel returns 200", r.status_code == 200)
refunded = client.get("/employees/EMP-008/leave-balances").json()
annual_refund = next(b["remaining_days"] for b in refunded if b["leave_type"] == "annual")
check(
    f"cancel refunded 3 back ({annual_after} -> {annual_refund})",
    abs(annual_refund - annual_before) < 1e-6,
)

print("\nAll Stage 1 checks passed. The HRMS foundation works.")