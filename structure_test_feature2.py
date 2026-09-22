from app.database import SessionLocal
from app.models import Employee, LeaveRequest
from app.services.leave import (
    is_manager_of, list_pending_for_manager, check_balance_consistency,
    approve_leave_request as svc_approve, InvalidTransition,
)
from app.agent.tools import (
    ALL_TOOLS, TOOLS_BY_NAME, WRITE_TOOL_NAMES,
    list_pending_approvals, approve_leave_request, reject_leave_request,
)
 
 
def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"
 
 
print("Feature 2 structure test - manager approval workflow\n")
 
# --- Tool registry ------------------------------------------------------
check(
    "manager tools are registered",
    "list_pending_approvals" in TOOLS_BY_NAME
    and "approve_leave_request" in TOOLS_BY_NAME
    and "reject_leave_request" in TOOLS_BY_NAME,
)
check(
    "approve/reject are marked as WRITE tools (HITL-gated)",
    "approve_leave_request" in WRITE_TOOL_NAMES
    and "reject_leave_request" in WRITE_TOOL_NAMES,
)
check(
    "read tools stayed OUT of WRITE_TOOL_NAMES",
    "list_pending_approvals" not in WRITE_TOOL_NAMES
    and "get_leave_balance" not in WRITE_TOOL_NAMES,
)
 
db = SessionLocal()
 
# --- Seeded data has real manager_code ----------------------------------
aisha = db.query(Employee).filter(Employee.employee_code == "EMP-001").one()
check("EMP-001 has a manager_code", aisha.manager_code == "EMP-101")
ravi = db.query(Employee).filter(Employee.employee_code == "EMP-101").one()
check("manager EMP-101 exists", ravi.designation == "Engineering Manager")
 
# --- Role gate ----------------------------------------------------------
check(
    "is_manager_of: Ravi -> Aisha is True",
    is_manager_of(db, "EMP-101", aisha.id) is True,
)
check(
    "is_manager_of: Ravi -> Carlos is False",
    is_manager_of(
        db, "EMP-101",
        db.query(Employee).filter(Employee.employee_code == "EMP-003").one().id,
    ) is False,
)
check(
    "is_manager_of: unknown approver_code is False",
    is_manager_of(db, "NOBODY", aisha.id) is False,
)
check(
    "is_manager_of: empty approver_code is False",
    is_manager_of(db, "", aisha.id) is False,
)
 
# --- list_pending_for_manager --------------------------------------------
ravi_pending = list_pending_for_manager(db, "EMP-101")
check(
    "Ravi sees only his team's pending requests",
    all(r.employee_id in {
        e.id for e in db.query(Employee).filter(
            Employee.manager_code == "EMP-101"
        ).all()
    } for r in ravi_pending),
)
check(
    "list_pending_for_manager with unknown code -> empty",
    list_pending_for_manager(db, "NOBODY") == [],
)
 
# --- Tool: approve refuses when not manager -----------------------------
if ravi_pending:
    target = ravi_pending[0].id
 
    r = approve_leave_request.invoke({
        "request_id": target, "approver_code": "EMP-001",   # self
    })
    check(
        "approve REFUSED when approver is not the manager",
        r.get("error") == "authorization_failed",
    )
 
    r = approve_leave_request.invoke({
        "request_id": target, "approver_code": "EMP-102",   # wrong manager
    })
    check(
        "approve REFUSED when approver is a DIFFERENT manager",
        r.get("error") == "authorization_failed",
    )
 
    # And the target is still pending (nothing was mutated).
    still_pending = db.query(LeaveRequest).filter(
        LeaveRequest.id == target
    ).one()
    check("refused approve did NOT mutate the request",
          still_pending.status == "pending")
 
    # --- Approve succeeds when caller IS the manager ---
    r = approve_leave_request.invoke({
        "request_id": target, "approver_code": "EMP-101",
    })
    check("approve SUCCEEDED for the correct manager",
          r.get("approved") is True)
 
    # The tool ran in its own session and committed. Our long-lived
    # session in this test still has the old row cached - expire it
    # so the next query reloads from disk.
    db.expire_all()
    approved_row = db.query(LeaveRequest).filter(
        LeaveRequest.id == target
    ).one()
    check("approved row status is now 'approved'",
          approved_row.status == "approved")
 
# --- Consistency still holds after all that ------------------------------
mm = check_balance_consistency(db)
check("balance consistency invariant still holds", mm == [])
 
db.close()
 
print("\nAll Feature 2 structure checks passed.")
