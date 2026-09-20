from app.database import SessionLocal
from app.services.leave import check_balance_consistency


def check(label, condition):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    assert condition, f"FAILED: {label}"


print("Domain consistency test\n")

db = SessionLocal()
try:
    mismatches = check_balance_consistency(db)
    if mismatches:
        print("MISMATCHES found:")
        for m in mismatches:
            print(f"  employee_id={m['employee_id']} "
                  f"type={m['leave_type']}: "
                  f"stored={m['stored_used_days']} "
                  f"computed={m['computed_from_requests']}")
    check("no balance/request mismatches", len(mismatches) == 0)
finally:
    db.close()

print("\nDomain invariant holds: leave balance consistent with requests.")