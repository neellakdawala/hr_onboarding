"""
Stage 4 END-TO-END smoke test.

Runs a mix of scenarios against the real LangGraph with Ollama:
  1. A pure read (no write, no interrupt) - baseline that Stage 3 still works.
  2. A write request that the "human" APPROVES  - request should be created.
  3. A write request that the "human" REJECTS   - request should NOT be created.

Prereqs:
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    python -m app.seed
    (optional) put LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env

Run:
    python smoke_test_stage4.py
"""
from app.agent.graph import run_agent_interactive
from app.agent.tracing import is_tracing_enabled
from app.agent.tools import get_leave_requests


CURRENT_USER = "EMP-005"    # Ethan - lightly used, easy to spot new requests


def format_state(final_state) -> str:
    lines = []
    lines.append(f"  path        : {' -> '.join(final_state.get('path', []))}")
    lines.append(f"  intent      : {final_state.get('intent')}")
    for tc in final_state.get("tool_calls", []) or []:
        lines.append(f"  planned     : {tc['name']}({tc.get('args', {})})")
    if final_state.get("pending_writes"):
        lines.append(f"  pending_writes: {final_state['pending_writes']}")
    if final_state.get("approval"):
        lines.append(f"  approval    : {final_state['approval']}"
                     + (f" (note: {final_state.get('approval_note')})"
                        if final_state.get("approval_note") else ""))
    for r in final_state.get("tool_results", []) or []:
        lines.append(f"  result      : {r}")
    lines.append(f"  answer      : {final_state.get('answer')}")
    return "\n".join(lines)


def count_requests(user_code: str) -> int:
    r = get_leave_requests.invoke({"employee_code": user_code})
    return len(r.get("requests", []))


def main() -> None:
    print("Stage 4 end-to-end smoke test")
    print(f"current_user = {CURRENT_USER}")
    print(f"Langfuse tracing enabled: {is_tracing_enabled()}\n")

    # --- Scenario 1: pure read, no interrupt --------------------------
    print("[scenario 1: read - baseline]")
    print("Q: How many days of annual leave do I have left?")
    state = run_agent_interactive(
        "How many days of annual leave do I have left?",
        current_user=CURRENT_USER,
        thread_id="s4-read",
    )
    print(format_state(state))
    print()

    # --- Scenario 2: write with APPROVAL ------------------------------
    before = count_requests(CURRENT_USER)
    print("[scenario 2: write - APPROVED]")
    print("Q: Please submit an annual leave request for Nov 20 to Nov 22 "
          "2026, 3 days, reason: family event.")
    state = run_agent_interactive(
        "Please submit an annual leave request for me from "
        "2026-11-20 to 2026-11-22, 3 days, reason: family event.",
        approve_writes=True,
        current_user=CURRENT_USER,
        thread_id="s4-approve",
        approval_note="looks fine",
    )
    print(format_state(state))
    after = count_requests(CURRENT_USER)
    print(f"  leave requests: before={before}, after={after} "
          f"(expected +1)")
    print()

    # --- Scenario 3: write with REJECTION -----------------------------
    before = count_requests(CURRENT_USER)
    print("[scenario 3: write - REJECTED]")
    print("Q: Please submit an unpaid leave request for Dec 15 to Dec 20 "
          "2026, 6 days.")
    state = run_agent_interactive(
        "Please submit an unpaid leave request for me from "
        "2026-12-15 to 2026-12-20, 6 days, reason: personal.",
        approve_writes=False,
        current_user=CURRENT_USER,
        thread_id="s4-reject",
        approval_note="wrong dates",
    )
    print(format_state(state))
    after = count_requests(CURRENT_USER)
    print(f"  leave requests: before={before}, after={after} "
          f"(expected UNCHANGED)")
    print()

    print("Done.")
    print("What to look for:")
    print("  * scenario 1: no human_approval interrupt (no writes planned)")
    print("  * scenario 2: request count went UP by 1")
    print("  * scenario 3: request count UNCHANGED, answer says it was skipped")


if __name__ == "__main__":
    main()
