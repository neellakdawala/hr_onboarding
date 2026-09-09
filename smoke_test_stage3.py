from app.agent.graph import run_agent


# The "current user" for personal-data questions. Change this to test
# other employees (EMP-001 through EMP-008 exist in seed data).
CURRENT_USER = "EMP-001"

QUESTIONS = [
    # (label, question, expected_intent, notes)
    (
        "policy-annual",
        "How many days of annual leave do I get per year?",
        "policy",
        "Should retrieve leave_policy.md, answer '20 days'.",
    ),
    (
        "personal-balance",
        "How many days of annual leave do I have left?",
        "personal_data",
        "Should call get_leave_balance and answer with the remaining days.",
    ),
    (
        "personal-attendance",
        "What was my attendance like this week?",
        "personal_data",
        "Should call get_attendance and answer with present/late/absent counts.",
    ),
    (
        "personal-requests",
        "Did any of my leave requests get approved?",
        "personal_data",
        "Should call get_leave_requests and answer based on statuses.",
    ),
    (
        "policy-password",
        "What are the password requirements at this company?",
        "policy",
        "Should retrieve security_policy.md, answer with the rules.",
    ),
]


def format_state(final_state) -> str:
    lines = []
    lines.append(f"  path        : {' -> '.join(final_state.get('path', []))}")
    lines.append(f"  intent      : {final_state.get('intent')}")
    if final_state.get("tool_calls"):
        for tc in final_state["tool_calls"]:
            lines.append(f"  tool_call   : {tc['name']}({tc.get('args', {})})")
    if final_state.get("grade"):
        lines.append(f"  grade       : {final_state.get('grade')} "
                     f"({final_state.get('grade_reason')})")
    if final_state.get("escalation_team"):
        lines.append(f"  ESCALATED   : {final_state['escalation_team']}")
    lines.append(f"  answer      : {final_state.get('answer')}")
    if final_state.get("citations"):
        lines.append(f"  citations   : {final_state['citations']}")
    return "\n".join(lines)


def main() -> None:
    print(f"Stage 3 end-to-end smoke test (current_user = {CURRENT_USER})\n")
    for label, question, expected_intent, notes in QUESTIONS:
        print(f"[{label}]  (expect intent = {expected_intent})")
        print(f"Q: {question}")
        print(f"Expect: {notes}")
        final_state = run_agent(question, current_user=CURRENT_USER)
        print(format_state(final_state))
        print()

    print("Done. Confirm the paths look right:")
    print("  - policy questions     -> classify_intent -> retrieve -> ...")
    print("  - personal questions   -> classify_intent -> tool_call -> tool_answer")


if __name__ == "__main__":
    main()
