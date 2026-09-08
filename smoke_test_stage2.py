"""
Stage 2 END-TO-END smoke test.

Runs three real questions through the LangGraph agent using Ollama.
This one requires Ollama to be running locally with the models pulled:

    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    ollama list      # should show both

Then run:
    python smoke_test_stage2.py

You should see three questions run, each showing:
  - the path through the graph (which nodes fired)
  - the final answer OR the escalated team
  - citations if answered
"""
from app.agent.graph import run_agent


QUESTIONS = [
    # (label, question, expectation to eyeball in the output)
    (
        "answerable-hr",
        "How many days of annual leave do I get per year?",
        "Should retrieve HR docs and answer '20 days', citing leave_policy.md.",
    ),
    (
        "answerable-security",
        "What are the password requirements at this company?",
        "Should retrieve Security docs and answer with the 12-char rule etc.",
    ),
    (
        "out-of-scope",
        "Can I bring my dog to the office on Fridays?",
        "Should not find it in policies. Expected to escalate (probably to HR).",
    ),
]


def format_state(final_state) -> str:
    lines = []
    lines.append(f"  path        : {' -> '.join(final_state.get('path', []))}")
    lines.append(f"  grade       : {final_state.get('grade')} "
                 f"({final_state.get('grade_reason')})")
    lines.append(f"  retry_count : {final_state.get('retry_count')}")
    if final_state.get("escalation_team"):
        lines.append(f"  ESCALATED   : {final_state['escalation_team']} "
                     f"({final_state.get('escalation_reason')})")
    lines.append(f"  answer      : {final_state.get('answer')}")
    if final_state.get("citations"):
        lines.append(f"  citations   : {final_state['citations']}")
    return "\n".join(lines)


def main() -> None:
    print("Stage 2 end-to-end smoke test\n")
    for label, question, expectation in QUESTIONS:
        print(f"[{label}]")
        print(f"Q: {question}")
        print(f"Expect: {expectation}")
        final_state = run_agent(question)
        print(format_state(final_state))
        print()

    print("Done. Eyeball the output above to confirm behaviour:")
    print("  - Answerable questions should end at 'generate_answer' "
          "with citations.")
    print("  - Out-of-scope questions should end at 'escalate' with a team.")


if __name__ == "__main__":
    main()
