"""
Diagnostic: show what the retriever actually returns for each test question.

If Stage 2 still misbehaves after the grader fix, run this FIRST. It bypasses
the LLM entirely and shows you the raw retrieved chunks. That tells you
whether the problem is:

  - retrieval  (wrong chunks come back)                    -> fix chunking/embeddings/k
  - grading    (right chunks come back but graded wrong)   -> fix grader prompt
  - generation (right chunks, right grade, wrong answer)   -> fix generator prompt

Run:  python diagnose_retrieval.py
"""
from app.agent.vector_store import get_retriever


QUESTIONS = [
    "How many days of annual leave do I get per year?",
    "What are the password requirements at this company?",
    "Can I bring my dog to the office on Fridays?",
]


def main() -> None:
    retriever = get_retriever()
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        docs = retriever.invoke(q)
        print(f"  -> {len(docs)} chunks returned")
        for i, d in enumerate(docs, start=1):
            source = d.metadata.get("source", "?")
            preview = d.page_content.strip().replace("\n", " ")[:140]
            print(f"  [{i}] {source}: {preview}...")


if __name__ == "__main__":
    main()
