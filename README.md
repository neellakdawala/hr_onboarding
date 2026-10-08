# HR Onboarding Assistant

A multi-agent HR onboarding assistant built with LangGraph, FastAPI, and
a local LLM. Three department specialists (HR, IT, Security) sit behind
a supervisor that routes every question to the right one. Writes go
through a human-in-the-loop approval step. Role authorization is
enforced below the LLM, so the system refuses actions even when the
model is willing to try.

Everything runs locally — no API keys, no cloud bills. One
`docker compose up` and the whole stack is live.

---
## Demo

https://github.com/user-attachments/assets/87f1af7c-4283-48d2-8426-334254c030e0

## At a glance

|                   |                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------- |
| **Stack**         | LangGraph · FastAPI · SQLAlchemy · Chroma · Ollama (Qwen 2.5 7B) · Streamlit · Langfuse |
| **Specialists**   | HR · IT · Security (routed by a supervisor agent)                                       |
| **Agent tools**   | 11 total — 6 reads, 5 writes (HITL-gated)                                               |
| **Eval accuracy** | TODO% overall · TODO% intent · TODO% escalation precision (16 cases)                    |
| **Avg latency**   | TODO s per question (3B/7B comparable on local M-series)                                |
| **Tests**         | 10 structure + consistency tests, every one green in CI                                 |
| **Lines of code** | ~2,000 Python, ~400 Markdown docs/README                                                |

> This is a **portfolio project** demonstrating agentic-AI patterns —
> multi-agent routing, HITL for writes, role-gated authorization,
> grounded RAG, atomic domain transitions, consistency checks in CI,
> local observability. It is not a production HRIS.

---

## What makes it interesting

Three things most portfolio agents don't do:

**1. Capability boundaries at the tool layer, not the prompt.** Each
specialist is bound to a _subset_ of tools when the LLM is called. The
IT specialist literally cannot invoke `submit_leave_request` because
that tool isn't in its binding — not because the prompt tells it not to.
That's a real safety property, not a hopeful instruction.

**2. Role authorization below the model.** Write tools check role
membership _inside the tool itself_, before any state change. A manager
asking to approve a request from someone outside their team gets
`{"error": "authorization_failed"}`. The LLM can't argue its way around it.

**3. Domain invariant enforced in CI.** Leave balances must equal the
sum of approved requests' days. There's a service module that enforces
this atomically on every status transition, and a consistency test in
CI that walks the database and verifies the invariant. If any future
code path breaks it, the build fails.

---

## Architecture

```
                            ┌──────────┐
    user question  ───────> │ supervisor│  picks specialist + intent
                            └─────┬─────┘  {hr, it, security} × {policy, personal_data}
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
          intent=policy                 intent=personal_data
                 │                                 │
                 v                                 v
          ┌──────────┐                      ┌──────────┐
          │ retrieve │  filtered by          │ tool_call│  bound to specialist's
          │  (Chroma)│  specialist dept      └─────┬────┘  tool subset only
          └─────┬────┘                             │
                v                                  v
         ┌──────────┐    out_of_scope       ┌──────────────┐
         │   grade  │─────────────────> ⟲   │human_approval│  interrupts on writes
         └─────┬────┘  (reroute, bounded)   └──────┬───────┘
         good  │ weak                              │
               v  └──> rewrite_query → retrieve    v
         ┌──────────┐                       ┌──────────────┐
         │ generate │                       │ tool_execute │  role gate inside
         │  answer  │                       └──────┬───────┘  the tool
         └────┬─────┘                              v
              │                             ┌──────────────┐
              v                             │  tool_answer │
             END                            └──────┬───────┘
                                                   v
                                                  END

4 agentic decision points: supervisor (specialist+intent),
grader (good/weak/none/out_of_scope), human_approval (writes),
reroute loop (specialist can hand back, bounded to 2 retries)
```

**Component layout:**

```
app/
├── main.py              FastAPI endpoints (HRMS CRUD + ticket workflows)
├── database.py          SQLAlchemy session + engine
├── models.py            Employee, LeaveRequest, LeaveBalance, Attendance,
│                        HRTicket, ChatMessage
├── seed.py              Deterministic seed — 8 employees + 4 managers,
│                        10 leave requests, consistent balances
├── schemas.py           Pydantic schemas for the REST layer
│
├── services/            Single enforcement point for every domain operation
│   ├── leave.py         Atomic status/balance transitions, role gates
│   ├── tickets.py       Ticket lifecycle, department-based routing
│   └── chats.py         Persistent chat history + conversation memory
│
└── agent/
    ├── graph.py         LangGraph state graph (11 nodes)
    ├── nodes.py         Supervisor, retrieve, grade, generate, tool_call,
    │                    human_approval, tool_execute, tool_answer, reroute
    ├── tools.py         11 tools (6 read, 5 write) with role checks
    ├── state.py         Typed AgentState
    ├── config.py        Specialist→department, specialist→tool subset,
    │                    specialist→ticket category maps
    ├── vector_store.py  Chroma + nomic-embed-text, department metadata
    ├── llm.py           Ollama chat + JSON-mode wrappers
    └── tracing.py       Langfuse callbacks (graceful if not configured)

data/policies/           HR, IT, Security policy docs (markdown)
evals/                   16-case eval harness, CSV + JSON reports
tests/                   10 structure + consistency tests (CI-wired)
ui.py                    Streamlit chat UI with "How the agent got here" panel
```

---

## Key features

### Multi-agent supervisor (3 specialists)

A supervisor node picks one of three specialists for every question in a
single LLM call, along with the intent (policy vs personal_data).
Specialists share graph nodes but differ meaningfully in three places:
system prompt, tool subset bound at the tool_call node, and department
filter applied at retrieval. Specialists can grade a retrieved chunk as
`out_of_scope` and reroute back to the supervisor — bounded to 2 retries
to prevent infinite loops.

### Human-in-the-loop writes

Every state-changing tool (`submit_leave_request`,
`approve_leave_request`, `reject_leave_request`, `create_hr_ticket`,
`update_ticket_status`) runs through LangGraph's `interrupt()`. The UI
shows the pending tool call with its arguments and waits for the user
to Approve or Reject before the write happens. All writes resumable
from the Streamlit sidebar via a stable thread_id.

### Role-gated authorization

Two separate authorization patterns, both enforced inside the tool:

- **Direct-manager gate** (leave approvals) — the approver must be the
  requester's direct manager (`Employee.manager_code`).
- **Department-manager gate** (ticket resolution) — the manager's
  department must cover the ticket's category
  (e.g., IT dept covers IT + Security tickets).

If the gate fails, the tool returns
`{"error": "authorization_failed"}` and no state changes. The LLM
relays the refusal to the user.

### Atomic domain operations

The leave service enforces this invariant:

```
leave_balance.used_days ==
    sum(request.days for request in leave_requests
        if request.status == "approved")
```

Every transition (create / approve / reject / cancel) runs status
change + balance update in a single SQLAlchemy transaction. A
dedicated `consistency_test.py` walks the database and asserts the
invariant — wired into CI, so any future code path that breaks it
fails the build.

### Grounded RAG with document grading

The policy path retrieves from a Chroma store (filtered by the
specialist's department), then grades each result as
`good / weak / none / out_of_scope`. Only `good` chunks feed the
answer generator; `weak` triggers a query rewrite (bounded retries);
`out_of_scope` reroutes to the supervisor; `none` and exhausted
retries escalate.

Answers cite the source files (e.g., `leave_policy.md`,
`security_policy.md`) they came from.

### Persistent chat + conversation memory

Chat history is persisted per-employee in `chat_messages`. Streamlit
restarts don't lose the conversation. Each user has an isolated chat
thread; switching employees in the sidebar shows that user's history
only. The supervisor and tool_call nodes see a 6-turn memory window
so follow-ups like "yes", "approve that one", or "make it 4 days
instead" resolve correctly.

### Local observability

Langfuse traces every graph run — supervisor decisions, retrieval,
grading, tool invocations, LLM latency per node. The whole trace is
visible as a timeline in the Langfuse UI with full input/output
inspection. Configured via `.env` and gracefully disabled when the
keys aren't set.

### Reproducible eval

A 16-case eval harness covers four categories: policy, personal_data
read, personal_data write, and out-of-scope / escalation. Scored on
overall accuracy, intent classification accuracy, escalation
precision, and avg latency. Runs in CI and writes a CSV + JSON report
to `evals/results/`. Current scores: **TODO%** overall, **TODO%**
intent, **TODO%** escalation precision.

---

## Quick start

### Prerequisites

- Docker Desktop (for the easiest path) OR Python 3.12 + Ollama
- About 8 GB free RAM
- One of these models in Ollama: `qwen2.5:7b` (recommended) or
  `llama3.2:3b` (faster, slightly less accurate)

### Run with Docker Compose (recommended)

```bash
git clone https://github.com/neellakdawala/hr-onboarding.git
cd hr-onboarding

# Pull models into the ollama container
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull nomic-embed-text

# Build and start everything
docker compose up -d --build
python -m app.seed    # seed the HRMS database

# Start the chat UI (runs on your Mac, not in Docker)
streamlit run ui.py
```

Open http://localhost:8501, pick an employee from the sidebar, and ask
a question.

### Run locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pull models in your host Ollama install
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# Seed the DB and start the API
python -m app.seed
uvicorn app.main:app --reload   # API at http://localhost:8000

# In another terminal
streamlit run ui.py             # UI at http://localhost:8501
```

### Try these questions

Sign in as **EMP-005 (Ethan)**:

- _How many days of annual leave do I have left?_ — HR specialist, tool-call path
- _What are the password requirements?_ — Security specialist, retrieval path
- _Please open an IT ticket — my VPN keeps disconnecting._ — IT specialist, write with HITL

Then switch to **EMP-102 (Nina, IT Manager)**:

- _What tickets need my attention?_ — see Ethan's VPN ticket
- _Resolve ticket 1 with note 'reset VPN client, stable now.'_ — update + HITL

Then as **EMP-101 (Ravi, Engineering Manager)**:

- _Resolve ticket 1._ — refused, Engineering doesn't cover IT

Expand "🧠 How the agent got here" below each answer to see the
specialist picked, the path through the graph, tool calls with
arguments, retrieval grade, and citations.

---

## Running the eval

```bash
docker compose exec hrms python -m evals.run
```

Produces:

- `evals/results.csv` — one row per case
- `evals/summary.json` — overall, intent, escalation scores + avg latency

Takes 5–10 minutes depending on the model.

---

## Running the tests

```bash
# All structure + consistency tests
for t in smoke_test.py consistency_test.py \
         structure_test_stage*.py structure_test_feature*.py; do
    echo "=== $t ==="; python $t
done
```

Every test should print `All ... passed.` or
`Domain invariant holds.`.

Also run in GitHub Actions on every push — see
`.github/workflows/ci.yml`.

---

## Observability

Optional. To see traces in Langfuse:

```bash
# .env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com   # or your region
```

Rebuild (`docker compose up -d --build`). Every graph run now shows up
as a trace with the full supervisor/retrieval/tool-call timeline.

Without the keys, the agent runs normally and the tracing callback is
a no-op.

---

## Design decisions worth defending

### Why single-specialist routing (not fan-out-merge)

The supervisor picks exactly one specialist per question. Fan-out +
merge is a textbook multi-agent pattern but at a 7B model scale it's
unreliable — small models produce incoherent syntheses when asked to
merge multiple answers. Single-specialist routing was the honest
design for the model actually in use. Would revisit on a frontier
model.

### Why shared nodes across specialists (not full sub-graphs)

Each specialist is distinct in system prompt, tool subset, and
retrieval filter — but the retrieval/grading/HITL logic is identical
across them. Duplicating it into three sub-graphs would have tripled
the LOC without changing behavior. Readability wins.

### Why role gates in the tool, not the prompt

Prompt-level instructions ("don't approve requests from other teams")
rely on the LLM to comply. Tool-level checks run regardless of what
the LLM asks for. For writes that touch real state, the stronger
guarantee is the right one.

### Why atomic transitions in a service module

Status and balance have to change together or the invariant breaks.
Spreading the logic across endpoints or tools would make it easy to
accidentally skip the balance update somewhere. One service module
means one place to check and one place to fix.

### Why local LLM over a hosted API

Three reasons: (1) zero per-call cost makes iteration fast, (2) data
never leaves the machine — relevant for a project touching HR records
even if synthetic, (3) demonstrates MLOps setup (Docker, container
orchestration, model lifecycle) rather than just API calls.

---

## Current limitations and future work

**Deployed only locally.** No live URL. The project is designed to run
on a laptop or a single VM; a production version would move the
database to Postgres, the vector store to a managed Chroma or
pgvector, and the LLM to a hosted inference endpoint.

**Single-turn eval.** The eval harness tests each question standalone
and doesn't exercise the multi-turn memory. A conversational eval
suite would catch regressions in follow-up handling.

**Keyword-based eval scoring with 1 LLM-as-judge sanity check.** Fine
for a demo, hits a ceiling on anything resembling freeform answers.
Full LLM-as-judge would scale better.

**No per-tool observability sampling.** Langfuse captures everything;
would add trace-level sampling for a higher-throughput deployment.

**No streaming responses.** The UI blocks on the full answer. Token
streaming would improve the chat feel on slower hardware.

---

## Tech stack

| Layer           | Choice                        | Why                                                            |
| --------------- | ----------------------------- | -------------------------------------------------------------- |
| Agent framework | LangGraph                     | Explicit graph is easier to reason about than LangChain chains |
| LLM             | Ollama / Qwen 2.5 7B          | Local, free, best-in-class tool calling at 7B                  |
| Embeddings      | nomic-embed-text              | Local, strong for its size                                     |
| Vector store    | Chroma                        | Simple persistent local store, supports metadata filtering     |
| HRMS            | FastAPI + SQLAlchemy + SQLite | Minimal footprint, same DB file shared with the agent          |
| UI              | Streamlit                     | Fast to iterate, built-in HITL via buttons                     |
| Observability   | Langfuse                      | Open-source, hosted free tier, clean trace view                |
| Container       | Docker Compose                | Reproducible local setup                                       |
| CI              | GitHub Actions                | Runs all 10 tests on push                                      |

---

## Repository layout

```
.
├── README.md                       (you are here)
├── DEPLOYMENT.md                   container + env notes
├── Dockerfile                      hrms service image
├── docker-compose.yml              hrms + ollama, bind-mounted DB
├── requirements.txt                Python deps
├── .env.example                    sample env for Langfuse keys
├── app/                            see "Component layout" above
├── data/policies/                  HR / IT / Security markdown docs
├── evals/                          16-case eval harness + runner
├── ui.py                           Streamlit chat UI
├── smoke_test.py                   HRMS API end-to-end checks
├── consistency_test.py             domain invariant check
├── structure_test_stage*.py        agent graph structure checks
├── structure_test_feature*.py      per-feature checks
└── .github/workflows/ci.yml        runs every test on push
```

---

## License

MIT — see LICENSE.
