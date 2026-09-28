from pathlib import Path

# ---- LLM ----
LLM_MODEL = "llama3.2:3b"      # user's choice; swap here to change everywhere
LLM_TEMPERATURE = 0.0          # deterministic for grading + generation

# ---- Embeddings ----
EMBED_MODEL = "nomic-embed-text"

# ---- Vector store ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "chroma"

# ---- Retrieval knobs ----
RETRIEVAL_K = 4                # how many chunks to fetch per query
CHUNK_SIZE = 500               # characters per chunk
CHUNK_OVERLAP = 80             # chunk overlap for context continuity

# ---- Agent behaviour ----
MAX_RETRIES = 2                # max query rewrites before giving up

# ---- Department routing ----
# Used by the escalate node to map a question topic to the right team.
KNOWN_DEPARTMENTS = ["HR", "IT", "Security", "Finance"]
DEFAULT_ESCALATION = "HR"      # if we truly cannot tell, fall back to HR


# ---- Multi-agent supervisor (Feature 4) ----
KNOWN_SPECIALISTS = ["hr", "it", "security"]
DEFAULT_SPECIALIST = "hr"
MAX_SUPERVISOR_REROUTES = 2   # after this many mis-routes, escalate

# Which department's docs each specialist reads from during retrieval.
SPECIALIST_TO_DEPARTMENT = {
    "hr": "HR",
    "it": "IT",
    "security": "Security",
}

# Which agent tool names each specialist can see.
# In interviews, worth naming: HR ends up with most tools because our
# demo domain is HR-heavy. In a real company IT and Security would have
# their own action tools (VPN provisioning, access grants, etc).
SPECIALIST_TO_TOOLS = {
    "hr": {
        "get_leave_balance", "get_leave_requests", "get_attendance",
        "list_pending_approvals", "list_my_tickets",
        "submit_leave_request", "approve_leave_request",
        "reject_leave_request", "create_hr_ticket",
    },
    "it": {"list_my_tickets", "create_hr_ticket"},
    "security": {"list_my_tickets", "create_hr_ticket"},
}

# The category to force onto a ticket when a specific specialist files one.
SPECIALIST_TICKET_CATEGORY = {
    "hr": "HR", "it": "IT", "security": "Security",
}