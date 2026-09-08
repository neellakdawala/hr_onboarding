"""
Central configuration for the agent.

All model names, paths, and tunable knobs live in ONE place so switching
models (e.g. llama3.2:3b -> llama3.1:8b -> qwen2.5:7b) is a one-line
change, not a hunt across the codebase.
"""
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
