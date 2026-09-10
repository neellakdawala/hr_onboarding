from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:                                              # noqa: BLE001
    pass


def _langfuse_configured() -> bool:
    """Return True iff both Langfuse keys are present."""
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


@lru_cache(maxsize=1)
def _make_handler() -> Any | None:
    """
    Build the Langfuse callback handler once and cache it.

    Returns None if Langfuse is not configured or the package is not
    installed - both are non-fatal, the agent just runs untraced.
    """
    if not _langfuse_configured():
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:                                          # noqa: BLE001
        return None
    return CallbackHandler()


def get_callbacks() -> list[Any]:
    """
    Return the list of LangChain callbacks to attach to a run.

    Nodes and the top-level `graph.invoke(...)` call pass this into
    `config={"callbacks": get_callbacks(), ...}`. Empty list means
    "no tracing", which is a valid state.
    """
    handler = _make_handler()
    return [handler] if handler is not None else []


def is_tracing_enabled() -> bool:
    """Convenience check for user-facing scripts."""
    return _make_handler() is not None