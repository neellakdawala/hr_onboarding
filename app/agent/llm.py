"""
LLM factory.

One function that returns the chat model. Every node calls this instead
of instantiating ChatOllama directly, so switching provider later
(hosted API, different local model) is a change in ONE place.

Two flavours:
  - get_llm()      : normal free-text responses (used by generate + rewrite)
  - get_json_llm() : forces valid JSON output via Ollama's format="json"
                     (used by grade + escalate, which need structured output)

The JSON-mode variant is what saves us from small-model output being
"almost JSON but with a stray sentence at the front". Ollama enforces
the format at the runtime level, not just via prompting.
"""
from langchain_ollama import ChatOllama

from app.agent import config


def get_llm() -> ChatOllama:
    """Return the configured chat model for free-text responses."""
    return ChatOllama(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )


def get_json_llm() -> ChatOllama:
    """Return the chat model in JSON mode: output is guaranteed valid JSON."""
    return ChatOllama(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        format="json",
    )
