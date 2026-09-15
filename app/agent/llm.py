import os

from langchain_ollama import ChatOllama

from app.agent import config


def _ollama_kwargs() -> dict:
    """Common kwargs for ChatOllama, honouring OLLAMA_HOST."""
    kwargs = {
        "model": config.LLM_MODEL,
        "temperature": config.LLM_TEMPERATURE,
    }
    host = os.getenv("OLLAMA_HOST")
    if host:
        kwargs["base_url"] = host
    return kwargs


def get_llm() -> ChatOllama:
    """Return the configured chat model for free-text responses."""
    return ChatOllama(**_ollama_kwargs())


def get_json_llm() -> ChatOllama:
    """Return the chat model in JSON mode: output is guaranteed valid JSON."""
    return ChatOllama(format="json", **_ollama_kwargs())