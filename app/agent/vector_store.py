"""
Vector store setup: turn the policy markdown files into a searchable
Chroma collection.

Two important details:
  1. Every chunk carries `department` metadata, extracted from the
     "**Department: X**" line in each policy doc. This metadata is what
     lets the escalate node route to the right team, and could later
     be used for metadata-filtered retrieval.
  2. This module builds the store lazily on first use and persists it
     to disk. Rebuilding on every run would be slow; persisting means
     you pay the embedding cost once.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agent import config


DEPT_PATTERN = re.compile(r"\*\*Department:\s*(\w+)\*\*", re.IGNORECASE)


def _extract_department(text: str, default: str = "HR") -> str:
    """Pull the department label out of a policy doc header."""
    match = DEPT_PATTERN.search(text)
    return match.group(1) if match else default


def _load_policy_documents() -> list[Document]:
    """Read every markdown file under POLICY_DIR into a Document."""
    docs: list[Document] = []
    for path in sorted(Path(config.POLICY_DIR).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        department = _extract_department(text)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "department": department,
                },
            )
        )
    return docs


def _chunk_documents(docs: list[Document]) -> list[Document]:
    """Split each doc into overlapping chunks; metadata is preserved."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Build or load the vector store.

    - If the persistence dir does not exist (or force_rebuild=True),
      re-ingest all policy docs and embed them.
    - Otherwise, open the existing persisted store.
    """
    persist_dir = Path(config.VECTOR_STORE_DIR)
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)

    if force_rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)

    if persist_dir.exists() and any(persist_dir.iterdir()):
        # Load existing.
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name="policies",
        )

    # First-time build.
    persist_dir.mkdir(parents=True, exist_ok=True)
    docs = _load_policy_documents()
    if not docs:
        raise RuntimeError(
            f"No policy documents found under {config.POLICY_DIR}."
        )
    chunks = _chunk_documents(docs)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="policies",
        persist_directory=str(persist_dir),
    )
    print(
        f"Vector store built: {len(docs)} documents, "
        f"{len(chunks)} chunks, embedded with '{config.EMBED_MODEL}'."
    )
    return store


def get_retriever():
    """Return a retriever configured with the project's k value."""
    store = build_vector_store()
    return store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
