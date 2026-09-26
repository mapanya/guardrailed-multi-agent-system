"""Long-term memory with Mem0, stored locally.

Storage (all inside the data folder, so it survives between sessions):
* SQLite file (mem0_history.db): Mem0's history of every memory added,
  changed or deleted, per user.
* Chroma folder: a local search index so related facts can be found by
  meaning, not just by exact words.

Token control: only a handful of short facts (max_recalled_facts) are handed
to the workers, never the full past conversation. With infer_facts set to
false, saving a fact needs no model call at all.

Memory is best effort: if it fails, the run carries on without it and the
problem is written to the log.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Optional, Protocol

from .config import MemorySettings, ModelSettings

logger = logging.getLogger(__name__)


class MemoryStore(Protocol):
    def recall(self, user_id: str, query: str) -> list[str]: ...

    def remember(self, user_id: str, fact: str, metadata: Optional[dict[str, Any]] = None) -> None: ...

    def list_all(self, user_id: str) -> list[str]: ...


def build_mem0_config(memory: MemorySettings, models: ModelSettings) -> dict[str, Any]:
    """The Mem0 settings: Groq for the language model, local files for storage."""
    return {
        "llm": {
            "provider": "groq",
            "config": {"model": models.helper_model, "temperature": 0, "max_tokens": 1000},
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": memory.embedder_model},
        },
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": memory.collection_name, "path": str(memory.vector_path)},
        },
        "history_db_path": str(memory.sqlite_path),
    }


class Mem0Store:
    """Mem0 memory kept on local disk (SQLite history plus Chroma index)."""

    def __init__(self, memory: MemorySettings, models: ModelSettings) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "False")   # do not send usage data out
        from mem0 import Memory

        memory.data_dir.mkdir(parents=True, exist_ok=True)
        self._settings = memory
        self._memory = Memory.from_config(build_mem0_config(memory, models))
        logger.info("Memory ready. SQLite history file: %s", memory.sqlite_path)

    @staticmethod
    def _texts(result: Any) -> list[str]:
        items = result.get("results", []) if isinstance(result, dict) else (result or [])
        return [item["memory"] for item in items if isinstance(item, dict) and item.get("memory")]

    def recall(self, user_id: str, query: str) -> list[str]:
        result = self._memory.search(query=query, filters={"user_id": user_id}, top_k=self._settings.max_recalled_facts)
        return self._texts(result)[: self._settings.max_recalled_facts]

    def remember(self, user_id: str, fact: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self._memory.add(
            [{"role": "user", "content": fact}],
            user_id=user_id,
            metadata=metadata or {},
            infer=self._settings.infer_facts,
        )

    def list_all(self, user_id: str) -> list[str]:
        return self._texts(self._memory.get_all(filters={"user_id": user_id}))


def sqlite_history_count(memory: MemorySettings) -> int:
    """How many rows the SQLite history file holds. Used as persistence evidence."""
    if not memory.sqlite_path.exists():
        return 0
    with sqlite3.connect(memory.sqlite_path) as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if "history" not in tables:
            return 0
        return int(connection.execute("SELECT COUNT(*) FROM history").fetchone()[0])


class InMemoryStore:
    """Simple stand-in used by the tests. Not persistent."""

    def __init__(self) -> None:
        self.facts: dict[str, list[str]] = {}

    def recall(self, user_id: str, query: str) -> list[str]:
        return list(self.facts.get(user_id, []))[:5]

    def remember(self, user_id: str, fact: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self.facts.setdefault(user_id, []).append(fact)

    def list_all(self, user_id: str) -> list[str]:
        return list(self.facts.get(user_id, []))
