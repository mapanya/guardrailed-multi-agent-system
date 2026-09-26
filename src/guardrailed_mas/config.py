"""Load settings, worker profiles and sample scenarios from the config folder.

All tunable values live in YAML files so that behaviour can change without
editing code. A few values can also be overridden with environment variables,
which is handy in Google Colab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The project root is two folders above this file: src/guardrailed_mas/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _read_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GuardrailSettings:
    max_executions: int = 3
    kill_switch: bool = False
    recursion_limit: int = 25


@dataclass(frozen=True)
class ModelSettings:
    base_url: str = "https://api.groq.com/openai/v1"
    worker_model: str = "openai/gpt-oss-120b"
    helper_model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    reasoning_effort: str = "low"
    max_retries: int = 6


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool = True
    data_dir: Path = PROJECT_ROOT / "data"
    sqlite_file: str = "mem0_history.db"
    vector_folder: str = "chroma"
    collection_name: str = "guardrailed_mas_memory"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_recalled_facts: int = 5
    infer_facts: bool = False

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_file

    @property
    def vector_path(self) -> Path:
        return self.data_dir / self.vector_folder


@dataclass(frozen=True)
class Settings:
    guardrails: GuardrailSettings = field(default_factory=GuardrailSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    checker_retries: int = 2
    log_folder: Path = PROJECT_ROOT / "logs"
    log_level: str = "INFO"


def _resolve_folder(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> Settings:
    """Read config/settings.yaml and apply any environment overrides."""
    raw = _read_yaml("settings.yaml")
    g = raw.get("guardrails", {})
    m = raw.get("models", {})
    mem = raw.get("memory", {})
    log = raw.get("logging", {})

    guardrails = GuardrailSettings(
        max_executions=int(g.get("max_executions", 3)),
        kill_switch=_env_flag("MAS_KILL_SWITCH", bool(g.get("kill_switch", False))),
        recursion_limit=int(g.get("recursion_limit", 25)),
    )
    models = ModelSettings(
        base_url=os.getenv("MAS_BASE_URL", m.get("base_url", ModelSettings.base_url)),
        worker_model=os.getenv("MAS_WORKER_MODEL", m.get("worker_model", ModelSettings.worker_model)),
        helper_model=os.getenv("MAS_HELPER_MODEL", m.get("helper_model", ModelSettings.helper_model)),
        temperature=float(m.get("temperature", 0)),
        reasoning_effort=str(m.get("reasoning_effort", "low")),
        max_retries=int(m.get("max_retries", 6)),
    )
    memory = MemorySettings(
        enabled=_env_flag("MAS_MEMORY_ENABLED", bool(mem.get("enabled", True))),
        data_dir=_resolve_folder(os.getenv("MAS_DATA_DIR", mem.get("data_dir", "data"))),
        sqlite_file=mem.get("sqlite_file", "mem0_history.db"),
        vector_folder=mem.get("vector_folder", "chroma"),
        collection_name=mem.get("collection_name", "guardrailed_mas_memory"),
        embedder_model=os.getenv(
            "MAS_EMBEDDER_MODEL", mem.get("embedder_model", MemorySettings.embedder_model)
        ),
        max_recalled_facts=int(mem.get("max_recalled_facts", 5)),
        infer_facts=bool(mem.get("infer_facts", False)),
    )
    return Settings(
        guardrails=guardrails,
        models=models,
        memory=memory,
        checker_retries=int(raw.get("validation", {}).get("checker_retries", 2)),
        log_folder=_resolve_folder(log.get("folder", "logs")),
        log_level=str(log.get("level", "INFO")).upper(),
    )


def load_profile(name: str | None = None) -> dict[str, Any]:
    """Return one worker profile (planner, executor and reviewer wording)."""
    raw = _read_yaml("agents.yaml")
    profile_name = name or raw.get("active_profile")
    profiles = raw.get("profiles", {})
    if profile_name not in profiles:
        known = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown profile '{profile_name}'. Known profiles: {known}")
    profile = dict(profiles[profile_name])
    profile["name"] = profile_name
    return profile


def load_scenarios() -> dict[str, dict[str, str]]:
    """Return the sample jobs listed in config/scenarios.yaml."""
    return _read_yaml("scenarios.yaml").get("scenarios", {})
