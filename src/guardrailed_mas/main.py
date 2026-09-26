"""Command line entry point.

Examples (run from the project folder):
    python -m guardrailed_mas.main list-scenarios
    python -m guardrailed_mas.main remember --user-id demo_user --fact "Our email platform is Microsoft 365."
    python -m guardrailed_mas.main recall --user-id demo_user
    python -m guardrailed_mas.main run --scenario phishing --user-id demo_user
    python -m guardrailed_mas.main run --task "Describe any incident here" --user-id demo_user
    python -m guardrailed_mas.main run --scenario ransomware --demo-guardrail
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Switch off usage reporting from the libraries before they are imported.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Library housekeeping notices are not useful to the reader of a run.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from .config import PROJECT_ROOT, Settings, load_profile, load_scenarios, load_settings  # noqa: E402
from .state import initial_state  # noqa: E402

logger = logging.getLogger("guardrailed_mas.main")


# --------------------------------------------------------------------------
# Logging: readable lines on screen plus a JSON Lines file per run.
# The file in the logs folder is the execution evidence for the submission.
# Each line is one step, for example:
# {"time": "...", "node": "budget_gate", "execution_count": 4, "exceeded": true}
# --------------------------------------------------------------------------
_STANDARD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
        }
        extras = {k: v for k, v in vars(record).items() if k not in _STANDARD}
        if extras:
            entry.update(extras)
        else:
            entry["message"] = record.getMessage()
        return json.dumps(entry, default=str)


def setup_logging(folder: Path, level: str = "INFO", run_name: str = "run") -> Path:
    """Send logs to the screen and to logs/<run_name>_<timestamp>.jsonl."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = folder / f"{run_name}_{stamp}.jsonl"

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    screen = logging.StreamHandler()
    screen.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    # On screen: our own step lines, plus warnings and errors from anything else.
    screen.addFilter(lambda record: record.name.startswith("guardrailed_mas") or record.levelno >= logging.WARNING)
    root.addHandler(screen)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(JsonLinesFormatter())
    # Only our own project lines go to the evidence file, not library chatter.
    file_handler.addFilter(logging.Filter("guardrailed_mas"))
    root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "openai", "groq", "chromadb", "sentence_transformers", "crewai",
                  "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Mem0 prints advice about optional add-ons we do not need; keep only real errors.
    logging.getLogger("mem0").setLevel(logging.ERROR)
    return log_path


def _load_env_file() -> None:
    """Read GROQ_API_KEY from a local .env file if one exists (never committed)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def build_memory(settings: Settings, enabled: bool = True):
    if not (enabled and settings.memory.enabled):
        return None
    from .memory import Mem0Store

    return Mem0Store(settings.memory, settings.models)


def run_job(
    task: str,
    user_id: str = "demo_user",
    profile_name: Optional[str] = None,
    demo_guardrail: bool = False,
    use_memory: bool = True,
    settings: Optional[Settings] = None,
    workers: Any = None,
    memory: Any = None,
) -> dict[str, Any]:
    """Run one job through the graph and return the final answer as a dictionary."""
    from .agents import AlwaysRejectReviewer, CrewAIWorkers
    from .graph import build_graph
    from .agents import crewai_llm, pydantic_ai_model
    from .validation import PydanticAIChecker

    settings = settings or load_settings()
    profile = load_profile(profile_name)

    if workers is None:
        checker = PydanticAIChecker(pydantic_ai_model(settings.models), retries=settings.checker_retries)
        workers = CrewAIWorkers(crewai_llm(settings.models), profile, checker)
    if demo_guardrail:
        workers = AlwaysRejectReviewer(workers)
    if memory is None and use_memory:
        memory = build_memory(settings)

    app = build_graph(workers, memory, kill_switch=settings.guardrails.kill_switch)
    state = initial_state(task, user_id, profile["name"], settings.guardrails.max_executions)
    logger.info("Starting job for user %s with profile %s (cap=%s)", user_id, profile["name"],
                settings.guardrails.max_executions)
    result_state = app.invoke(state, config={"recursion_limit": settings.guardrails.recursion_limit})
    return result_state["final"]


def _save_result(settings: Settings, final: dict[str, Any]) -> str:
    settings.log_folder.mkdir(parents=True, exist_ok=True)
    path = settings.log_folder / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    return str(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="guardrailed_mas", description="Guardrailed Planner, Executor, Reviewer system")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a job through the three workers")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", help="Name of a sample job in config/scenarios.yaml")
    source.add_argument("--task", help="Describe any job in your own words")
    run.add_argument("--user-id", default="demo_user")
    run.add_argument("--profile", default=None, help="Worker profile from config/agents.yaml")
    run.add_argument("--demo-guardrail", action="store_true", help="Reviewer rejects everything to show the cap")
    run.add_argument("--no-memory", action="store_true")

    remember = sub.add_parser("remember", help="Save a fact to long-term memory")
    remember.add_argument("--user-id", default="demo_user")
    remember.add_argument("--fact", required=True)

    recall = sub.add_parser("recall", help="Show what memory holds for a user")
    recall.add_argument("--user-id", default="demo_user")
    recall.add_argument("--query", default=None)

    sub.add_parser("list-scenarios", help="Show the sample jobs")
    sub.add_parser("draw-graph", help="Print the workflow diagram as Mermaid text")

    args = parser.parse_args(argv)
    _load_env_file()
    settings = load_settings()
    log_path = setup_logging(settings.log_folder, settings.log_level, run_name=args.command.replace("-", "_"))

    if args.command == "list-scenarios":
        for name, item in load_scenarios().items():
            print(f"{name:18} [{item['profile']}] {item['task'][:90]}")
        return 0

    if args.command == "draw-graph":
        from .agents import AlwaysRejectReviewer  # any object with the three methods will do
        from .graph import build_graph

        app = build_graph(AlwaysRejectReviewer(None))  # type: ignore[arg-type]
        print(app.get_graph().draw_mermaid())
        return 0

    if args.command in {"remember", "recall"}:
        from .memory import sqlite_history_count

        memory = build_memory(settings)
        if memory is None:
            print("Memory is switched off in config/settings.yaml.")
            return 1
        if args.command == "remember":
            memory.remember(args.user_id, args.fact, {"kind": "user_fact"})
            print(f"Saved for {args.user_id}: {args.fact}")
        else:
            facts = memory.recall(args.user_id, args.query) if args.query else memory.list_all(args.user_id)
            print(f"Memory for {args.user_id} ({len(facts)} item(s)):")
            for fact in facts:
                print(f"  - {fact}")
        print(f"SQLite history file: {settings.memory.sqlite_path} ({sqlite_history_count(settings.memory)} row(s))")
        return 0

    profile_name = args.profile
    if args.scenario:
        scenarios = load_scenarios()
        if args.scenario not in scenarios:
            print(f"Unknown scenario '{args.scenario}'. Choose from: {', '.join(scenarios)}")
            return 2
        task = scenarios[args.scenario]["task"]
        profile_name = profile_name or scenarios[args.scenario].get("profile")
    else:
        task = args.task

    from .agents import MissingKeyError

    try:
        final = run_job(task, args.user_id, profile_name, args.demo_guardrail, not args.no_memory, settings)
    except MissingKeyError as error:
        print(f"\nSTOPPED: {error}")
        return 3
    print("\n================ FINAL ANSWER ================")
    print(json.dumps(final, indent=2))
    print(f"\nResult saved to: {_save_result(settings, final)}")
    print(f"Step log saved to: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
