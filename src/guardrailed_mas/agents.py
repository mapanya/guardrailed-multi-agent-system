"""The three workers (Planner, Executor, Reviewer), built with CrewAI,
plus the connection to the Groq model service they use.

The graph only needs an object with three methods: plan, execute and review.
CrewAIWorkers is the real implementation. Tests use simple stand-in workers
with the same three methods, so the graph and guardrails can be tested
without any model or key.

Every reply goes through validation.validate_output before it is returned,
so the graph only ever receives objects that match the schemas.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Any, Optional, Protocol

from pydantic import BaseModel

from .config import ModelSettings
from .schemas import ExecutorOutput, PlannerOutput, ReviewerOutput
from .validation import Checker, ValidatedOutput, validate_output

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Connection to the language model service (Groq)
#
# To keep the dependency list short, this project talks to Groq through its 
# OpenAI-compatible address using theofficial OpenAI and Groq client libraries. 
# --------------------------------------------------------------------------
class MissingKeyError(RuntimeError):
    """Raised when GROQ_API_KEY has not been set."""


def groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise MissingKeyError(
            "GROQ_API_KEY is not set. In Colab add it under Secrets (the key icon). "
            "On your Laptop/Desktop put it in a .env file. Never paste it into the code."
        )
    return key


def crewai_llm(models: ModelSettings) -> Any:
    """Model used by the three CrewAI workers."""
    from crewai import LLM

    return LLM(
        model=models.worker_model,
        provider="openai",          # keeps the full model name, for example "openai/gpt-oss-120b"
        base_url=models.base_url,   # Groq's OpenAI-compatible address
        api_key=groq_api_key(),
        temperature=models.temperature,
        reasoning_effort=models.reasoning_effort,
        max_retries=models.max_retries,
    )


def pydantic_ai_model(models: ModelSettings) -> Any:
    """Model used by the PydanticAI output checker."""
    from groq import AsyncGroq
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    # The Groq library adds "/openai/v1" itself, so give it the root address.
    root = models.base_url.split("/openai/")[0]
    client = AsyncGroq(api_key=groq_api_key(), base_url=root, max_retries=models.max_retries)
    return GroqModel(models.helper_model, provider=GroqProvider(groq_client=client))


class Workers(Protocol):
    """What the graph expects from any set of workers."""

    def plan(self, task: str, memory_context: list[str]) -> ValidatedOutput[PlannerOutput]: ...

    def execute(
        self, task: str, plan: PlannerOutput, feedback: list[str], memory_context: list[str]
    ) -> ValidatedOutput[ExecutorOutput]: ...

    def review(self, task: str, plan: PlannerOutput, draft: ExecutorOutput) -> ValidatedOutput[ReviewerOutput]: ...


# --------------------------------------------------------------------------
# Prompt helpers
# --------------------------------------------------------------------------
def _schema_block(schema: type[BaseModel]) -> str:
    return (
        "Reply with ONE JSON object only, with no text before or after it. "
        "It must match this JSON schema exactly (no extra fields):\n"
        + json.dumps(schema.model_json_schema(), indent=1)
    )


def _facts_block(memory_context: list[str]) -> str:
    if not memory_context:
        return "Known facts about this organisation: none recorded yet."
    lines = "\n".join(f"- {fact}" for fact in memory_context)
    return f"Known facts about this organisation (use them where relevant):\n{lines}"


def planner_prompt(task: str, memory_context: list[str], instructions: str) -> str:
    return "\n\n".join(
        [f"Job: {task}", _facts_block(memory_context), f"Instructions: {instructions}", _schema_block(PlannerOutput)]
    )


def executor_prompt(
    task: str, plan: PlannerOutput, feedback: list[str], memory_context: list[str], instructions: str
) -> str:
    parts = [
        f"Job: {task}",
        _facts_block(memory_context),
        "Approved plan:\n" + plan.model_dump_json(indent=1),
        f"Instructions: {instructions}",
    ]
    if feedback:
        parts.append("The reviewer rejected the previous attempt. Fix ALL of these:\n" + "\n".join(f"- {f}" for f in feedback))
    parts.append(_schema_block(ExecutorOutput))
    return "\n\n".join(parts)


def reviewer_prompt(task: str, plan: PlannerOutput, draft: ExecutorOutput, instructions: str) -> str:
    return "\n\n".join(
        [
            f"Job: {task}",
            "Plan:\n" + plan.model_dump_json(indent=1),
            "Work to review:\n" + draft.model_dump_json(indent=1),
            f"Review rules: {instructions}",
            "If you reject, required_changes must list each exact change.",
            _schema_block(ReviewerOutput),
        ]
    )


# --------------------------------------------------------------------------
# Real workers
# --------------------------------------------------------------------------
class CrewAIWorkers:
    """Planner, Executor and Reviewer as CrewAI agents, one small crew per call."""

    def __init__(self, llm: Any, profile: dict[str, Any], checker: Optional[Checker] = None) -> None:
        from crewai import Agent  # imported here so the tests do not need CrewAI

        self._llm = llm
        self._profile = profile
        self._checker = checker
        self._agents = {
            name: Agent(
                role=profile[name]["role"],
                goal=profile[name]["goal"],
                backstory=profile[name]["backstory"],
                llm=llm,
                allow_delegation=False,   # workers cannot hand work to each other on their own
                max_iter=1,               # one reasoning pass per call; the graph controls retries
                verbose=False,
            )
            for name in ("planner", "executor", "reviewer")
        }

    def _run(self, name: str, description: str, expected: str) -> str:
        from crewai import Crew, Process, Task

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # hide library housekeeping notices
            task = Task(description=description, expected_output=expected, agent=self._agents[name])
            crew = Crew(agents=[self._agents[name]], tasks=[task], process=Process.sequential, verbose=False)
            result = crew.kickoff()
        return str(result.raw)

    def plan(self, task: str, memory_context: list[str]) -> ValidatedOutput[PlannerOutput]:
        prompt = planner_prompt(task, memory_context, self._profile["planner"]["instructions"])
        raw = self._run("planner", prompt, "A JSON object matching the PlannerOutput schema.")
        return validate_output(raw, PlannerOutput, self._checker)

    def execute(
        self, task: str, plan: PlannerOutput, feedback: list[str], memory_context: list[str]
    ) -> ValidatedOutput[ExecutorOutput]:
        prompt = executor_prompt(task, plan, feedback, memory_context, self._profile["executor"]["instructions"])
        raw = self._run("executor", prompt, "A JSON object matching the ExecutorOutput schema.")
        return validate_output(raw, ExecutorOutput, self._checker)

    def review(self, task: str, plan: PlannerOutput, draft: ExecutorOutput) -> ValidatedOutput[ReviewerOutput]:
        prompt = reviewer_prompt(task, plan, draft, self._profile["reviewer"]["instructions"])
        raw = self._run("reviewer", prompt, "A JSON object matching the ReviewerOutput schema.")
        return validate_output(raw, ReviewerOutput, self._checker)


class AlwaysRejectReviewer:
    """Wraps real workers but replaces the Reviewer with one that always says no.

    Used by --demo-guardrail to prove the cap and fallback work with real
    Planner and Executor calls, without spending tokens on reviews.
    """

    def __init__(self, inner: Workers) -> None:
        self._inner = inner

    def plan(self, task: str, memory_context: list[str]) -> ValidatedOutput[PlannerOutput]:
        return self._inner.plan(task, memory_context)

    def execute(self, task, plan, feedback, memory_context):  # type: ignore[no-untyped-def]
        return self._inner.execute(task, plan, feedback, memory_context)

    def review(self, task: str, plan: PlannerOutput, draft: ExecutorOutput) -> ValidatedOutput[ReviewerOutput]:
        rejection = ReviewerOutput(
            approved=False,
            score=0,
            issues=["Guardrail demonstration: this reviewer rejects every attempt."],
            required_changes=["Demonstration only: no change can satisfy this reviewer."],
        )
        return ValidatedOutput(rejection, "direct")
