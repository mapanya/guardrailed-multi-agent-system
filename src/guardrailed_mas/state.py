"""The shared record (graph state) that LangGraph passes from step to step.

Think of it as the job folder that moves between desks. Every worker reads
from it and writes its result back into it. The execution_count field is the
counter that the guardrail checks.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # Inputs
    task: str
    user_id: str
    profile: str

    # Short facts recalled from memory (never the full past conversation)
    memory_context: list[str]

    # Worker results, stored as plain dictionaries after validation
    plan: Optional[dict[str, Any]]
    draft: Optional[dict[str, Any]]
    review: Optional[dict[str, Any]]
    feedback: list[str]

    # Guardrail fields
    execution_count: int          # attempt counter checked by the guardrail
    attempts_completed: int       # how many times the Executor actually ran
    max_executions: int           # the cap (3)
    halt_reason: Optional[str]    # why a guardrail stopped the run, if it did

    # Final answer (FinalResult or FallbackResponse as a dictionary)
    status: str
    final: Optional[dict[str, Any]]


def initial_state(task: str, user_id: str, profile: str, max_executions: int) -> GraphState:
    """Build a clean state for a new job. The counter always starts at 0."""
    return GraphState(
        task=task,
        user_id=user_id,
        profile=profile,
        memory_context=[],
        plan=None,
        draft=None,
        review=None,
        feedback=[],
        execution_count=0,
        attempts_completed=0,
        max_executions=max_executions,
        halt_reason=None,
        status="running",
        final=None,
    )
