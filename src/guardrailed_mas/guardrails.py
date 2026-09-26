"""Guardrails: the attempt counter, the conditional edges and the fallback.

These functions contain no model calls at all. They are plain, predictable
Python, which is exactly what a safety control should be. That also means
they can be tested on any laptop/desktop without a key.

The rule (from the Alibaba ROME case study):
    execution_count > MAX_EXECUTIONS  ->  stop immediately, return the fallback.
"""

from __future__ import annotations

import os
from typing import Literal

from .schemas import FallbackResponse
from .state import GraphState

MAX_EXECUTIONS = 3  # default cap; the live value comes from config/settings.yaml


def kill_switch_active(config_value: bool = False) -> bool:
    """True when the emergency stop is on in settings or in the environment."""
    env = os.getenv("MAS_KILL_SWITCH", "").strip().lower()
    return config_value or env in {"1", "true", "yes", "on"}


def budget_exceeded(execution_count: int, max_executions: int = MAX_EXECUTIONS) -> bool:
    """The exact rule from the brief: more than the cap means stop."""
    return execution_count > max_executions


# --------------------------------------------------------------------------
# Conditional edges. LangGraph calls these to decide where to go next.
# --------------------------------------------------------------------------
def route_after_planner(state: GraphState) -> Literal["budget_gate", "fallback"]:
    """Go on to the attempt gate, unless planning failed or the stop is on."""
    if state.get("halt_reason"):
        return "fallback"
    return "budget_gate"


def check_budget(state: GraphState) -> Literal["executor", "fallback"]:
    """The conditional edge that enforces the cap before every Executor run."""
    if state.get("halt_reason"):
        return "fallback"
    if budget_exceeded(state["execution_count"], state["max_executions"]):
        return "fallback"
    return "executor"


def route_after_executor(state: GraphState) -> Literal["reviewer", "budget_gate", "fallback"]:
    """Valid work goes to the Reviewer. Invalid work uses up an attempt."""
    if state.get("halt_reason"):
        return "fallback"
    if state.get("draft") is None:
        return "budget_gate"
    return "reviewer"


def route_after_review(state: GraphState) -> Literal["finalize", "budget_gate", "fallback"]:
    """Approved work is finished. Rejected work goes back through the gate."""
    if state.get("halt_reason"):
        return "fallback"
    review = state.get("review") or {}
    if review.get("approved"):
        return "finalize"
    return "budget_gate"


# --------------------------------------------------------------------------
# Fallback answer
# --------------------------------------------------------------------------
def build_fallback(state: GraphState) -> FallbackResponse:
    """Create the fixed, safe answer. The wording never comes from a model."""
    reason = state.get("halt_reason") or "execution_budget_exceeded"
    review = state.get("review") or {}
    attempts = state.get("attempts_completed", 0)
    return FallbackResponse(
        reason=reason,
        task=state.get("task", ""),
        attempts_used=attempts,
        last_issues=list(review.get("required_changes") or review.get("issues") or [])[:5],
    )
