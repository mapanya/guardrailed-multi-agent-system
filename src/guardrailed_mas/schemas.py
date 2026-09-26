"""Strict output forms (schemas) for every worker.

Each worker must hand back data that fits one of these forms exactly.
The same classes are given to PydanticAI as its required output type, so a
reply is either a valid, typed Python object or it is rejected. Nothing
half-formed is ever passed to the next worker.

Design rules used here:
* extra="forbid": unknown fields are an error, not silently ignored.
* Length and range limits: stops runaway or empty answers.
* Fixed word lists (Literal): priorities and statuses cannot be invented.
* Cross-field checks: a rejection must always say what to change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Priority = Literal["critical", "high", "medium", "low"]


class StrictModel(BaseModel):
    """Base class: reject unknown fields and trim stray spaces."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------
class PlanStep(StrictModel):
    step_number: int = Field(ge=1, le=10, description="Position of the step, starting at 1.")
    action: str = Field(min_length=5, max_length=300, description="What must be done.")
    owner_role: str = Field(min_length=2, max_length=80, description="Team role that owns it.")


class PlannerOutput(StrictModel):
    """The ordered plan produced by the Planner."""

    objective: str = Field(min_length=10, max_length=400)
    steps: list[PlanStep] = Field(min_length=3, max_length=10)
    key_risks: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def steps_are_numbered_in_order(self) -> "PlannerOutput":
        expected = list(range(1, len(self.steps) + 1))
        actual = [step.step_number for step in self.steps]
        if actual != expected:
            raise ValueError(f"Steps must be numbered 1..{len(self.steps)} in order, got {actual}")
        return self


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------
class WorkItem(StrictModel):
    order: int = Field(ge=1, le=20)
    action: str = Field(min_length=5, max_length=300)
    owner_role: str = Field(min_length=2, max_length=80)
    priority: Priority
    time_target: str = Field(min_length=2, max_length=60, description='For example "within 15 minutes".')


class ExecutorOutput(StrictModel):
    """The finished work (for example a first-response checklist)."""

    title: str = Field(min_length=5, max_length=150)
    summary: str = Field(min_length=10, max_length=600)
    items: list[WorkItem] = Field(min_length=3, max_length=15)
    escalation_contacts: list[str] = Field(min_length=1, max_length=6)


# --------------------------------------------------------------------------
# Reviewer
# --------------------------------------------------------------------------
class ReviewerOutput(StrictModel):
    """The Reviewer's decision on the Executor's work."""

    approved: bool
    score: int = Field(ge=0, le=10, description="Overall quality from 0 to 10.")
    issues: list[str] = Field(default_factory=list, max_length=10)
    required_changes: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def rejection_needs_reasons(self) -> "ReviewerOutput":
        if not self.approved and not self.required_changes:
            raise ValueError("A rejection must list at least one required change.")
        return self


# --------------------------------------------------------------------------
# Final answers handed back to the person who asked
# --------------------------------------------------------------------------
class FinalResult(StrictModel):
    """Returned when the Reviewer approves within the attempt budget."""

    status: Literal["approved"] = "approved"
    task: str
    attempts_used: int = Field(ge=1)
    plan: PlannerOutput
    output: ExecutorOutput
    review: ReviewerOutput


HaltReason = Literal["execution_budget_exceeded", "kill_switch", "planner_failed"]

SAFE_FALLBACK_MESSAGE = (
    "The automated workflow was stopped by a safety control before it could "
    "produce an approved answer. No further automated actions were taken. "
    "Please hand this job to a human analyst using your standard escalation procedure."
)


class FallbackResponse(StrictModel):
    """The fixed, safe answer returned whenever a guardrail stops the run."""

    status: Literal["halted"] = "halted"
    reason: HaltReason
    message: str = SAFE_FALLBACK_MESSAGE
    task: str
    attempts_used: int = Field(ge=0)
    last_issues: list[str] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def message_is_fixed(cls, value: str) -> str:
        # The fallback text is defined here, never written by a model.
        if value != SAFE_FALLBACK_MESSAGE:
            raise ValueError("The fallback message cannot be changed at run time.")
        return value
