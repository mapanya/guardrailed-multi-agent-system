"""Automated tests for the guardrailed multi-agent system.

All tests use stand-in workers instead of a real model, so they run in a few 
seconds on any workstation, with no key and no internet connection.

Sections:
  1. Stand-in workers and shared test data
  2. Output forms (schemas) and validation
  3. Guardrail rules and conditional edges
  4. The whole LangGraph workflow
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from guardrailed_mas import guardrails
from guardrailed_mas.graph import build_graph
from guardrailed_mas.memory import InMemoryStore
from guardrailed_mas.schemas import (
    SAFE_FALLBACK_MESSAGE,
    ExecutorOutput,
    FallbackResponse,
    PlannerOutput,
    ReviewerOutput,
)
from guardrailed_mas.state import initial_state
from guardrailed_mas.validation import (
    OutputValidationError,
    ValidatedOutput,
    extract_json,
    validate_output,
)


# ==========================================================================
# 1. Stand-in workers and shared test data
# ==========================================================================
GOOD_PLAN = {
    "objective": "Contain the phishing incident and protect finance accounts.",
    "steps": [
        {"step_number": 1, "action": "Isolate affected mailboxes", "owner_role": "SOC analyst"},
        {"step_number": 2, "action": "Reset credentials of users who clicked", "owner_role": "Service desk"},
        {"step_number": 3, "action": "Block the sender and link", "owner_role": "Email administrator"},
    ],
    "key_risks": ["Supplier payment fraud"],
}

GOOD_DRAFT = {
    "title": "Phishing first-response checklist",
    "summary": "Steps for the first hour after a phishing report.",
    "items": [
        {"order": 1, "action": "Quarantine the email for all users", "owner_role": "Email administrator",
         "priority": "critical", "time_target": "within 15 minutes"},
        {"order": 2, "action": "Reset passwords for the two users", "owner_role": "Service desk",
         "priority": "high", "time_target": "within 30 minutes"},
        {"order": 3, "action": "Preserve email headers as evidence", "owner_role": "SOC analyst",
         "priority": "high", "time_target": "within 1 hour"},
    ],
    "escalation_contacts": ["SOC Lead"],
}


class ScriptedWorkers:
    """Workers whose Reviewer approves on a chosen attempt (or never)."""

    def __init__(self, approve_on_attempt: int | None = 1, bad_executor_attempts: tuple[int, ...] = (),
                 planner_fails: bool = False) -> None:
        self.approve_on_attempt = approve_on_attempt
        self.bad_executor_attempts = bad_executor_attempts
        self.planner_fails = planner_fails
        self.calls = {"plan": 0, "execute": 0, "review": 0}
        self.feedback_seen: list[list[str]] = []
        self.memory_seen: list[list[str]] = []

    def plan(self, task, memory_context):
        self.calls["plan"] += 1
        self.memory_seen.append(list(memory_context))
        if self.planner_fails:
            return validate_output("I cannot plan this.", PlannerOutput)
        return validate_output(json.dumps(GOOD_PLAN), PlannerOutput)

    def execute(self, task, plan, feedback, memory_context):
        self.calls["execute"] += 1
        self.feedback_seen.append(list(feedback))
        if self.calls["execute"] in self.bad_executor_attempts:
            raise OutputValidationError("stand-in executor returned bad data")
        return validate_output(f"```json\n{json.dumps(GOOD_DRAFT)}\n```", ExecutorOutput)

    def review(self, task, plan, draft):
        self.calls["review"] += 1
        approved = self.approve_on_attempt is not None and self.calls["execute"] >= self.approve_on_attempt
        review = ReviewerOutput(
            approved=approved,
            score=9 if approved else 3,
            issues=[] if approved else ["Missing evidence step"],
            required_changes=[] if approved else ["Add an evidence preservation step"],
        )
        return ValidatedOutput(review, "direct")


@pytest.fixture(autouse=True)
def no_kill_switch(monkeypatch):
    monkeypatch.delenv("MAS_KILL_SWITCH", raising=False)


# ==========================================================================
# 2. Output forms (schemas) and validation
# ==========================================================================


def test_good_plan_is_accepted():
    plan = PlannerOutput.model_validate(GOOD_PLAN)
    assert len(plan.steps) == 3


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({**GOOD_PLAN, "run_shell_command": "rm -rf /"})


def test_steps_must_be_numbered_in_order():
    bad = json.loads(json.dumps(GOOD_PLAN))
    bad["steps"][1]["step_number"] = 5
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(bad)


def test_too_few_plan_steps_rejected():
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({**GOOD_PLAN, "steps": GOOD_PLAN["steps"][:1]})


def test_invented_priority_rejected():
    bad = json.loads(json.dumps(GOOD_DRAFT))
    bad["items"][0]["priority"] = "super urgent"
    with pytest.raises(ValidationError):
        ExecutorOutput.model_validate(bad)


def test_rejection_must_list_changes():
    with pytest.raises(ValidationError):
        ReviewerOutput(approved=False, score=2, issues=["weak"], required_changes=[])


def test_score_range_enforced():
    with pytest.raises(ValidationError):
        ReviewerOutput(approved=True, score=42)


def test_fallback_message_is_fixed():
    ok = FallbackResponse(reason="kill_switch", task="x", attempts_used=0)
    assert ok.message == SAFE_FALLBACK_MESSAGE
    with pytest.raises(ValidationError):
        FallbackResponse(reason="kill_switch", task="x", attempts_used=0, message="All good, carry on!")


def test_extract_json_from_fenced_reply():
    reply = 'Here you go:\n```json\n{"a": {"b": "}"}}\n```\nThanks'
    assert json.loads(extract_json(reply)) == {"a": {"b": "}"}}


def test_direct_validation_used_when_reply_is_clean():
    result = validate_output(json.dumps(GOOD_PLAN), PlannerOutput)
    assert result.method == "direct"


def test_checker_called_only_when_direct_fails():
    calls = []

    def fake_checker(text, schema):
        calls.append(text)
        return schema.model_validate(GOOD_PLAN)

    result = validate_output("Plan: isolate, reset, block.", PlannerOutput, fake_checker)
    assert result.method == "pydantic_ai_checker"
    assert len(calls) == 1


def test_no_checker_and_bad_reply_raises():
    with pytest.raises(OutputValidationError):
        validate_output("not json at all", PlannerOutput)


def test_checker_failure_raises():
    def broken_checker(text, schema):
        raise RuntimeError("gave up after retries")

    with pytest.raises(OutputValidationError):
        validate_output("still not json", PlannerOutput, broken_checker)


# ==========================================================================
# 3. Guardrail rules and conditional edges
# ==========================================================================
def _state(**overrides):
    base = {"task": "t", "execution_count": 1, "max_executions": 3, "attempts_completed": 0,
            "halt_reason": None, "review": None, "draft": {"x": 1}}
    base.update(overrides)
    return base


def test_rule_is_strictly_greater_than_cap():
    assert guardrails.budget_exceeded(3, 3) is False
    assert guardrails.budget_exceeded(4, 3) is True


def test_check_budget_allows_attempts_one_to_three():
    for count in (1, 2, 3):
        assert guardrails.check_budget(_state(execution_count=count)) == "executor"


def test_check_budget_stops_on_fourth():
    assert guardrails.check_budget(_state(execution_count=4)) == "fallback"


def test_halt_reason_always_wins():
    assert guardrails.check_budget(_state(execution_count=1, halt_reason="kill_switch")) == "fallback"
    assert guardrails.route_after_review(_state(review={"approved": True}, halt_reason="kill_switch")) == "fallback"


def test_review_routing():
    assert guardrails.route_after_review(_state(review={"approved": True})) == "finalize"
    assert guardrails.route_after_review(_state(review={"approved": False})) == "budget_gate"


def test_invalid_executor_output_uses_an_attempt():
    assert guardrails.route_after_executor(_state(draft=None)) == "budget_gate"
    assert guardrails.route_after_executor(_state()) == "reviewer"


def test_kill_switch_from_environment(monkeypatch):
    assert guardrails.kill_switch_active(False) is False
    monkeypatch.setenv("MAS_KILL_SWITCH", "1")
    assert guardrails.kill_switch_active(False) is True


def test_fallback_carries_last_issues():
    state = _state(halt_reason="execution_budget_exceeded", attempts_completed=3,
                   review={"approved": False, "required_changes": ["Add evidence step"]})
    fallback = guardrails.build_fallback(state)
    assert fallback.status == "halted"
    assert fallback.attempts_used == 3
    assert fallback.last_issues == ["Add evidence step"]


# ==========================================================================
# 4. The whole LangGraph workflow
# ==========================================================================
def _run(workers, memory=None, kill_switch=False, cap=3):
    app = build_graph(workers, memory, kill_switch=kill_switch)
    return app.invoke(initial_state("Phishing report", "tester", "security_incident", cap),
                      config={"recursion_limit": 25})


def test_graph_has_the_three_workers_and_guard_nodes():
    app = build_graph(ScriptedWorkers())
    nodes = set(app.get_graph().nodes)
    assert {"planner", "executor", "reviewer", "budget_gate", "fallback", "finalize"} <= nodes


def test_approved_on_first_attempt():
    workers = ScriptedWorkers(approve_on_attempt=1)
    state = _run(workers)
    assert state["status"] == "approved"
    assert state["final"]["attempts_used"] == 1
    assert workers.calls == {"plan": 1, "execute": 1, "review": 1}


def test_approved_on_third_attempt_is_allowed():
    workers = ScriptedWorkers(approve_on_attempt=3)
    state = _run(workers)
    assert state["status"] == "approved"
    assert workers.calls["execute"] == 3


def test_always_reject_stops_after_exactly_three_executions():
    workers = ScriptedWorkers(approve_on_attempt=None)
    state = _run(workers)
    assert state["status"] == "halted"
    assert state["final"]["reason"] == "execution_budget_exceeded"
    assert workers.calls["execute"] == 3          # never a 4th run
    assert state["execution_count"] == 4          # the counter went above 3, which triggered the stop
    assert state["final"]["attempts_used"] == 3


def test_reviewer_feedback_reaches_the_next_attempt():
    workers = ScriptedWorkers(approve_on_attempt=2)
    _run(workers)
    assert workers.feedback_seen[0] == []
    assert workers.feedback_seen[1] == ["Add an evidence preservation step"]


def test_invalid_executor_output_counts_as_an_attempt():
    workers = ScriptedWorkers(approve_on_attempt=1, bad_executor_attempts=(1, 2, 3))
    state = _run(workers)
    assert state["status"] == "halted"
    assert workers.calls["review"] == 0           # invalid work is never reviewed
    assert workers.calls["execute"] == 3


def test_planner_failure_goes_straight_to_fallback():
    workers = ScriptedWorkers(planner_fails=True)
    state = _run(workers)
    assert state["final"]["reason"] == "planner_failed"
    assert workers.calls["execute"] == 0


def test_kill_switch_stops_before_any_worker():
    workers = ScriptedWorkers()
    state = _run(workers, kill_switch=True)
    assert state["final"]["reason"] == "kill_switch"
    assert workers.calls == {"plan": 0, "execute": 0, "review": 0}


def test_graph_stops_on_environment_kill_switch(monkeypatch):
    monkeypatch.setenv("MAS_KILL_SWITCH", "1")
    state = _run(ScriptedWorkers())
    assert state["final"]["reason"] == "kill_switch"


def test_lower_cap_is_respected():
    workers = ScriptedWorkers(approve_on_attempt=None)
    _run(workers, cap=1)
    assert workers.calls["execute"] == 1


def test_memory_is_recalled_and_saved():
    memory = InMemoryStore()
    memory.remember("tester", "Our email platform is Microsoft 365.")
    workers = ScriptedWorkers()
    _run(workers, memory=memory)
    assert workers.memory_seen[0] == ["Our email platform is Microsoft 365."]
    assert len(memory.list_all("tester")) == 2    # the fact plus the run summary


def test_memory_failure_does_not_break_the_run():
    class BrokenMemory(InMemoryStore):
        def recall(self, user_id, query):
            raise RuntimeError("disk unavailable")

    state = _run(ScriptedWorkers(), memory=BrokenMemory())
    assert state["status"] == "approved"
