"""The LangGraph workflow: who works when and when everything must stop.

    START -> recall_memory -> planner --(ok)--> budget_gate --(count <= 3)--> executor
                                   \\--(failed)--> fallback      \\--(count > 3 or kill switch)--> fallback
    executor --(valid)--> reviewer --(approved)--> finalize -> save_memory -> END
             \\--(invalid)--> budget_gate      \\--(rejected)--> budget_gate
    fallback -> save_memory -> END

budget_gate is where execution_count is increased by 1. The conditional edge
check_budget then applies the rule "execution_count > max_executions means stop".
So the Executor can run at most max_executions (3) times.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from . import guardrails
from .agents import Workers
from .memory import MemoryStore
from .schemas import ExecutorOutput, FinalResult, PlannerOutput, ReviewerOutput
from .state import GraphState
from .validation import OutputValidationError

logger = logging.getLogger(__name__)


def _event(node: str, **details: Any) -> None:
    """One structured log line per step. These lines are the run evidence."""
    logger.info("node=%s %s", node, " ".join(f"{k}={v!r}" for k, v in details.items()), extra={"node": node, **details})


def build_graph(workers: Workers, memory: Optional[MemoryStore] = None, kill_switch: bool = False):
    """Wire the nodes and edges together and return a compiled, runnable graph."""

    # ------------------------------------------------------------------ nodes
    def recall_memory(state: GraphState) -> dict[str, Any]:
        facts: list[str] = []
        if memory is not None:
            try:
                facts = memory.recall(state["user_id"], state["task"])
            except Exception as error:  # memory is best effort
                logger.warning("Memory recall failed, continuing without it: %s", error)
        _event("recall_memory", user_id=state["user_id"], facts_found=len(facts))
        return {"memory_context": facts}

    def planner(state: GraphState) -> dict[str, Any]:
        if guardrails.kill_switch_active(kill_switch):
            _event("planner", halted="kill_switch")
            return {"halt_reason": "kill_switch"}
        try:
            result = workers.plan(state["task"], state.get("memory_context", []))
        except OutputValidationError as error:
            _event("planner", valid=False, error=str(error)[:200])
            return {"halt_reason": "planner_failed"}
        _event("planner", valid=True, validated_by=result.method, steps=len(result.value.steps))
        return {"plan": result.value.model_dump()}

    def budget_gate(state: GraphState) -> dict[str, Any]:
        if guardrails.kill_switch_active(kill_switch):
            _event("budget_gate", halted="kill_switch", execution_count=state["execution_count"])
            return {"halt_reason": "kill_switch"}
        count = state["execution_count"] + 1
        exceeded = guardrails.budget_exceeded(count, state["max_executions"])
        _event("budget_gate", execution_count=count, max_executions=state["max_executions"], exceeded=exceeded)
        update: dict[str, Any] = {"execution_count": count}
        if exceeded:
            update["halt_reason"] = "execution_budget_exceeded"
        return update

    def executor(state: GraphState) -> dict[str, Any]:
        plan = PlannerOutput.model_validate(state["plan"])
        attempt = state.get("attempts_completed", 0) + 1
        try:
            result = workers.execute(state["task"], plan, state.get("feedback", []), state.get("memory_context", []))
        except OutputValidationError as error:
            # Fail closed: invalid work counts as a used attempt and is never reviewed.
            _event("executor", attempt=attempt, valid=False, error=str(error)[:200])
            rejection = ReviewerOutput(
                approved=False,
                score=0,
                issues=["The previous answer did not match the required form."],
                required_changes=["Return the work as one JSON object that matches the schema exactly."],
            )
            return {"attempts_completed": attempt, "draft": None, "review": rejection.model_dump(),
                    "feedback": rejection.required_changes}
        _event("executor", attempt=attempt, valid=True, validated_by=result.method, items=len(result.value.items))
        return {"attempts_completed": attempt, "draft": result.value.model_dump(), "review": None}

    def reviewer(state: GraphState) -> dict[str, Any]:
        plan = PlannerOutput.model_validate(state["plan"])
        draft = ExecutorOutput.model_validate(state["draft"])
        try:
            result = workers.review(state["task"], plan, draft)
            review = result.value
            method = result.method
        except OutputValidationError as error:
            # Fail closed: an unreadable review is treated as a rejection, never an approval.
            review = ReviewerOutput(
                approved=False, score=0, issues=[f"Review could not be validated: {str(error)[:120]}"],
                required_changes=["Resubmit the work so it can be reviewed again."],
            )
            method = "failed_closed"
        _event("reviewer", approved=review.approved, score=review.score, validated_by=method,
               execution_count=state["execution_count"])
        return {"review": review.model_dump(), "feedback": list(review.required_changes)}

    def finalize(state: GraphState) -> dict[str, Any]:
        final = FinalResult(
            task=state["task"],
            attempts_used=state["attempts_completed"],
            plan=PlannerOutput.model_validate(state["plan"]),
            output=ExecutorOutput.model_validate(state["draft"]),
            review=ReviewerOutput.model_validate(state["review"]),
        )
        _event("finalize", status="approved", attempts_used=final.attempts_used)
        return {"status": "approved", "final": final.model_dump()}

    def fallback(state: GraphState) -> dict[str, Any]:
        response = guardrails.build_fallback(state)
        _event("fallback", status="halted", reason=response.reason, attempts_used=response.attempts_used,
               execution_count=state.get("execution_count", 0))
        return {"status": "halted", "final": response.model_dump()}

    def save_memory(state: GraphState) -> dict[str, Any]:
        if memory is None:
            return {}
        final = state.get("final") or {}
        if final.get("status") == "approved":
            title = final["output"]["title"]
            note = f"Completed job '{state['task'][:120]}' with approved output '{title}' after {final['attempts_used']} attempt(s)."
        else:
            note = f"Job '{state['task'][:120]}' was halted by guardrail '{final.get('reason')}' and handed to a human."
        try:
            memory.remember(state["user_id"], note, {"kind": "run_summary", "status": final.get("status")})
            _event("save_memory", saved=True)
        except Exception as error:
            logger.warning("Memory save failed: %s", error)
            _event("save_memory", saved=False)
        return {}

    # ------------------------------------------------------------------ graph
    graph = StateGraph(GraphState)
    for name, func in [
        ("recall_memory", recall_memory),
        ("planner", planner),
        ("budget_gate", budget_gate),
        ("executor", executor),
        ("reviewer", reviewer),
        ("finalize", finalize),
        ("fallback", fallback),
        ("save_memory", save_memory),
    ]:
        graph.add_node(name, func)

    graph.add_edge(START, "recall_memory")
    graph.add_edge("recall_memory", "planner")
    graph.add_conditional_edges("planner", guardrails.route_after_planner, ["budget_gate", "fallback"])
    graph.add_conditional_edges("budget_gate", guardrails.check_budget, ["executor", "fallback"])
    graph.add_conditional_edges("executor", guardrails.route_after_executor, ["reviewer", "budget_gate", "fallback"])
    graph.add_conditional_edges("reviewer", guardrails.route_after_review, ["finalize", "budget_gate", "fallback"])
    graph.add_edge("finalize", "save_memory")
    graph.add_edge("fallback", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()
