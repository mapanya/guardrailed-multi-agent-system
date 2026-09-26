# Design Decisions

This note explains the main choices behind the system, with most attention on the two the brief asks about: the loop cap (N <= 3) and the schema design.

## 1. Why the loop cap is 3 (N <= 3)

**The problem.** In the Alibaba ROME incident an autonomous agent kept acting with no hard stop. A Planner, Executor and Reviewer loop has the same weakness: a strict Reviewer and a weak Executor can reject and retry forever, burning tokens, money and time. Every extra loop is another chance for unwanted actions.

**Why 3 and not more.**

* **Most of the value comes early.** The first attempt produces the work. The second fixes the Reviewer's specific points. A third is a last chance for anything still missing. If three attempts guided by specific feedback cannot satisfy the Reviewer, the problem is usually the task, the data or the Reviewer's rules. A human is the right next step.
* **Cost is bounded and predictable.** With the cap, one job makes at most 1 Planner call, 3 Executor calls and 3 Reviewer calls (7 worker calls). That upper limit can be planned against the free daily allowance and against rate limits.
* **Time is bounded.** A first-response checklist is only useful if it arrives quickly. Three short attempts keep the worst case to a few minutes.
* **It matches the brief.** The brief asks for N <= 3 with an immediate stop when `execution_count > 3`.

**Why not fewer.** A cap of 1 removes the Reviewer's ability to get improvements, which defeats the point of having a Reviewer. The cap is a setting (`max_executions` in `config/settings.yaml`), so a stricter environment can lower it without code changes. The tests check that a cap of 1 is honoured.

**How it is enforced.**

* `execution_count` is a field of the graph state (`GraphState`), so it travels with the job and cannot be reset by a worker.
* It is increased in one place only, the `budget_gate` node, which every Executor run must pass through, including the first one.
* The conditional edge `check_budget` applies the literal rule `execution_count > max_executions` and routes to `fallback`. The decision is plain Python with no model involved, so it cannot be argued with or talked around.
* An Executor reply that fails validation still uses up an attempt. Otherwise a model could loop forever by returning broken output.
* LangGraph's `recursion_limit` (25 steps) is an independent second limit on the whole graph.
* A kill-switch (`kill_switch` setting or `MAS_KILL_SWITCH=1`) stops the run before any worker is called and is checked again before every attempt.

**What the fallback returns.** `FallbackResponse` has a fixed status (`halted`), a reason from a fixed list (`execution_budget_exceeded`, `kill_switch`, `planner_failed`), the number of attempts used, the Reviewer's last required changes and a fixed message telling the user to hand the job to a human. The message is defined in code and a validator refuses any other text, so a model can never write its own "everything is fine" ending.

## 2. Why the schemas are designed this way

**The problem.** Free text from a model is unpredictable. If the next worker, or a person, relies on it, one malformed or invented field can break the flow or slip something unsafe through.

**Design Choices.**

| Choice | Reason |
|---|---|
| One schema per worker (`PlannerOutput`, `ExecutorOutput`, `ReviewerOutput`) plus `FinalResult` and `FallbackResponse` | Each hand-over between workers has a clear, testable contract |
| `extra="forbid"` on every model | Unknown fields are an error. A model cannot sneak in something like a `command_to_run` field |
| Length and count limits (for example 3 to 10 plan steps, at most 15 checklist items) | Stops empty answers and stops runaway answers that waste tokens |
| Fixed word lists (`priority` must be critical, high, medium or low) | Values stay consistent and machine-readable |
| Numeric ranges (`score` from 0 to 10) | Nonsense values are rejected |
| Cross-field rules (plan steps must be numbered 1, 2, 3 in order; a rejection must list required changes) | Catches answers that are well formed but still unusable |
| Reviewer decision is a true or false field (`approved`) | Routing depends on a typed value, not on searching text for the word "approved" |
| Generic field names (`items`, `owner_role`, `time_target`) | The same schemas serve other job types, which keeps the engine reusable |

**Why two layers of validation.** The direct check is free and fully predictable, so it runs first. The PydanticAI checker is only called when the direct check fails. PydanticAI gives the model the schema as its required output type, validates the answer and sends the errors back for a limited number of retries (`checker_retries: 2`). This keeps token use low on the normal path and still recovers from small formatting slips.

**Fail closed.** If a reply still cannot be validated, it is never passed on. A failed Executor reply counts as a used attempt. A failed Reviewer reply counts as a rejection, never an approval. A failed plan ends the run with the fallback.

## 3. Other decisions

**LangGraph for control, CrewAI for the workers.** LangGraph owns every routing decision, the counter and the stop. Each LangGraph node runs one CrewAI agent with a role, goal and background. The CrewAI agents have `allow_delegation=False`, no tools and `max_iter=1`, so they cannot start their own loops or hand work to each other. Only the graph can loop and the graph is capped.

**Memory: Mem0 on local SQLite and Chroma.** Mem0 keeps its change history in a SQLite file and a Chroma search index in the `data` folder, both on local disk. The workers receive at most 5 short, relevant facts per run (for example "incidents escalate to the SOC lead"), never the full earlier conversation, so memory does not grow the prompt over time. With `infer_facts: false`, saving a fact needs no model call at all. Memories are stored per `user_id`. Memory is best effort: if it fails the run continues and the problem is logged.

**Models.** Groq's free tier currently offers `openai/gpt-oss-120b` and `openai/gpt-oss-20b`. The larger one does the work and the smaller one does the checking, following the class advice to separate the work model from the helper model. `reasoning_effort: low` and `temperature: 0` keep answers fast, consistent and cheap. The client retries automatically when the service asks it to slow down. Model names are settings, so a retired model is a one-line change.

**Short supply chain.** The Week 1 case study was a supply-chain attack through a compromised model-routing package. This project connects to Groq with the official OpenAI and Groq libraries, pins every version in `requirements.txt` and turns off library telemetry.

**Evidence by design.** Every step writes one JSON line (node, counter, decision, validation method) to `logs/`. These logs, the executed notebook and screenshots form the run evidence.

**Tests without a key.** The guardrails, schemas and graph are tested with stand-in workers that approve, reject or return broken output on demand. The 33 tests run in under a second on any workstation and on GitHub Actions after every push.

## 4. Known limits and next steps

* The Reviewer is itself a model. Its rules are strict but it can still be wrong, which is why the fallback always routes to a human.
* Colab storage is temporary. For long-term memory, point `MAS_DATA_DIR` at Google Drive or run the system on a permanent server.
* A production version would add a spending limit per day, tracing (covered in Week 3) and access control on who may switch the kill-switch off.
