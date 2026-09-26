"""Output validation: every worker reply must fit its schema before it is used.

Two layers, cheapest first:

1. Direct check (no model call, fully predictable)
   Pull the JSON out of the reply and validate it against the Pydantic schema.

2. PydanticAI output checker (only if layer 1 fails)
   A small PydanticAI agent whose output_type IS the schema. PydanticAI forces
   the model to answer in that exact shape and automatically sends validation
   errors back for another try, up to a fixed number of retries.

If both layers fail, OutputValidationError is raised and the graph treats it
as a failed attempt. Invalid data is never passed on.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class OutputValidationError(Exception):
    """Raised when a worker reply cannot be turned into its schema."""


@dataclass
class ValidatedOutput(Generic[T]):
    value: T
    method: str  # "direct" or "pydantic_ai_checker"


def extract_json(text: str) -> Optional[str]:
    """Return the first complete JSON object found in a reply, or None."""
    if not text:
        return None
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : index + 1]
    return None


def validate_direct(text: str, schema: type[T]) -> T:
    """Layer 1: strict, local validation with no model call."""
    raw = extract_json(text)
    if raw is None:
        raise OutputValidationError(f"No JSON object found for {schema.__name__}.")
    try:
        json.loads(raw)
        return schema.model_validate_json(raw)
    except (ValueError, ValidationError) as error:
        raise OutputValidationError(f"{schema.__name__} failed validation: {error}") from error


# A checker takes (reply text, schema) and returns a schema instance.
Checker = Callable[[str, type[BaseModel]], BaseModel]


class PydanticAIChecker:
    """Layer 2: a PydanticAI agent that must return an instance of the schema."""

    INSTRUCTIONS = (
        "You convert a worker's reply into the required structured form. "
        "Keep the worker's meaning. Do not add new facts. "
        "Fill every required field and respect every limit in the form."
    )

    def __init__(self, model: Any, retries: int = 2) -> None:
        self._model = model
        self._retries = retries
        self._agents: dict[type[BaseModel], Any] = {}

    def _agent_for(self, schema: type[BaseModel]) -> Any:
        if schema not in self._agents:
            from pydantic_ai import Agent  # imported here so tests do not need it

            self._agents[schema] = Agent(
                self._model,
                output_type=schema,
                instructions=self.INSTRUCTIONS,
                name=f"{schema.__name__}_checker",
                retries={"tools": 1, "output": self._retries},
            )
        return self._agents[schema]

    def __call__(self, text: str, schema: type[BaseModel]) -> BaseModel:
        result = self._agent_for(schema).run_sync(f"Worker reply to convert:\n\n{text}")
        return result.output


def validate_output(text: str, schema: type[T], checker: Optional[Checker] = None) -> ValidatedOutput[T]:
    """Validate a worker reply against its schema, using the checker only if needed."""
    try:
        return ValidatedOutput(validate_direct(text, schema), "direct")
    except OutputValidationError as first_error:
        if checker is None:
            raise
        logger.warning("Direct validation failed, calling the PydanticAI checker: %s", first_error)
    try:
        value = checker(text, schema)
    except Exception as error:  # the checker gave up after its retries
        raise OutputValidationError(f"PydanticAI checker could not repair {schema.__name__}: {error}") from error
    if not isinstance(value, schema):
        raise OutputValidationError(f"Checker returned {type(value).__name__}, expected {schema.__name__}.")
    return ValidatedOutput(value, "pydantic_ai_checker")
