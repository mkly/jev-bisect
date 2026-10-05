"""Keep the search arithmetic in Python; ask Jev only for a direction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

MAX_TURNS = 20
Number = int | float


class Direction(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"
    EXACTLY = "exactly"


class SearchExhaustedError(ValueError):
    """A direction leaves no valid next guess or floating-point progress."""


@dataclass(frozen=True)
class SearchState:
    """Inclusive bounds and the guess to evaluate next."""

    max: Number
    min: Number
    last_guess: Number

    def __post_init__(self) -> None:
        for name in ("min", "max", "last_guess"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite int or float")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self.min <= self.last_guess <= self.max:
            raise ValueError("state must satisfy min <= last_guess <= max")

    def to_dict(self) -> dict[str, Number]:
        return {"max": self.max, "min": self.min, "last_guess": self.last_guess}


@dataclass(frozen=True)
class Turn:
    """One evaluated guess and Jev's decision."""

    state: SearchState
    choice: Direction
    confidence: float
    probabilities: Mapping[str, float]
    model: str


@dataclass(frozen=True)
class SearchResult:
    """Final evaluated state; only an exactly decision supplies an answer."""

    state: SearchState
    history: tuple[Turn, ...]
    stop_reason: Literal["exactly", "max_turns", "stalled"]

    @property
    def converged(self) -> bool:
        return self.stop_reason == "exactly"

    @property
    def answer(self) -> Number | None:
        return self.state.last_guess if self.converged else None

    @property
    def turns(self) -> int:
        return len(self.history)


def _state(value: SearchState | Mapping[str, Number]) -> SearchState:
    return value if isinstance(value, SearchState) else SearchState(**value)


def _validate_integer(state: SearchState, integer: bool) -> None:
    if not isinstance(integer, bool):
        raise ValueError("integer must be a bool")
    if integer and any(type(v) is not int for v in state.to_dict().values()):
        raise ValueError("integer mode requires int bounds and last_guess")


def advance(
    state: SearchState | Mapping[str, Number],
    choice: Direction | str,
    *,
    integer: bool = False,
) -> SearchState:
    """Apply one choice without calling Jev or mutating the caller's state.

    In continuous mode, move the relevant bound to last_guess and halve
    the distance to the other bound. Integer mode excludes the rejected
    guess and takes the floor midpoint of the remaining inclusive range.
    """
    current = _state(state)
    _validate_integer(current, integer)
    direction = Direction(choice)
    if direction == Direction.EXACTLY:
        return current

    low, high = current.min, current.max
    guess = current.last_guess
    if direction == Direction.HIGHER:
        if guess >= high:
            raise SearchExhaustedError("higher at max leaves no possible answer")
        low = guess + 1 if integer else guess
    else:
        if guess <= low:
            raise SearchExhaustedError("lower at min leaves no possible answer")
        high = guess - 1 if integer else guess

    if integer:
        midpoint = (low + high) // 2
    else:
        # Avoid overflowing high-low when bounds straddle zero.
        try:
            midpoint = low / 2 + high / 2 if low < 0 < high else low + (high - low) / 2
        except OverflowError as exc:
            raise SearchExhaustedError(
                "bounds exceed floating-point range; use integer mode"
            ) from exc
    if midpoint == guess or not low <= midpoint <= high:
        raise SearchExhaustedError("no representable midpoint makes progress")
    return SearchState(max=high, min=low, last_guess=midpoint)


def bisect(
    state: SearchState | Mapping[str, Number],
    question: str,
    *,
    client: TypeSafeClient | None = None,
    model: str = "jev-latest",
    max_turns: int = MAX_TURNS,
    integer: bool = False,
) -> SearchResult:
    """Ask Jev higher/lower/exactly until exactly or at most 20 turns.

    Supply a numeric question, e.g. "How many minutes are in three hours?".
    A caller-supplied client stays open; an internally created client is
    closed on every exit. API failures propagate. SDK retries are disabled
    for these calls so the turn limit also bounds HTTP evaluation attempts.
    """
    current = _state(state)
    _validate_integer(current, integer)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if type(max_turns) is not int or not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(f"max_turns must be an integer from 1 to {MAX_TURNS}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")

    comparison = Choice(
        instructions=(
            f"Numeric question: {question}\n"
            "Determine the numeric answer to this question and compare it with "
            "state.last_guess. Select higher if the answer is greater than "
            "last_guess, lower if it is less, or exactly if it equals last_guess. "
            "state.min and state.max are the current search bounds. "
            "Do not select exactly merely because the interval is small."
        ),
        criteria={
            "higher": "The answer is strictly greater than state.last_guess.",
            "lower": "The answer is strictly less than state.last_guess.",
            "exactly": "The answer equals state.last_guess.",
        },
    )
    owned = client is None
    active_client = TypeSafeClient() if client is None else client
    history: list[Turn] = []
    try:
        for turn in range(max_turns):
            response = active_client.system_one(
                state=current.to_dict(),
                questions={"direction": comparison},
                model=model,
                retry=RetryPolicy(max_retries=0),
            )
            answer = response.choices["direction"]
            direction = Direction(answer.choice)
            history.append(
                Turn(
                    state=current,
                    choice=direction,
                    confidence=answer.confidence,
                    probabilities=dict(answer.probabilities),
                    model=response.model,
                )
            )
            if direction == Direction.EXACTLY:
                return SearchResult(current, tuple(history), "exactly")
            if turn + 1 == max_turns:
                return SearchResult(current, tuple(history), "max_turns")
            try:
                current = advance(current, direction, integer=integer)
            except SearchExhaustedError:
                return SearchResult(current, tuple(history), "stalled")
    finally:
        if owned:
            active_client.close()
    raise AssertionError("unreachable: max_turns is at least one")
