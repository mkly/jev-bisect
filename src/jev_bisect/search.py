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
class SearchConfig:
    """Optional search settings; each evaluated guess consumes one turn."""

    max_turns: int = MAX_TURNS

    def __post_init__(self) -> None:
        if type(self.max_turns) is not int or not 1 <= self.max_turns <= MAX_TURNS:
            raise ValueError(f"max_turns must be an integer from 1 to {MAX_TURNS}")


@dataclass(frozen=True)
class SearchState:
    """Inclusive bounds and an existing guess to evaluate next.

    Pass min and max to bisect to generate the initial guess.
    """

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


def _midpoint(low: Number, high: Number, integer: bool) -> Number:
    if low == high:
        return low
    if integer:
        return (low + high) // 2
    # Avoid overflowing high-low when bounds straddle zero.
    try:
        return low / 2 + high / 2 if low < 0 < high else low + (high - low) / 2
    except OverflowError as exc:
        raise SearchExhaustedError(
            "bounds exceed floating-point range; use integer mode"
        ) from exc


def _initial_state(
    *, min: Number, max: Number, last_guess: Number | None, integer: bool
) -> SearchState:
    # Validate the bounds before computing the initial guess.
    bounds = SearchState(
        max=max, min=min, last_guess=min if last_guess is None else last_guess
    )
    _validate_integer(bounds, integer)
    if last_guess is not None:
        return bounds
    return SearchState(
        max=bounds.max,
        min=bounds.min,
        last_guess=_midpoint(bounds.min, bounds.max, integer),
    )


def _validate_integer(state: SearchState, integer: bool) -> None:
    if not isinstance(integer, bool):
        raise ValueError("integer must be a bool")
    if integer and any(type(v) is not int for v in state.to_dict().values()):
        raise ValueError("integer mode requires int bounds and last_guess")


def advance(
    choice: Direction | str,
    *,
    min: Number,
    max: Number,
    last_guess: Number | None = None,
    integer: bool = False,
) -> SearchState:
    """Apply one choice without calling Jev or mutating the caller's state.

    In continuous mode, move the relevant bound to last_guess and halve
    the distance to the other bound. Integer mode excludes the rejected
    guess and takes the floor midpoint of the remaining inclusive range.
    If last_guess is omitted or None, apply the choice to the bounds' midpoint.
    """
    current = _initial_state(min=min, max=max, last_guess=last_guess, integer=integer)
    return _advance(current, choice, integer=integer)


def _advance(
    current: SearchState, choice: Direction | str, *, integer: bool
) -> SearchState:
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

    midpoint = _midpoint(low, high, integer)
    if midpoint == guess or not low <= midpoint <= high:
        raise SearchExhaustedError("no representable midpoint makes progress")
    return SearchState(max=high, min=low, last_guess=midpoint)


def bisect(
    question: str,
    *,
    min: Number,
    max: Number,
    last_guess: Number | None = None,
    client: TypeSafeClient | None = None,
    model: str = "jev-latest",
    config: SearchConfig | None = None,
    integer: bool = False,
) -> SearchResult:
    """Ask Jev higher/lower/exactly until exactly or at most 20 turns.

    Supply a numeric question and keyword-only min and max bounds.
    The initial guess is the bounds' midpoint (floor midpoint in integer
    mode), and its evaluation counts as turn one. Equal bounds still require
    Jev to select exactly. Integer mode requires int bounds; invalid bounds
    are rejected before an API call. A supplied last_guess is an optional
    starting-point override; omitted or None uses the midpoint.

    Fractional mode may not reach an endpoint or every real number exactly.
    Omit config to allow up to 20 guesses, or pass SearchConfig(max_turns=5)
    for a smaller limit. Returns answer=None on max_turns or stalled rather
    than assuming success.
    A caller-supplied client stays open; an internally created client is
    closed on every exit. API failures propagate. SDK retries are disabled
    for these calls so the turn limit also bounds HTTP evaluation attempts.
    """
    settings = SearchConfig() if config is None else config
    if not isinstance(settings, SearchConfig):
        raise ValueError("config must be a SearchConfig or None")
    current = _initial_state(min=min, max=max, last_guess=last_guess, integer=integer)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
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
        for turn in range(settings.max_turns):
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
            if turn + 1 == settings.max_turns:
                return SearchResult(current, tuple(history), "max_turns")
            try:
                current = _advance(current, direction, integer=integer)
            except SearchExhaustedError:
                return SearchResult(current, tuple(history), "stalled")
    finally:
        if owned:
            active_client.close()
    raise AssertionError("unreachable: max_turns is at least one")
