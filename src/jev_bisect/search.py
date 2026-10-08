from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

MAX_TURNS = 20
Number = int | float


class Direction(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"
    EXACTLY = "exactly"


class SearchExhaustedError(ValueError):
    """A direction leaves no valid next guess or floating-point progress."""

    def __init__(
        self,
        message: str,
        *,
        state: SearchState | None = None,
        history: tuple[Turn, ...] = (),
    ) -> None:
        super().__init__(message)
        self.state = state
        self.history = history


class MaxTurnsExceededError(RuntimeError):
    """Jev did not select exactly within the configured turn limit."""

    def __init__(
        self, max_turns: int, *, state: SearchState, history: tuple[Turn, ...]
    ) -> None:
        super().__init__(f"No exactly decision within {max_turns} turns")
        self.max_turns = max_turns
        self.state = state
        self.history = history


@dataclass(frozen=True)
class SearchConfig:
    """Optional search settings; each evaluated guess consumes one turn."""

    max_turns: int = MAX_TURNS
    precision: Number = 1

    def __post_init__(self) -> None:
        if type(self.max_turns) is not int or not 1 <= self.max_turns <= MAX_TURNS:
            raise ValueError(f"max_turns must be an integer from 1 to {MAX_TURNS}")
        if (
            isinstance(self.precision, bool)
            or not isinstance(self.precision, (int, float))
            or self.precision <= 0
            or (isinstance(self.precision, float) and not math.isfinite(self.precision))
        ):
            raise ValueError("precision must be a positive finite int or float")


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
    """A successful search's answer, final state, and evaluated turns."""

    state: SearchState
    history: tuple[Turn, ...]

    @property
    def answer(self) -> Number:
        return self.state.last_guess

    @property
    def turns(self) -> int:
        return len(self.history)


def _settings(config: SearchConfig | None) -> SearchConfig:
    if config is None:
        return SearchConfig()
    if not isinstance(config, SearchConfig):
        raise ValueError("config must be a SearchConfig or None")
    return config


def _ratio(value: Number, config: SearchConfig) -> tuple[int, int]:
    numerator, denominator = Decimal(str(value)).as_integer_ratio()
    step_numerator, step_denominator = Decimal(str(config.precision)).as_integer_ratio()
    return numerator * step_denominator, denominator * step_numerator


def _ticks(value: Number, config: SearchConfig, *, ceiling: bool = False) -> int:
    numerator, denominator = _ratio(value, config)
    return -(-numerator // denominator) if ceiling else numerator // denominator


def _value(ticks: int, config: SearchConfig) -> Number:
    step_numerator, step_denominator = Decimal(str(config.precision)).as_integer_ratio()
    if step_denominator == 1:
        return ticks * step_numerator
    try:
        value = ticks * step_numerator / step_denominator
    except OverflowError as exc:
        raise SearchExhaustedError("guess exceeds floating-point range") from exc
    numerator, denominator = _ratio(value, config)
    if numerator != ticks * denominator:
        raise SearchExhaustedError("guess cannot be represented at this precision")
    return value


def _initial_state(
    *,
    min: Number,
    max: Number,
    last_guess: Number | None,
    config: SearchConfig,
    allow_singleton: bool = False,
) -> SearchState:
    # Validate the bounds before computing the initial guess.
    bounds = SearchState(
        max=max, min=min, last_guess=min if last_guess is None else last_guess
    )
    if bounds.min == bounds.max and not allow_singleton:
        raise ValueError("min must be strictly less than max")
    low = _ticks(bounds.min, config, ceiling=True)
    high = _ticks(bounds.max, config)
    if low > high:
        raise SearchExhaustedError(
            "no value at the configured precision lies in bounds"
        )
    if last_guess is None:
        guess = _value((low + high) // 2, config)
    else:
        numerator, denominator = _ratio(last_guess, config)
        if numerator % denominator:
            raise ValueError(
                "last_guess must be a multiple of the configured precision"
            )
        guess = _value(numerator // denominator, config)
    return SearchState(
        max=_value(high, config),
        min=_value(low, config),
        last_guess=guess,
    )


def advance(
    choice: Direction | str,
    *,
    min: Number,
    max: Number,
    last_guess: Number | None = None,
    config: SearchConfig | None = None,
) -> SearchState:
    """Apply one choice without calling Jev or mutating the caller's state.

    The default precision is 1, for whole-number guesses.
    Exclude the rejected guess and take the floor midpoint of the remaining
    steps at config.precision, without binary floating-point arithmetic.
    If last_guess is omitted or None, apply the choice to the bounds' midpoint.
    Equal bounds are accepted with an explicit last_guess to continue a search
    that has narrowed to one candidate.
    """
    settings = _settings(config)
    current = _initial_state(
        min=min,
        max=max,
        last_guess=last_guess,
        config=settings,
        allow_singleton=last_guess is not None,
    )
    return _advance(current, choice, config=settings)


def _advance(
    current: SearchState, choice: Direction | str, *, config: SearchConfig
) -> SearchState:
    direction = Direction(choice)
    if direction == Direction.EXACTLY:
        return current

    low, high = _ticks(current.min, config), _ticks(current.max, config)
    guess = _ticks(current.last_guess, config)
    if direction == Direction.HIGHER:
        if guess >= high:
            raise SearchExhaustedError("higher at max leaves no possible answer")
        low = guess + 1
    else:
        if guess <= low:
            raise SearchExhaustedError("lower at min leaves no possible answer")
        high = guess - 1

    midpoint = (low + high) // 2
    if midpoint == guess or not low <= midpoint <= high:
        raise SearchExhaustedError("no representable midpoint makes progress")
    return SearchState(
        max=_value(high, config),
        min=_value(low, config),
        last_guess=_value(midpoint, config),
    )


def bisect(
    question: str,
    *,
    min: Number,
    max: Number,
    last_guess: Number | None = None,
    client: TypeSafeClient | None = None,
    model: str = "jev-latest",
    config: SearchConfig | None = None,
) -> SearchResult:
    """Ask Jev higher/lower/exactly until exactly or at most 20 turns.

    Supply a numeric question and keyword-only min and max bounds.
    The initial guess is the floor midpoint at the configured precision,
    and its evaluation counts as turn one. Input bounds must satisfy min < max;
    invalid or equal bounds are rejected before an API call.
    A supplied last_guess is an optional
    starting-point override; omitted or None uses the midpoint.

    Precision defaults to 1. Use SearchConfig(precision=0.01) for answers
    rounded to hundredths. Candidates are multiples of precision from zero.
    Omit config to allow up to 20 guesses, or pass SearchConfig(max_turns=5)
    for a smaller limit. Returns a result only after exactly. Raises
    MaxTurnsExceededError at the turn limit or SearchExhaustedError when the
    search cannot progress. Both include the last state and turn history.
    A caller-supplied client stays open; an internally created client is
    closed on every exit. API failures propagate. SDK retries are disabled
    for these calls so the turn limit also bounds HTTP evaluation attempts.
    """
    settings = _settings(config)
    current = _initial_state(min=min, max=max, last_guess=last_guess, config=settings)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")

    comparison = Choice(
        instructions=(
            f"Numeric question: {question}\n"
            "Determine the numeric answer to this question. "
            f"Round the numeric answer to the nearest multiple of {settings.precision} "
            "before comparing it with state.last_guess. Round halfway values "
            "away from zero. "
            "Compare that answer with state.last_guess. "
            "Select higher if the answer is greater than "
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
                return SearchResult(current, tuple(history))
            if turn + 1 == settings.max_turns:
                raise MaxTurnsExceededError(
                    settings.max_turns, state=current, history=tuple(history)
                )
            try:
                current = _advance(current, direction, config=settings)
            except SearchExhaustedError as exc:
                raise SearchExhaustedError(
                    str(exc), state=current, history=tuple(history)
                ) from exc
    finally:
        if owned:
            active_client.close()
    raise AssertionError("unreachable: max_turns is at least one")
