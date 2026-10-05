# jev-bisect

A Python library that uses Jev to find a numeric answer through repeated
`higher`, `lower`, or `exactly` decisions. Supply a question and `min`/`max`
bounds; the library chooses the first guess and every subsequent guess.
It uses the official
[TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python) and defaults to
[`jev-latest`](https://docs.typesafe.ai/models).

## Install and run

Requires Python 3.10 or newer. From this directory:

```sh
pip install -e .
export TYPESAFE_API_KEY="your-typesafe-api-key"
```

```python
from jev_bisect import bisect

result = bisect(
    "How many centimeters are in three quarters of a meter?",
    min=0,
    max=100,
)

print(result.answer)  # 75.0 if Jev selects higher, then exactly
print(result.converged)  # True only when Jev selected exactly
print(result.turns)
print(result.stop_reason)  # exactly, max_turns, or stalled

for turn in result.history:
    print(turn.state.last_guess, turn.choice.value, turn.confidence)
```

The TypeSafe SDK reads `TYPESAFE_API_KEY` from the environment.

## Search behavior

Pass finite numeric `min` and `max` keyword arguments with `min <= max`.
The normal call is `bisect(question, min=0, max=100)`.
The library starts at `(min + max) / 2`, so these bounds start with a guess
of 50. No `last_guess` is needed from the caller.

The library records this generated guess as `last_guess` in the state sent
to Jev and in the returned history. Every turn sends the current state and
the same numeric question to Jev as a three-option `Choice` question.

- `higher`: set `min = last_guess`, then guess `(min + max) / 2`.
- `lower`: set `max = last_guess`, then guess `(min + max) / 2`.
- `exactly`: return the current guess as the answer immediately.

For example, starting with `min=0, max=100`, the choices
`higher`, `lower`, `exactly` evaluate `50`, `75`, and `62.5`.

The default and hard maximum is **20 turns**. The optional `config` argument
lets you set a smaller limit with `SearchConfig(max_turns=5)`. Each evaluation
of a guess counts as one turn, including the initial midpoint. SDK retries
are disabled for these evaluations. A direction
outside the bounds or a midpoint that cannot make floating-point progress
stops with `stalled`. Exhausting the turn limit stops with `max_turns`.
For either unsuccessful stop, `answer` is `None`, and `state` contains the
last **evaluated** guess and its bounds; the full decisions remain in `history`.
Authentication, transport, and other SDK errors propagate to the caller.

Bisection assumes a single numeric answer within the starting bounds and
consistent comparisons. Jev can make incorrect decisions. No tolerance is
silently treated as `exactly`.

## Optional configuration

```python
from jev_bisect import SearchConfig, bisect

result = bisect(
    "How many centimeters are in three quarters of a meter?",
    min=0,
    max=100,
    config=SearchConfig(max_turns=5),
)
```

Omitting `config`, passing `config=None`, or passing `SearchConfig()` uses the
default of 20 turns. `max_turns` must be a Python integer from 1 through
20; other values, including booleans, raise `ValueError`. If Jev selects
`exactly` on the final allowed guess, the search succeeds. `result.turns`
reports the number of guesses evaluated.

## Edge cases when starting with bounds only

| Arguments or condition | Behavior |
| --- | --- |
| `min=0, max=5` | First guess is `2.5`; with `integer=True`, it is `2`. |
| `min=-5, max=0, integer=True` | First guess is `-3`: floor rounding goes toward negative infinity. |
| `min == max` | The shared value is evaluated once. It becomes the answer only if Jev selects `exactly`; equal bounds alone do not establish success. |
| Reversed bounds, `NaN`, infinity, strings, or booleans | Invalid bounds raise `ValueError` before an API call. Both `min` and `max` are required. |
| Float bounds such as `5.0` with `integer=True` | Rejected with `ValueError`; integer mode requires Python `int` values. |
| The answer is at an endpoint | Integer mode can reach both endpoints. Fractional mode may keep approaching an endpoint without reaching it; use integer mode for whole-number answers. |
| No `exactly` decision within the configured turn limit | Returns `stop_reason="max_turns"` and `answer=None`. The initial midpoint evaluation counts as turn 1. |
| A direction leaves no possible next guess, or floating-point rounding prevents progress | Returns `stop_reason="stalled"` and `answer=None` if turns remain. It never treats a tiny interval as proof of equality. |
| Very large integer bounds in fractional mode | If computing the midpoint exceeds floating-point range, raises `SearchExhaustedError` before an API call. Integer mode computes the midpoint using integer arithmetic. |

Fractional bisection cannot reach every real number exactly. Even with correct
directions, an answer such as `0.1` within `[0, 1]` may not be evaluated within
the turn limit. Choose bounds that contain the answer; an answer outside the
range is not discovered by expanding the bounds.

Omit `last_guess`, or pass `last_guess=None`, to request automatic midpoint
initialization. An explicit numeric `last_guess` is an optional override and
must lie within the bounds. For example,
`bisect(question, min=0, max=100, last_guess=25)` evaluates 25 first. Zero is
a valid override, and integer mode requires an integer override.

`min`, `max`, and `last_guess` are keyword-only arguments. Both bounds are
required; omitting either or passing them positionally raises `TypeError`.
State is tracked internally and exposed as `SearchState` in results and
history. The search functions do not accept an input state dictionary.

## Integer answers

Use `integer=True` for counts, years, or other whole-number answers. The bounds
and any supplied `last_guess` must be Python integers. If `last_guess` is
omitted, the initial guess is the floor midpoint. A rejected guess is excluded
from the remaining range (`min = last_guess + 1` or `max = last_guess - 1`),
and the next guess is the floor midpoint. This allows reaching both endpoints
and avoids fractional guesses.

```python
result = bisect(
    "How many minutes are in three hours?",
    min=0,
    max=300,
    integer=True,
)
```

## Apply a choice yourself

`advance` performs one bisection step without an API call. It takes the
choice first, followed by the same keyword-only numeric arguments.

```python
from jev_bisect import advance

state = advance("higher", min=0, max=100)
assert state.to_dict() == {"max": 100, "min": 50, "last_guess": 75.0}

state = advance("lower", min=state.min, max=state.max, last_guess=state.last_guess)
assert state.last_guess == 62.5
```

When `last_guess` is omitted or `None`, `advance` applies the supplied choice
to the midpoint of the given bounds (50 in the first call above). Supply the
returned state's bounds and guess to subsequent calls so each choice applies
to the current guess.

`advance` raises `SearchExhaustedError` if the choice leaves no possible next
guess, and `ValueError` for invalid states or choices. `exactly` returns the
state unchanged.

## Reuse a client or pin a model

```python
from typesafe_sdk import TypeSafeClient
from jev_bisect import bisect

with TypeSafeClient() as client:
    result = bisect(
        "How many centimeters are in three quarters of a meter?",
        min=0,
        max=100,
        client=client,
        model="jev-1.13.0",
    )
```

A supplied client remains open. Otherwise, `bisect` creates and closes its own
client. Each history entry records the model version, choice, confidence, and
probabilities returned by Jev.

## Development

```sh
pip install -e '.[dev]'
pytest
ruff check .
python -m build
```

Tests use deterministic responses and a mock HTTP transport; they require no
API key and make no network requests.
