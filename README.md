# jev-bisect

A Python library that uses Jev to find a numeric answer through repeated
`higher`, `lower`, or `exactly` decisions. It uses the official
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
    state={"max": 100, "min": 0, "last_guess": 50},
    question="How many centimeters are in three quarters of a meter?",
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

The initial state must satisfy `min <= last_guess <= max`, with finite numeric
values. Every turn sends the current state and the same numeric question to
Jev as a three-option `Choice` question.

- `higher`: set `min = last_guess`, then guess `(min + max) / 2`.
- `lower`: set `max = last_guess`, then guess `(min + max) / 2`.
- `exactly`: return the current guess as the answer immediately.

For example, starting at `{min: 0, max: 100, last_guess: 50}`, the choices
`higher`, `lower`, `exactly` evaluate `50`, `75`, and `62.5`.

The default and hard maximum is **20 turns**. You can set a smaller limit with
`max_turns=5`. SDK retries are disabled for these evaluations. A direction
outside the bounds or a midpoint that cannot make floating-point progress
stops with `stalled`. Exhausting the turn limit stops with `max_turns`.
For either unsuccessful stop, `answer` is `None`, and `state` contains the
last **evaluated** guess and its bounds; the full decisions remain in `history`.
Authentication, transport, and other SDK errors propagate to the caller.

Bisection assumes a single numeric answer within the starting bounds and
consistent comparisons. Jev can make incorrect decisions. Fractional bisection
cannot reach every real number exactly, so the turn cap may be reached even
with correct directions. No tolerance is silently treated as `exactly`.

## Integer answers

Use `integer=True` for counts, years, or other whole-number answers. All three
state values must be Python integers. A rejected guess is excluded from the
remaining range (`min = last_guess + 1` or `max = last_guess - 1`), and the next
guess is the floor midpoint. This allows reaching both endpoints and avoids
fractional guesses.

```python
result = bisect(
    {"max": 300, "min": 0, "last_guess": 150},
    "How many minutes are in three hours?",
    integer=True,
)
```

## Apply a choice yourself

`advance` performs one bisection step without an API call. Neither function
mutates the state you supply.

```python
from jev_bisect import advance

state = advance({"max": 100, "min": 0, "last_guess": 50}, "higher")
assert state.to_dict() == {"max": 100, "min": 50, "last_guess": 75.0}
```

`advance` raises `SearchExhaustedError` if the choice leaves no possible next
guess, and `ValueError` for invalid states or choices. `exactly` returns the
state unchanged.

## Reuse a client or pin a model

```python
from typesafe_sdk import TypeSafeClient
from jev_bisect import bisect

with TypeSafeClient() as client:
    result = bisect(
        {"max": 100, "min": 0, "last_guess": 50},
        "How many centimeters are in three quarters of a meter?",
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
