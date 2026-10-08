# jev-bisect

Find numeric answers with Jev through repeated `higher`, `lower`, or `exactly`
decisions. Supply a question and bounds; the library calculates the guesses
using the official [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python).

## Install and run

Requires Python 3.10 or newer. Install from PyPI:

```sh
pip install jev-bisect
export TYPESAFE_API_KEY="your-typesafe-api-key"
```

```python
from jev_bisect import bisect

result = bisect(
    "How many slices in a pizza?",
    min=0,
    max=100,
)
print(result.answer)
print(result.turns)
```

`min` and `max` are required keyword arguments and must be finite numbers with
**`min < max`**. Omit `last_guess` or pass `None` to start at the midpoint.
To choose the first guess, supply an override such as
`bisect(question, min=0, max=100, last_guess=25)`.

A result is returned only after Jev selects `exactly`. It contains `answer`,
`turns`, the final `state`, and `history`. Each history entry records the
state, choice, confidence, probabilities, and model version.

## Configuration

```python
from jev_bisect import SearchConfig, bisect

result = bisect(
    "What is pi?",
    min=3,
    max=4,
    config=SearchConfig(precision=0.01, max_turns=10),
)
print(result.answer)  # 3.14 if Jev makes the correct comparisons
```

| Field | Default | Meaning |
| --- | --- | --- |
| `precision` | `1` | Smallest search increment; a positive finite `int` or `float`. |
| `max_turns` | `20` | Maximum evaluated guesses; an integer from 1 through 20. |

Omitting `config` or passing `None` uses these defaults. Both settings belong
in `SearchConfig`. Each evaluated guess counts as one turn, including the
initial midpoint. SDK retries are disabled for these calls.

## Precision and bounds behavior

Candidates are multiples of `precision` measured from zero: `1` searches
whole numbers, `0.01` searches hundredths, and `2.5` searches values such as
`0`, `2.5`, and `5`. Jev is instructed to round the answer to the nearest
multiple before comparing, with halfway values rounded away from zero.
Precision specifies search resolution, not significant digits or an error
tolerance.

After `higher` or `lower`, the current guess is excluded and the next guess
is the floor midpoint of the remaining candidates. For `min=0, max=100` at
default precision, the choices `higher`, `lower`, `exactly` evaluate `50`,
`75`, and `62`. Rounding down also applies to negative numbers: the first
guess for `min=-5, max=0` is `-3`.

Bounds snap inward to the candidate grid. With `min=0.001, max=0.019` and
precision 0.01, the only candidate is `0.01`. With `min=0.1, max=5.9` and
precision 1, candidates are the integers 1 through 5. Any explicit
`last_guess` must be within the bounds and a multiple of `precision`.

Endpoints on the grid are reachable. A valid search may narrow internally to
one candidate, which still needs an `exactly` decision. Choose bounds that
contain the rounded answer; the search does not expand them. A large range
or fine precision may require more guesses than the configured limit, and
Jev can make incorrect comparisons.

Guesses use exact rational arithmetic and integer candidate indexes internally.
Whole-number increments, including `1.0`, return integers; fractional increments
return floats. Trailing zeros are not preserved.

## Exceptions

| Exception | Condition |
| --- | --- |
| `ValueError` | Invalid bounds (including `min == max`), config, question, model, or starting guess. |
| `SearchExhaustedError` | No candidate within bounds, no possible next guess, or a candidate cannot be represented as a float at the configured precision. |
| `MaxTurnsExceededError` | No `exactly` decision within `max_turns`. |

Failures during the model loop include the last evaluated `state` and completed
`history`; `MaxTurnsExceededError` also includes `max_turns`. Exhaustion before
any evaluation has `state=None` and empty history. SDK errors propagate.

```python
from jev_bisect import MaxTurnsExceededError, SearchExhaustedError, bisect

try:
    result = bisect("How many minutes are in three hours?", min=0, max=300)
except (MaxTurnsExceededError, SearchExhaustedError) as error:
    print(error)
    print(error.state, error.history)
else:
    print(result.answer)
```

## Apply a choice yourself

`advance` performs one step without an API call and returns a `SearchState`.
When `last_guess` is omitted, the choice applies to the bounds' midpoint.
Supply the returned bounds and guess for the next step, using the same config.

```python
from jev_bisect import advance

state = advance("higher", min=0, max=100)
assert state.to_dict() == {"max": 100, "min": 51, "last_guess": 75}

state = advance("lower", min=state.min, max=state.max, last_guess=state.last_guess)
assert state.last_guess == 62
```

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

The default model is [`jev-latest`](https://docs.typesafe.ai/models). A supplied
client remains open; otherwise, `bisect` creates and closes its own client on
success or error.

## Development

```sh
pip install -e '.[dev]'
pytest
ruff check .
python -m build
```

Tests use deterministic responses and a mock HTTP transport, requiring no API
key or network requests.

## Prepare a release

Update `version` in `pyproject.toml` for each new release, then build and check
the wheel and source distribution:

```sh
pip install -e '.[dev,release]'
pytest
ruff check .
python -m build
python -m twine check --strict dist/*
```

These commands prepare files locally. When ready to publish, set
`TWINE_USERNAME=__token__` and `TWINE_PASSWORD` to a PyPI API token, then upload
only the files for that release:

```sh
python -m twine upload dist/jev_bisect-0.1.0-py3-none-any.whl dist/jev_bisect-0.1.0.tar.gz
```

## License

MIT; see [LICENSE](LICENSE).
