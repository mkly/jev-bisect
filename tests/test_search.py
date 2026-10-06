import json
import math
from decimal import Decimal
from types import SimpleNamespace

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient

from jev_bisect import (
    Direction,
    MaxTurnsExceededError,
    SearchConfig,
    SearchExhaustedError,
    SearchState,
    advance,
    bisect,
)


class Oracle:
    """Respond by comparing each actual request state to a known target."""

    def __init__(self, target=None, choices=None):
        self.target = target
        self.choices = iter(choices) if choices is not None else None
        self.requests = []
        self.closed = False

    def system_one(self, **kwargs):
        self.requests.append(kwargs)
        guess = kwargs["state"]["last_guess"]
        if self.choices is not None:
            choice = next(self.choices)
        else:
            choice = (
                "higher"
                if self.target > guess
                else "lower"
                if self.target < guess
                else "exactly"
            )
        return SimpleNamespace(
            model="jev-test",
            choices={
                "direction": SimpleNamespace(
                    choice=choice,
                    confidence=0.9,
                    probabilities={"higher": 0.1, "lower": 0.1, "exactly": 0.8},
                )
            },
        )

    def close(self):
        self.closed = True


def test_fractional_search_updates_bounds_and_preserves_input():
    state = {"max": 100, "min": 0, "last_guess": 50}
    client = Oracle(target=62.5)
    result = bisect(
        "What is the target?",
        **state,
        client=client,
        config=SearchConfig(precision=0.01),
    )
    assert result.answer == 62.5
    assert result.turns == 3
    assert [t.state.to_dict() for t in result.history] == [
        {"max": 100, "min": 0, "last_guess": 50},
        {"max": 100, "min": 50.01, "last_guess": 75.0},
        {"max": 74.99, "min": 50.01, "last_guess": 62.5},
    ]
    assert [t.choice for t in result.history] == [
        Direction.HIGHER,
        Direction.LOWER,
        Direction.EXACTLY,
    ]
    assert state == {"max": 100, "min": 0, "last_guess": 50}
    assert not client.closed
    assert result.history[0].model == "jev-test"
    assert result.history[0].confidence == 0.9


@pytest.mark.parametrize("precision, initial_guess", [(0.01, 2.5), (1, 2)])
def test_omitted_guess_starts_at_midpoint(precision, initial_guess):
    state = {"max": 5, "min": 0}
    client = Oracle(initial_guess)
    result = bisect(
        "Target?", **state, client=client, config=SearchConfig(precision=precision)
    )
    assert result.answer == initial_guess
    assert result.turns == 1
    assert client.requests[0]["state"]["last_guess"] == initial_guess
    assert state == {"max": 5, "min": 0}


@pytest.mark.parametrize("precision", [0.01, 1])
def test_omitted_guess_runs_multiple_turns(precision):
    result = bisect(
        "Target?",
        max=100,
        min=0,
        client=Oracle(75),
        config=SearchConfig(precision=precision),
    )
    assert result.answer == 75
    assert [turn.state.last_guess for turn in result.history] == [50, 75]


@pytest.mark.parametrize("guess", [0, 10])
def test_explicit_guess_is_evaluated_first(guess):
    result = bisect("Target?", max=100, min=0, last_guess=guess, client=Oracle(guess))
    assert result.answer == guess
    assert result.turns == 1


@pytest.mark.parametrize("precision, expected", [(0.01, 2.5), (1, 2)])
def test_none_guess_starts_at_midpoint(precision, expected):
    result = bisect(
        "Target?",
        min=0,
        max=5,
        last_guess=None,
        client=Oracle(expected),
        config=SearchConfig(precision=precision),
    )
    assert result.answer == expected
    assert result.turns == 1


@pytest.mark.parametrize(
    "args, kwargs",
    [
        (("Target?",), {"min": 0}),
        (("Target?",), {"max": 100}),
        (("Target?", 0, 100), {}),
    ],
)
def test_bounds_are_required_keyword_arguments(args, kwargs):
    with pytest.raises(TypeError):
        bisect(*args, **kwargs)


def test_advance_with_omitted_guess_applies_choice_to_midpoint():
    assert advance("higher", max=100, min=0).to_dict() == {
        "max": 100,
        "min": 51,
        "last_guess": 75.0,
    }


@pytest.mark.parametrize("precision", [0.01, 1])
def test_equal_bounds_are_rejected_before_request(precision):
    client = Oracle(3)
    config = SearchConfig(precision=precision)
    with pytest.raises(ValueError, match="strictly less"):
        bisect("Target?", max=3, min=3, client=client, config=config)
    assert not client.requests
    with pytest.raises(ValueError, match="strictly less"):
        advance("exactly", max=3, min=3, config=config)


@pytest.mark.parametrize(
    "state, precision",
    [
        ({"max": 0, "min": 5}, 0.01),
        ({"max": float("inf"), "min": 0}, 0.01),
        ({"max": 5, "min": True}, 0.01),
    ],
)
def test_invalid_bounds_without_guess_are_rejected(state, precision):
    client = Oracle(0)
    with pytest.raises(ValueError):
        bisect(
            "Target?", **state, client=client, config=SearchConfig(precision=precision)
        )
    assert not client.requests


def test_large_integer_bounds_initialize_without_losing_precision():
    base = 10**400
    result = bisect("Target?", max=base + 10, min=base, client=Oracle(base + 5))
    assert result.answer == base + 5
    assert result.turns == 1


@pytest.mark.parametrize("choice, expected", [("higher", 75), ("lower", 24)])
def test_advance_halves_distance(choice, expected):
    state = SearchState(max=100, min=0, last_guess=50)
    assert (
        advance(
            choice, min=state.min, max=state.max, last_guess=state.last_guess
        ).last_guess
        == expected
    )
    assert (
        advance("exactly", min=state.min, max=state.max, last_guess=state.last_guess)
        == state
    )


def test_exactly_stops_at_initial_guess():
    result = bisect("Target?", max=2, min=0, last_guess=2, client=Oracle(2))
    assert result.answer == 2
    assert result.turns == 1


@pytest.mark.parametrize("target", range(-10, 11))
def test_integer_search_finds_every_value_including_endpoints(target):
    client = Oracle(target)
    result = bisect("Target integer?", max=10, min=-10, last_guess=0, client=client)
    assert result.answer == target
    assert result.turns <= 6
    assert all(type(t.state.last_guess) is int for t in result.history)


@pytest.mark.parametrize("limit", [1, 5, 20])
def test_turn_limit_is_exact_and_does_not_claim_convergence(limit):
    client = Oracle(choices=["higher"] * 20)
    with pytest.raises(MaxTurnsExceededError) as caught:
        bisect(
            "Target?",
            max=100,
            min=0,
            client=client,
            config=SearchConfig(max_turns=limit, precision=1e-8),
        )
    assert len(client.requests) == len(caught.value.history) == limit
    assert caught.value.max_turns == limit
    assert caught.value.state.to_dict() == client.requests[-1]["state"]
    assert all(req["retry"].max_retries == 0 for req in client.requests)


def test_exactly_on_twentieth_turn_wins_over_limit():
    result = bisect(
        "Target?",
        max=1_000_000_000,
        min=0,
        last_guess=50,
        client=Oracle(choices=["higher"] * 19 + ["exactly"]),
    )
    assert result.turns == 20


@pytest.mark.parametrize(
    "state, choice",
    [
        ({"max": 10, "min": 0, "last_guess": 10}, "higher"),
        ({"max": 10, "min": 0, "last_guess": 0}, "lower"),
        ({"max": math.nextafter(1.0, math.inf), "min": 0, "last_guess": 1.0}, "higher"),
    ],
)
def test_impossible_direction_or_precision_stops(state, choice):
    config = SearchConfig(precision=1e-16)
    with pytest.raises(SearchExhaustedError):
        advance(choice, **state, config=config)
    client = Oracle(choices=[choice])
    with pytest.raises(SearchExhaustedError) as caught:
        bisect("Target?", **state, client=client, config=config)
    assert len(caught.value.history) == len(client.requests) == 1
    assert caught.value.state.to_dict() == state


def test_extreme_opposite_bounds_do_not_overflow_initial_midpoint():
    result = bisect(
        "Target?",
        max=1e308,
        min=-1e308,
        client=Oracle(0),
        config=SearchConfig(precision=0.01),
    )
    assert result.answer == 0.0


def test_large_integer_search_preserves_precision():
    base = 10**400
    result = bisect(
        "Target?",
        max=base + 10,
        min=base,
        last_guess=base + 5,
        client=Oracle(base + 9),
    )
    assert result.answer == base + 9


@pytest.mark.parametrize(
    "state",
    [
        {"max": 1, "min": 2, "last_guess": 1},
        {"max": 10, "min": 0, "last_guess": 11},
        {"max": float("inf"), "min": 0, "last_guess": 1},
        {"max": 10, "min": 0, "last_guess": float("nan")},
        {"max": 10, "min": False, "last_guess": 1},
        {"max": "10", "min": 0, "last_guess": 1},
    ],
)
def test_invalid_state_is_rejected_before_request(state):
    client = Oracle(1)
    with pytest.raises(ValueError):
        bisect("Target?", **state, client=client)
    assert not client.requests


@pytest.mark.parametrize(
    "options",
    [
        {"config": {}},
        {"model": ""},
    ],
)
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        bisect("Target?", max=10.0, min=0, last_guess=5, **options)


@pytest.mark.parametrize("limit", [0, -1, 21, True, 1.5, None, "20"])
def test_config_rejects_invalid_guess_limits(limit):
    with pytest.raises(ValueError, match="max_turns"):
        SearchConfig(max_turns=limit)


@pytest.mark.parametrize("config", [None, SearchConfig()])
def test_default_config_limits_evaluations_to_twenty(config):
    client = Oracle(choices=["higher"] * 21)
    with pytest.raises(MaxTurnsExceededError) as caught:
        bisect("Target?", min=0, max=1_000_000_000, client=client, config=config)
    assert len(caught.value.history) == len(client.requests) == 20
    assert caught.value.max_turns == 20


@pytest.mark.parametrize("question", ["", " ", None, 5])
def test_invalid_question_is_rejected(question):
    with pytest.raises(ValueError):
        bisect(question, max=10, min=0, last_guess=5)


def test_unknown_choice_is_rejected():
    with pytest.raises(ValueError):
        advance("maybe", max=10, min=0, last_guess=5)
    with pytest.raises(ValueError):
        bisect("Target?", max=10, min=0, last_guess=5, client=Oracle(choices=["maybe"]))


@pytest.mark.parametrize("fail", [False, True])
def test_owned_client_closes_even_on_error(monkeypatch, fail):
    client = Oracle(5)
    if fail:

        def raise_error(**kwargs):
            raise RuntimeError("API failed")

        client.system_one = raise_error
    monkeypatch.setattr("jev_bisect.search.TypeSafeClient", lambda **kwargs: client)
    if fail:
        with pytest.raises(RuntimeError, match="API failed"):
            bisect("Target?", max=10, min=0, last_guess=5)
    else:
        assert bisect("Target?", max=10, min=0, last_guess=5).answer == 5
    assert client.closed


def test_real_sdk_serialization_and_response_parsing(monkeypatch):
    requests = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def handle(request):
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        requests.append(payload)
        guess = payload["state"]["last_guess"]
        choice = "higher" if guess == 50 else "exactly"
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 100, "output_tokens": 0},
                "answers": {
                    "direction": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.99,
                        "probabilities": {
                            "higher": 0.99,
                            "lower": 0.005,
                            "exactly": 0.005,
                        },
                    }
                },
            },
        )

    monkeypatch.setattr(
        "jev_bisect.search.TypeSafeClient",
        lambda: TypeSafeClient(transport=httpx2.MockTransport(handle)),
    )
    result = bisect(
        "How many centimeters are in three quarters of a meter?",
        max=100,
        min=0,
        last_guess=50,
    )
    assert result.answer == 75
    assert len(requests) == 2
    assert requests[0]["model"] == "jev-latest"
    question = requests[0]["questions"]["direction"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"higher", "lower", "exactly"}
    assert "three quarters" in question["instructions"]


def test_integer_mode_is_default():
    assert SearchConfig().precision == 1
    result = bisect("Target?", min=0, max=5, client=Oracle(2))
    assert result.answer == 2
    assert type(result.answer) is int


@pytest.mark.parametrize("target", [i / 10 for i in range(-10, 11)])
def test_decimal_search_reaches_every_candidate_including_endpoints(target):
    result = bisect(
        "Target?",
        min=-1,
        max=1,
        client=Oracle(target),
        config=SearchConfig(precision=0.1),
    )
    assert result.answer == target
    assert result.turns <= 5
    assert all(
        Decimal(str(turn.state.last_guess)).as_tuple().exponent >= -1
        for turn in result.history
    )


@pytest.mark.parametrize("target", [0.0, 0.01, 0.1, 0.99, 1.0, -0.1])
def test_hundredth_precision_reaches_candidates_and_sets_instructions(target):
    client = Oracle(target)
    result = bisect(
        "Target?",
        min=-1,
        max=1,
        client=client,
        config=SearchConfig(precision=0.01),
    )
    assert result.answer == target
    instructions = client.requests[0]["questions"]["direction"].instructions
    assert "Round the numeric answer to the nearest multiple of 0.01" in instructions
    assert "away from zero" in instructions


def test_thousandth_precision_reaches_candidates():
    result = bisect(
        "Target?",
        min=0,
        max=1,
        client=Oracle(0.123),
        config=SearchConfig(precision=0.001),
    )
    assert result.answer == 0.123
    assert all(
        Decimal(str(turn.state.last_guess)).as_tuple().exponent >= -3
        for turn in result.history
    )


def test_decimal_bounds_are_rounded_inward():
    result = bisect(
        "Target?",
        min=0.001,
        max=0.019,
        client=Oracle(0.01),
        config=SearchConfig(precision=0.01),
    )
    assert result.state.to_dict() == {"min": 0.01, "max": 0.01, "last_guess": 0.01}


def test_no_decimal_candidate_in_bounds_fails_before_request():
    client = Oracle(0)
    with pytest.raises(SearchExhaustedError, match="no value"):
        bisect(
            "Target?",
            min=0.001,
            max=0.009,
            client=client,
            config=SearchConfig(precision=0.01),
        )
    assert not client.requests


def test_decimal_override_must_match_configured_precision():
    client = Oracle(0.123)
    with pytest.raises(ValueError, match="precision"):
        bisect(
            "Target?",
            min=0,
            max=1,
            last_guess=0.123,
            client=client,
            config=SearchConfig(precision=0.01),
        )
    assert not client.requests


def test_unit_precision_accepts_float_bounds_and_returns_integers():
    result = bisect(
        "Target?",
        min=0.1,
        max=5.9,
        client=Oracle(4),
        config=SearchConfig(precision=1),
    )
    assert result.answer == 4.0
    assert type(result.answer) is int


def test_advance_uses_decimal_config_on_subsequent_steps():
    config = SearchConfig(precision=0.01)
    state = advance("higher", min=0, max=1, config=config)
    assert state.to_dict() == {"min": 0.51, "max": 1.0, "last_guess": 0.75}
    state = advance(
        "lower",
        min=state.min,
        max=state.max,
        last_guess=state.last_guess,
        config=config,
    )
    assert state.to_dict() == {"min": 0.51, "max": 0.74, "last_guess": 0.62}


@pytest.mark.parametrize(
    "value", [0, -1, -0.01, True, "2", None, float("inf"), float("nan")]
)
def test_config_rejects_invalid_precision(value):
    with pytest.raises(ValueError, match="precision"):
        SearchConfig(precision=value)


@pytest.mark.parametrize(
    "precision, target",
    [
        (0.05, 0.15),
        (0.25, -0.75),
        (2.5, 7.5),
        (3, 9),
    ],
)
def test_arbitrary_positive_increments_use_multiples_from_zero(precision, target):
    client = Oracle(target)
    result = bisect(
        "Target?",
        min=-10,
        max=11,
        client=client,
        config=SearchConfig(precision=precision),
    )
    assert result.answer == target
    for turn in result.history:
        ratio = Decimal(str(turn.state.last_guess)) / Decimal(str(precision))
        assert ratio == ratio.to_integral_value()
    assert f"nearest multiple of {precision}" in (
        client.requests[0]["questions"]["direction"].instructions
    )


def test_explicit_guess_must_be_aligned_with_arbitrary_increment():
    with pytest.raises(ValueError, match="multiple"):
        bisect(
            "Target?",
            min=0,
            max=10,
            last_guess=1,
            config=SearchConfig(precision=2.5),
        )


@pytest.mark.parametrize("precision", [1, 1.0])
def test_integral_precision_returns_integer_guesses(precision):
    result = bisect(
        "Target?",
        min=0.1,
        max=5.9,
        client=Oracle(4),
        config=SearchConfig(precision=precision),
    )
    assert result.answer == 4
    assert type(result.answer) is int


@pytest.mark.parametrize("equal", [0, 0.5, -1])
@pytest.mark.parametrize("last_guess", [None, "same"])
def test_equal_bounds_fail_with_or_without_explicit_guess(equal, last_guess):
    with pytest.raises(ValueError, match="strictly less"):
        bisect(
            "Target?",
            min=equal,
            max=equal,
            last_guess=equal if last_guess == "same" else None,
            config=SearchConfig(precision=0.01),
        )


def test_success_result_does_not_include_stop_reason():
    result = bisect("Target?", min=0, max=10, client=Oracle(5))
    assert result.answer == 5
    assert not hasattr(result, "stop_reason")
