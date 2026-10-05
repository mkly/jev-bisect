import json
import math
from types import SimpleNamespace

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient

from jev_bisect import (
    Direction,
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
    result = bisect("What is the target?", **state, client=client)
    assert result.answer == 62.5
    assert result.converged
    assert result.stop_reason == "exactly"
    assert result.turns == 3
    assert [t.state.to_dict() for t in result.history] == [
        {"max": 100, "min": 0, "last_guess": 50},
        {"max": 100, "min": 50, "last_guess": 75.0},
        {"max": 75.0, "min": 50, "last_guess": 62.5},
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


@pytest.mark.parametrize("integer, initial_guess", [(False, 2.5), (True, 2)])
def test_omitted_guess_starts_at_midpoint(integer, initial_guess):
    state = {"max": 5, "min": 0}
    client = Oracle(initial_guess)
    result = bisect("Target?", **state, client=client, integer=integer)
    assert result.answer == initial_guess
    assert result.turns == 1
    assert client.requests[0]["state"]["last_guess"] == initial_guess
    assert state == {"max": 5, "min": 0}


@pytest.mark.parametrize("integer", [False, True])
def test_omitted_guess_runs_multiple_turns(integer):
    result = bisect("Target?", max=100, min=0, client=Oracle(75), integer=integer)
    assert result.answer == 75
    assert [turn.state.last_guess for turn in result.history] == [50, 75]


@pytest.mark.parametrize("guess", [0, 10])
def test_explicit_guess_is_evaluated_first(guess):
    result = bisect("Target?", max=100, min=0, last_guess=guess, client=Oracle(guess))
    assert result.answer == guess
    assert result.turns == 1


@pytest.mark.parametrize("integer, expected", [(False, 2.5), (True, 2)])
def test_none_guess_starts_at_midpoint(integer, expected):
    result = bisect(
        "Target?",
        min=0,
        max=5,
        last_guess=None,
        client=Oracle(expected),
        integer=integer,
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
        "min": 50.0,
        "last_guess": 75.0,
    }


@pytest.mark.parametrize("integer", [False, True])
def test_equal_bounds_supply_the_only_guess(integer):
    result = bisect("Target?", max=3, min=3, client=Oracle(3), integer=integer)
    assert result.answer == 3
    assert result.turns == 1


@pytest.mark.parametrize(
    "state, integer",
    [
        ({"max": 0, "min": 5}, False),
        ({"max": float("inf"), "min": 0}, False),
        ({"max": 5, "min": True}, False),
        ({"max": 5.0, "min": 0}, True),
    ],
)
def test_invalid_bounds_without_guess_are_rejected(state, integer):
    client = Oracle(0)
    with pytest.raises(ValueError):
        bisect("Target?", **state, client=client, integer=integer)
    assert not client.requests


def test_large_integer_bounds_initialize_without_losing_precision():
    base = 10**400
    result = bisect(
        "Target?", max=base + 10, min=base, client=Oracle(base + 5), integer=True
    )
    assert result.answer == base + 5
    assert result.turns == 1


@pytest.mark.parametrize("choice, expected", [("higher", 75), ("lower", 25)])
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
    result = bisect("Target?", max=2, min=2, last_guess=2, client=Oracle(2))
    assert result.answer == 2
    assert result.turns == 1


@pytest.mark.parametrize("target", range(-10, 11))
def test_integer_search_finds_every_value_including_endpoints(target):
    client = Oracle(target)
    result = bisect(
        "Target integer?", max=10, min=-10, last_guess=0, client=client, integer=True
    )
    assert result.answer == target
    assert result.turns <= 6
    assert all(type(t.state.last_guess) is int for t in result.history)


@pytest.mark.parametrize("limit", [1, 5, 20])
def test_turn_limit_is_exact_and_does_not_claim_convergence(limit):
    client = Oracle(choices=["higher"] * 20)
    result = bisect(
        "Target?",
        max=100,
        min=0,
        client=client,
        config=SearchConfig(max_turns=limit),
    )
    assert len(client.requests) == result.turns == limit
    assert result.stop_reason == "max_turns"
    assert not result.converged
    assert result.answer is None
    assert result.state.to_dict() == client.requests[-1]["state"]
    assert all(req["retry"].max_retries == 0 for req in client.requests)


def test_exactly_on_twentieth_turn_wins_over_limit():
    result = bisect(
        "Target?",
        max=100,
        min=0,
        last_guess=50,
        client=Oracle(choices=["higher"] * 19 + ["exactly"]),
    )
    assert result.converged
    assert result.turns == 20


@pytest.mark.parametrize(
    "state, choice",
    [
        ({"max": 10, "min": 0, "last_guess": 10}, "higher"),
        ({"max": 10, "min": 0, "last_guess": 0}, "lower"),
        ({"max": 1, "min": 1, "last_guess": 1}, "higher"),
        ({"max": math.nextafter(1.0, math.inf), "min": 0, "last_guess": 1.0}, "higher"),
    ],
)
def test_impossible_direction_or_precision_stops(state, choice):
    with pytest.raises(SearchExhaustedError):
        advance(choice, **state)
    client = Oracle(choices=[choice])
    result = bisect("Target?", **state, client=client)
    assert result.stop_reason == "stalled"
    assert result.turns == 1
    assert result.answer is None


def test_extreme_opposite_bounds_do_not_overflow_midpoint():
    state = advance("higher", max=1e308, min=-1e308, last_guess=-1e308)
    assert state.last_guess == 0.0


def test_large_integer_search_preserves_precision():
    base = 10**400
    result = bisect(
        "Target?",
        max=base + 10,
        min=base,
        last_guess=base + 5,
        client=Oracle(base + 9),
        integer=True,
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
        {"integer": True},
        {"integer": "yes"},
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
    result = bisect("Target?", min=0, max=100, client=client, config=config)
    assert result.turns == len(client.requests) == 20
    assert result.stop_reason == "max_turns"
    assert result.answer is None


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
