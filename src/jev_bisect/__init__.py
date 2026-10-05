"""Numeric bisection driven by Jev's structured choices."""

from .search import (
    MAX_TURNS,
    Direction,
    SearchConfig,
    SearchExhaustedError,
    SearchResult,
    SearchState,
    Turn,
    advance,
    bisect,
)

__all__ = [
    "MAX_TURNS",
    "Direction",
    "SearchConfig",
    "SearchExhaustedError",
    "SearchResult",
    "SearchState",
    "Turn",
    "advance",
    "bisect",
]
