"""Validated, immutable configuration for a hockey data simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from hockeyai.models import AnomalyType, DEFAULT_ANOMALY_TYPES


@dataclass(frozen=True, slots=True)
class GoalProbabilityCoefficients:
    """Documented simulation coefficients; they are illustrative, not fitted."""

    beta_0: float = -1.2
    beta_distance: float = -0.08
    beta_angle: float = -0.015
    beta_skill: float = 1.2
    beta_goalie: float = 1.2
    beta_powerplay: float = 0.25

    def __post_init__(self) -> None:
        for name, value in (
            ("beta_0", self.beta_0),
            ("beta_distance", self.beta_distance),
            ("beta_angle", self.beta_angle),
            ("beta_skill", self.beta_skill),
            ("beta_goalie", self.beta_goalie),
            ("beta_powerplay", self.beta_powerplay),
        ):
            _validate_finite_number(name, value)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """All user-controlled settings that affect a simulation."""

    scenario_name: str = "reference"
    n_matches: int = 1000
    n_teams: int = 12
    seed: int = 42
    offensive_intensity: int = 50
    team_imbalance: int = 30
    penalties_per_match: float = 5.0
    home_advantage: float = 0.05
    anomaly_rate_percent: float = 2.0
    anomaly_types: frozenset[AnomalyType] = field(
        default_factory=lambda: DEFAULT_ANOMALY_TYPES
    )
    goal_probability: GoalProbabilityCoefficients = field(
        default_factory=GoalProbabilityCoefficients
    )
    players_per_team: int = field(default=18, init=False)
    goalies_per_team: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_name, str):
            raise ValueError("scenario_name must be a string")
        if not 1 <= len(self.scenario_name) <= 50:
            raise ValueError("scenario_name must contain 1 to 50 characters")
        if not self.scenario_name.strip():
            raise ValueError("scenario_name must not be blank")

        _validate_integer("n_matches", self.n_matches, minimum=100, maximum=10_000)
        _validate_integer("n_teams", self.n_teams, minimum=4, maximum=16)
        if self.n_teams % 2:
            raise ValueError("n_teams must be even")
        _validate_integer("seed", self.seed, minimum=0, maximum=2**32 - 1)
        _validate_integer(
            "offensive_intensity", self.offensive_intensity, minimum=0, maximum=100
        )
        _validate_integer("team_imbalance", self.team_imbalance, minimum=0, maximum=100)
        _validate_range("penalties_per_match", self.penalties_per_match, 0, 15)
        _validate_range("home_advantage", self.home_advantage, 0, 0.20)
        _validate_range(
            "anomaly_rate_percent", self.anomaly_rate_percent, 0, 20
        )

        if not isinstance(self.anomaly_types, frozenset) or any(
            not isinstance(item, AnomalyType) for item in self.anomaly_types
        ):
            raise ValueError("anomaly_types must be a frozenset of AnomalyType values")
        if self.anomaly_rate_percent > 0 and not self.anomaly_types:
            raise ValueError("anomaly_types must not be empty when anomaly rate is > 0")
        if not isinstance(self.goal_probability, GoalProbabilityCoefficients):
            raise ValueError("goal_probability must be GoalProbabilityCoefficients")

    @property
    def scenario_slug(self) -> str:
        """Return a path-safe slug without changing the configured display name."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.scenario_name.casefold()).strip("-")
        return slug or "scenario"


def _validate_integer(
    name: str, value: object, *, minimum: int, maximum: int
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _validate_finite_number(name: str, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _validate_range(name: str, value: object, minimum: float, maximum: float) -> None:
    number = _validate_finite_number(name, value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
