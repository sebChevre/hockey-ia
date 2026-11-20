"""Typed records for clean simulation tables and their reference truth."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from hockeyai.geometry import (
    REGULATION_LENGTH_SECONDS,
    period_for_elapsed_seconds,
    shot_geometry,
    validate_position,
)


class EventType(StrEnum):
    FACEOFF = "FACEOFF"
    SHOT = "SHOT"
    SAVE = "SAVE"
    GOAL = "GOAL"
    PENALTY = "PENALTY"
    PENALTY_END = "PENALTY_END"


class PlayerPosition(StrEnum):
    FORWARD = "F"
    DEFENSE = "D"
    GOALIE = "G"


class ShotOutcome(StrEnum):
    SAVED = "SAVED"
    GOAL = "GOAL"
    MISSED = "MISSED"


class AnomalyType(StrEnum):
    MISSING = "missing"
    DUPLICATE = "duplicate"
    INVALID_TIME = "invalid_time"
    UNKNOWN_TEAM = "unknown_team"
    INVALID_POSITION = "invalid_position"


DEFAULT_ANOMALY_TYPES = frozenset(AnomalyType)


@dataclass(frozen=True, slots=True)
class ManpowerSituation:
    """Skater counts in home-away order; each side retains at least three."""

    home_skaters: int = 5
    away_skaters: int = 5

    def __post_init__(self) -> None:
        for name, value in (
            ("home_skaters", self.home_skaters),
            ("away_skaters", self.away_skaters),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 3 <= value <= 5
            ):
                raise ValueError(f"{name} must be an integer between 3 and 5")

    @property
    def code(self) -> str:
        """Serialize counts in home-away order (for example, ``5v4``)."""
        return f"{self.home_skaters}v{self.away_skaters}"


@dataclass(frozen=True, slots=True)
class Team:
    team_id: str
    team_name: str
    attack_rating: float
    defense_rating: float
    goalie_rating: float

    def __post_init__(self) -> None:
        _validate_identifier("team_id", self.team_id)
        _validate_identifier("team_name", self.team_name)
        for name, value in (
            ("attack_rating", self.attack_rating),
            ("defense_rating", self.defense_rating),
            ("goalie_rating", self.goalie_rating),
        ):
            _validate_unit_interval(name, value)


@dataclass(frozen=True, slots=True)
class Player:
    player_id: str
    team_id: str
    position: PlayerPosition
    shooting_skill: float
    passing_skill: float
    defensive_skill: float
    games_played: int = 0
    ice_time_seconds: int = 0

    def __post_init__(self) -> None:
        _validate_identifier("player_id", self.player_id)
        _validate_identifier("team_id", self.team_id)
        if not isinstance(self.position, PlayerPosition):
            raise ValueError("position must be a PlayerPosition")
        for name, value in (
            ("shooting_skill", self.shooting_skill),
            ("passing_skill", self.passing_skill),
            ("defensive_skill", self.defensive_skill),
        ):
            _validate_unit_interval(name, value)
        _validate_nonnegative_integer("games_played", self.games_played)
        _validate_nonnegative_integer("ice_time_seconds", self.ice_time_seconds)


@dataclass(frozen=True, slots=True)
class Match:
    match_id: str
    home_team_id: str
    away_team_id: str
    home_score: int
    away_score: int
    simulation_seed: int

    def __post_init__(self) -> None:
        _validate_identifier("match_id", self.match_id)
        _validate_identifier("home_team_id", self.home_team_id)
        _validate_identifier("away_team_id", self.away_team_id)
        if self.home_team_id == self.away_team_id:
            raise ValueError("home_team_id and away_team_id must differ")
        _validate_nonnegative_integer("home_score", self.home_score)
        _validate_nonnegative_integer("away_score", self.away_score)
        if (
            isinstance(self.simulation_seed, bool)
            or not isinstance(self.simulation_seed, int)
            or not 0 <= self.simulation_seed <= 2**32 - 1
        ):
            raise ValueError("simulation_seed must be between 0 and 2^32 - 1")


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    match_id: str
    event_sequence: int
    elapsed_seconds: int
    period: int
    event_type: EventType
    team_id: str | None
    player_id: str | None
    x: float | None
    y: float | None
    related_shot_id: str | None
    home_score_before: int
    away_score_before: int
    manpower_situation: ManpowerSituation

    def __post_init__(self) -> None:
        _validate_identifier("event_id", self.event_id)
        _validate_identifier("match_id", self.match_id)
        _validate_nonnegative_integer("event_sequence", self.event_sequence)
        _validate_elapsed_seconds(self.elapsed_seconds)
        if (
            isinstance(self.period, bool)
            or not isinstance(self.period, int)
            or self.period != period_for_elapsed_seconds(self.elapsed_seconds)
        ):
            raise ValueError("period does not match elapsed_seconds")
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        for name, value in (
            ("team_id", self.team_id),
            ("player_id", self.player_id),
            ("related_shot_id", self.related_shot_id),
        ):
            if value is not None:
                _validate_identifier(name, value)
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must either both be set or both be None")
        if self.x is not None and self.y is not None:
            validate_position(self.x, self.y)
        _validate_nonnegative_integer("home_score_before", self.home_score_before)
        _validate_nonnegative_integer("away_score_before", self.away_score_before)
        if not isinstance(self.manpower_situation, ManpowerSituation):
            raise ValueError("manpower_situation must be a ManpowerSituation")


@dataclass(frozen=True, slots=True)
class Shot:
    shot_id: str
    event_id: str
    match_id: str
    team_id: str
    player_id: str
    elapsed_seconds: int
    x: float
    y: float
    distance_m: float
    angle_deg: float
    shot_type: str
    manpower_situation: ManpowerSituation
    shooting_skill: float
    goalie_skill: float
    score_difference_before: int
    is_goal: bool
    shot_outcome: ShotOutcome

    def __post_init__(self) -> None:
        for name in ("shot_id", "event_id", "match_id", "team_id", "player_id", "shot_type"):
            _validate_identifier(name, getattr(self, name))
        _validate_elapsed_seconds(self.elapsed_seconds)
        validate_position(self.x, self.y)
        expected_geometry = shot_geometry(self.x, self.y)
        _validate_finite_number("distance_m", self.distance_m)
        _validate_finite_number("angle_deg", self.angle_deg)
        if not math.isclose(self.distance_m, expected_geometry.distance_m, abs_tol=1e-9):
            raise ValueError("distance_m does not match the shot coordinates")
        if not math.isclose(self.angle_deg, expected_geometry.angle_deg, abs_tol=1e-9):
            raise ValueError("angle_deg does not match the shot coordinates")
        if not isinstance(self.manpower_situation, ManpowerSituation):
            raise ValueError("manpower_situation must be a ManpowerSituation")
        _validate_unit_interval("shooting_skill", self.shooting_skill)
        _validate_unit_interval("goalie_skill", self.goalie_skill)
        if isinstance(self.score_difference_before, bool) or not isinstance(
            self.score_difference_before, int
        ):
            raise ValueError("score_difference_before must be an integer")
        if not isinstance(self.is_goal, bool):
            raise ValueError("is_goal must be a boolean")
        if not isinstance(self.shot_outcome, ShotOutcome):
            raise ValueError("shot_outcome must be a ShotOutcome")
        if self.is_goal != (self.shot_outcome is ShotOutcome.GOAL):
            raise ValueError("is_goal must agree with shot_outcome")


@dataclass(frozen=True, slots=True)
class ShotGroundTruth:
    shot_id: str
    true_goal_probability: float

    def __post_init__(self) -> None:
        _validate_identifier("shot_id", self.shot_id)
        _validate_unit_interval("true_goal_probability", self.true_goal_probability)


@dataclass(frozen=True, slots=True)
class SimulationDataset:
    """In-memory typed tables; cross-table invariants are checked in phase B."""

    teams: tuple[Team, ...]
    players: tuple[Player, ...]
    matches: tuple[Match, ...]
    events: tuple[Event, ...]
    shots: tuple[Shot, ...]
    shot_ground_truth: tuple[ShotGroundTruth, ...] = ()

    def __post_init__(self) -> None:
        for name, rows, row_type in (
            ("teams", self.teams, Team),
            ("players", self.players, Player),
            ("matches", self.matches, Match),
            ("events", self.events, Event),
            ("shots", self.shots, Shot),
            ("shot_ground_truth", self.shot_ground_truth, ShotGroundTruth),
        ):
            if not isinstance(rows, tuple) or any(
                not isinstance(row, row_type) for row in rows
            ):
                raise ValueError(f"{name} must be a tuple of {row_type.__name__}")


def _validate_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_finite_number(name: str, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _validate_unit_interval(name: str, value: object) -> None:
    number = _validate_finite_number(name, value)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_elapsed_seconds(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= REGULATION_LENGTH_SECONDS
    ):
        raise ValueError(
            f"elapsed_seconds must be an integer between 0 and {REGULATION_LENGTH_SECONDS}"
        )
