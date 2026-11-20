"""Deterministic event-driven simulator; independent of Streamlit and Pandas."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from dataclasses import replace
from typing import Callable

import numpy as np

from hockeyai.config import GoalProbabilityCoefficients, SimulationConfig
from hockeyai.geometry import (
    PERIOD_LENGTH_SECONDS,
    REGULATION_LENGTH_SECONDS,
    period_for_elapsed_seconds,
    shot_geometry,
)
from hockeyai.models import (
    Event,
    EventType,
    ManpowerSituation,
    Match,
    Player,
    PlayerPosition,
    Shot,
    ShotGroundTruth,
    ShotOutcome,
    SimulationDataset,
    Team,
)
from hockeyai.roster import generate_roster

ProgressCallback = Callable[[int, int], None]
_BASE_SHOTS_PER_MATCH = 62.5
_SHOT_TYPES = ("WRIST", "SLAP", "SNAP", "BACKHAND")


@dataclass(frozen=True, slots=True)
class _Penalty:
    event_id: str
    team_id: str
    player_id: str
    expires_at: float


def goal_probability(
    distance_m: float,
    angle_deg: float,
    shooting_skill: float,
    goalie_skill: float,
    is_powerplay: bool,
    coefficients: GoalProbabilityCoefficients | None = None,
) -> float:
    """Return bounded synthetic goal probability; coefficients are not empirical."""
    coefficients = coefficients or GoalProbabilityCoefficients()
    _finite("distance_m", distance_m)
    _finite("angle_deg", angle_deg)
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")
    _unit_interval("shooting_skill", shooting_skill)
    _unit_interval("goalie_skill", goalie_skill)
    if not isinstance(is_powerplay, bool):
        raise ValueError("is_powerplay must be a boolean")

    logit = (
        coefficients.beta_0
        + coefficients.beta_distance * distance_m
        + coefficients.beta_angle * abs(angle_deg)
        + coefficients.beta_skill * (shooting_skill - 0.5)
        - coefficients.beta_goalie * (goalie_skill - 0.5)
        + coefficients.beta_powerplay * is_powerplay
    )
    if logit >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        sigmoid = exp_logit / (1.0 + exp_logit)
    return float(np.clip(sigmoid, 0.005, 0.70))


def simulate_dataset(
    config: SimulationConfig,
    progress_callback: ProgressCallback | None = None,
) -> SimulationDataset:
    """Generate clean, deterministic tables using only a local PCG64 stream."""
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")

    rng = np.random.Generator(np.random.PCG64(config.seed))
    roster = generate_roster(config, rng)
    teams_by_id = {team.team_id: team for team in roster.teams}
    players_by_team = _players_by_team(roster.players)
    goalies_by_team = {
        team_id: tuple(
            player
            for player in team_players
            if player.position is PlayerPosition.GOALIE
        )
        for team_id, team_players in players_by_team.items()
    }
    team_games_played = {team.team_id: 0 for team in roster.teams}
    goalie_games_played = {goalie.player_id: 0 for goalies in goalies_by_team.values() for goalie in goalies}
    matches: list[Match] = []
    events: list[Event] = []
    shots: list[Shot] = []
    truth: list[ShotGroundTruth] = []

    for match_number in range(1, config.n_matches + 1):
        home_id, away_id = _choose_teams(config.n_teams, rng)
        home_goalies = goalies_by_team[home_id]
        away_goalies = goalies_by_team[away_id]
        home_goalie = home_goalies[team_games_played[home_id] % len(home_goalies)]
        away_goalie = away_goalies[team_games_played[away_id] % len(away_goalies)]
        team_games_played[home_id] += 1
        team_games_played[away_id] += 1
        goalie_games_played[home_goalie.player_id] += 1
        goalie_games_played[away_goalie.player_id] += 1
        match_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        match_rng = np.random.Generator(np.random.PCG64(match_seed))
        match_id = f"MATCH-{match_number:06d}"
        match, match_events, match_shots, match_truth = _simulate_match(
            config,
            match_id,
            home_id,
            away_id,
            match_seed,
            teams_by_id,
            players_by_team,
            home_goalie,
            away_goalie,
            match_rng,
        )
        matches.append(match)
        events.extend(match_events)
        shots.extend(match_shots)
        truth.extend(match_truth)
        if progress_callback is not None:
            progress_callback(match_number, config.n_matches)

    players = tuple(
        replace(
            player,
            games_played=(
                goalie_games_played[player.player_id]
                if player.position is PlayerPosition.GOALIE
                else team_games_played[player.team_id]
            ),
            ice_time_seconds=(
                goalie_games_played[player.player_id] * REGULATION_LENGTH_SECONDS
                if player.position is PlayerPosition.GOALIE
                else team_games_played[player.team_id] * 1000
            ),
        )
        for player in roster.players
    )
    return SimulationDataset(
        teams=roster.teams,
        players=players,
        matches=tuple(matches),
        events=tuple(events),
        shots=tuple(shots),
        shot_ground_truth=tuple(truth),
    )


def _simulate_match(
    config: SimulationConfig,
    match_id: str,
    home_team_id: str,
    away_team_id: str,
    match_seed: int,
    teams_by_id: dict[str, Team],
    players_by_team: dict[str, tuple[Player, ...]],
    home_goalie: Player,
    away_goalie: Player,
    rng: np.random.Generator,
) -> tuple[Match, tuple[Event, ...], tuple[Shot, ...], tuple[ShotGroundTruth, ...]]:
    home_team = teams_by_id[home_team_id]
    away_team = teams_by_id[away_team_id]
    events: list[Event] = []
    shots: list[Shot] = []
    truth: list[ShotGroundTruth] = []
    active_penalties: dict[str, _Penalty] = {}
    home_score = 0
    away_score = 0
    shot_number = 0
    event_number = 0
    pending: list[tuple[float, int, int, str, str | None]] = []
    pending_order = 0

    def schedule(at: float, priority: int, kind: str, token: str | None = None) -> None:
        nonlocal pending_order
        pending_order += 1
        heapq.heappush(pending, (at, priority, pending_order, kind, token))

    for period_start in (0, PERIOD_LENGTH_SECONDS, 2 * PERIOD_LENGTH_SECONDS):
        schedule(float(period_start), 1, "FACEOFF")
    penalty_count = int(rng.poisson(config.penalties_per_match))
    for penalty_number in range(penalty_count):
        start = float(rng.uniform(0, REGULATION_LENGTH_SECONDS - 120))
        schedule(start, 2, "PENALTY_START", str(penalty_number))

    def manpower() -> ManpowerSituation:
        home_active = sum(
            penalty.team_id == home_team_id for penalty in active_penalties.values()
        )
        away_active = sum(
            penalty.team_id == away_team_id for penalty in active_penalties.values()
        )
        return ManpowerSituation(max(3, 5 - home_active), max(3, 5 - away_active))

    def add_event(
        at: float,
        event_type: EventType,
        team_id: str | None = None,
        player_id: str | None = None,
        x: float | None = None,
        y: float | None = None,
        related_shot_id: str | None = None,
        situation: ManpowerSituation | None = None,
    ) -> Event:
        nonlocal event_number
        event_number += 1
        elapsed = min(REGULATION_LENGTH_SECONDS, max(0, int(at)))
        event = Event(
            event_id=f"{match_id}-EVENT-{event_number:06d}",
            match_id=match_id,
            event_sequence=event_number,
            elapsed_seconds=elapsed,
            period=period_for_elapsed_seconds(elapsed),
            event_type=event_type,
            team_id=team_id,
            player_id=player_id,
            x=x,
            y=y,
            related_shot_id=related_shot_id,
            home_score_before=home_score,
            away_score_before=away_score,
            manpower_situation=situation or manpower(),
        )
        events.append(event)
        return event

    current_time = 0.0
    while current_time < REGULATION_LENGTH_SECONDS:
        situation = manpower()
        home_rate = _team_attack_factor(
            home_team, away_team, situation.home_skaters, situation.away_skaters,
            is_home=True, home_advantage=config.home_advantage,
        )
        away_rate = _team_attack_factor(
            away_team, home_team, situation.away_skaters, situation.home_skaters,
            is_home=False, home_advantage=config.home_advantage,
        )
        total_rate = (
            _BASE_SHOTS_PER_MATCH
            * (0.5 + config.offensive_intensity / 100)
            * (home_rate + away_rate)
            / (2 * REGULATION_LENGTH_SECONDS)
        )
        shot_at = (
            current_time + float(rng.exponential(1 / total_rate))
            if total_rate > 0
            else math.inf
        )
        next_external_at = pending[0][0] if pending else math.inf

        if shot_at < next_external_at and shot_at < REGULATION_LENGTH_SECONDS:
            current_time = shot_at
            shooter_team_id = _choose_shooting_team(
                home_team_id, away_team_id, home_rate, away_rate, rng
            )
            opponent_id = away_team_id if shooter_team_id == home_team_id else home_team_id
            shooter = _choose_shooter(players_by_team[shooter_team_id], rng)
            attacking_skaters = (
                situation.home_skaters
                if shooter_team_id == home_team_id
                else situation.away_skaters
            )
            defending_skaters = (
                situation.away_skaters
                if shooter_team_id == home_team_id
                else situation.home_skaters
            )
            is_powerplay = attacking_skaters > defending_skaters
            x = float(rng.triangular(37, 52, 58))
            y = float(rng.triangular(5, 15, 25))
            geometry = shot_geometry(x, y)
            probability = goal_probability(
                geometry.distance_m,
                geometry.angle_deg,
                shooter.shooting_skill,
                teams_by_id[opponent_id].goalie_rating,
                is_powerplay,
                config.goal_probability,
            )
            is_goal = bool(rng.binomial(1, probability))
            outcome = (
                ShotOutcome.GOAL
                if is_goal
                else ShotOutcome.SAVED
                if rng.random() < 0.70
                else ShotOutcome.MISSED
            )
            shot_number += 1
            shot_id = f"{match_id}-SHOT-{shot_number:05d}"
            score_difference = (
                home_score - away_score
                if shooter_team_id == home_team_id
                else away_score - home_score
            )
            shot_event = add_event(
                current_time,
                EventType.SHOT,
                shooter_team_id,
                shooter.player_id,
                x,
                y,
                situation=situation,
            )
            shots.append(
                Shot(
                    shot_id=shot_id,
                    event_id=shot_event.event_id,
                    match_id=match_id,
                    team_id=shooter_team_id,
                    player_id=shooter.player_id,
                    elapsed_seconds=shot_event.elapsed_seconds,
                    x=x,
                    y=y,
                    distance_m=geometry.distance_m,
                    angle_deg=geometry.angle_deg,
                    shot_type=str(rng.choice(_SHOT_TYPES)),
                    manpower_situation=situation,
                    shooting_skill=shooter.shooting_skill,
                    goalie_skill=teams_by_id[opponent_id].goalie_rating,
                    score_difference_before=score_difference,
                    is_goal=is_goal,
                    shot_outcome=outcome,
                )
            )
            truth.append(ShotGroundTruth(shot_id, probability))
            if outcome is ShotOutcome.GOAL:
                add_event(
                    current_time,
                    EventType.GOAL,
                    shooter_team_id,
                    shooter.player_id,
                    x,
                    y,
                    related_shot_id=shot_id,
                    situation=situation,
                )
                if shooter_team_id == home_team_id:
                    home_score += 1
                else:
                    away_score += 1
                add_event(current_time, EventType.FACEOFF, situation=situation)
            elif outcome is ShotOutcome.SAVED:
                goalie = away_goalie if shooter_team_id == home_team_id else home_goalie
                add_event(
                    current_time,
                    EventType.SAVE,
                    opponent_id,
                    goalie.player_id,
                    x,
                    y,
                    related_shot_id=shot_id,
                    situation=situation,
                )
            continue

        if not pending or next_external_at > REGULATION_LENGTH_SECONDS:
            break
        at, _, _, kind, token = heapq.heappop(pending)
        current_time = at
        if kind == "FACEOFF":
            add_event(current_time, EventType.FACEOFF)
        elif kind == "PENALTY_END":
            penalty = active_penalties.pop(token, None)
            if penalty is not None:
                add_event(
                    current_time,
                    EventType.PENALTY_END,
                    penalty.team_id,
                    penalty.player_id,
                    situation=manpower(),
                )
        else:
            penalized_team_id = (
                home_team_id if rng.integers(0, 2) == 0 else away_team_id
            )
            team_skaters = (
                situation.home_skaters
                if penalized_team_id == home_team_id
                else situation.away_skaters
            )
            if team_skaters <= 3:
                continue
            eligible = tuple(
                player
                for player in players_by_team[penalized_team_id]
                if player.position is not PlayerPosition.GOALIE
                and all(
                    active.player_id != player.player_id
                    for active in active_penalties.values()
                )
            )
            if not eligible:
                continue
            player = eligible[int(rng.integers(0, len(eligible)))]
            penalty_event = add_event(
                current_time,
                EventType.PENALTY,
                penalized_team_id,
                player.player_id,
                situation=ManpowerSituation(
                    max(3, situation.home_skaters - (penalized_team_id == home_team_id)),
                    max(3, situation.away_skaters - (penalized_team_id == away_team_id)),
                ),
            )
            expires_at = current_time + 120
            active_penalties[penalty_event.event_id] = _Penalty(
                penalty_event.event_id, penalized_team_id, player.player_id, expires_at
            )
            schedule(expires_at, 0, "PENALTY_END", penalty_event.event_id)

    return (
        Match(
            match_id=match_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_score=home_score,
            away_score=away_score,
            simulation_seed=match_seed,
        ),
        tuple(events),
        tuple(shots),
        tuple(truth),
    )


def _choose_teams(n_teams: int, rng: np.random.Generator) -> tuple[str, str]:
    selected = rng.choice(n_teams, size=2, replace=False)
    return f"TEAM-{int(selected[0]) + 1:02d}", f"TEAM-{int(selected[1]) + 1:02d}"


def _players_by_team(players: tuple[Player, ...]) -> dict[str, tuple[Player, ...]]:
    result: dict[str, list[Player]] = {}
    for player in players:
        result.setdefault(player.team_id, []).append(player)
    return {team_id: tuple(team_players) for team_id, team_players in result.items()}


def _team_attack_factor(
    team: Team,
    opponent: Team,
    team_skaters: int,
    opponent_skaters: int,
    *,
    is_home: bool,
    home_advantage: float,
) -> float:
    strength = (0.75 + 0.5 * team.attack_rating) * (
        1.15 - 0.3 * opponent.defense_rating
    )
    home_factor = 1 + home_advantage if is_home else 1.0
    if team_skaters > opponent_skaters:
        manpower_factor = 1.35
    elif team_skaters < opponent_skaters:
        manpower_factor = 0.75
    else:
        manpower_factor = 1.0
    return strength * home_factor * manpower_factor


def _choose_shooting_team(
    home_team_id: str,
    away_team_id: str,
    home_rate: float,
    away_rate: float,
    rng: np.random.Generator,
) -> str:
    return (
        home_team_id
        if rng.random() < home_rate / (home_rate + away_rate)
        else away_team_id
    )


def _choose_shooter(
    players: tuple[Player, ...], rng: np.random.Generator
) -> Player:
    skaters = tuple(
        player for player in players if player.position is not PlayerPosition.GOALIE
    )
    weights = np.array(
        [
            0.65 / 12
            if player.position is PlayerPosition.FORWARD
            else 0.35 / 6
            for player in skaters
        ]
    )
    weights /= weights.sum()
    return skaters[int(rng.choice(len(skaters), p=weights))]


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _unit_interval(name: str, value: float) -> None:
    _finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
