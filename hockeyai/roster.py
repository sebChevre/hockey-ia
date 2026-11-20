"""Reproducible team and player creation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hockeyai.config import SimulationConfig
from hockeyai.models import Player, PlayerPosition, Team

_FACEOFF_SKILL_STREAM = 1


@dataclass(frozen=True, slots=True)
class Roster:
    teams: tuple[Team, ...]
    players: tuple[Player, ...]


def generate_roster(
    config: SimulationConfig, rng: np.random.Generator
) -> Roster:
    """Create stable team/player IDs and the specified 18 skaters + 2 goalies."""
    # Keep the new skill stream separate so existing roster draws stay stable.
    faceoff_rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([config.seed, _FACEOFF_SKILL_STREAM])
        )
    )
    rating_spread = 0.025 + 0.18 * config.team_imbalance / 100
    teams: list[Team] = []
    players: list[Player] = []

    for team_number in range(1, config.n_teams + 1):
        team_id = f"TEAM-{team_number:02d}"
        attack_rating = _rating(rng, rating_spread)
        defense_rating = _rating(rng, rating_spread)
        goalie_rating = _rating(rng, rating_spread)
        team = Team(
            team_id=team_id,
            team_name=f"Team {team_number:02d}",
            attack_rating=attack_rating,
            defense_rating=defense_rating,
            goalie_rating=goalie_rating,
        )
        teams.append(team)

        positions = (
            [PlayerPosition.FORWARD] * 12
            + [PlayerPosition.DEFENSE] * 6
            + [PlayerPosition.GOALIE] * config.goalies_per_team
        )
        for player_number, position in enumerate(positions, start=1):
            players.append(
                Player(
                    player_id=f"{team_id}-PLAYER-{player_number:02d}",
                    team_id=team_id,
                    position=position,
                    shooting_skill=_skill(rng, attack_rating, position),
                    passing_skill=_skill(rng, 0.5, position),
                    defensive_skill=_skill(rng, defense_rating, position),
                    faceoff_skill=float(
                        np.clip(faceoff_rng.normal(0.5, 0.14), 0, 1)
                    ),
                )
            )

    return Roster(teams=tuple(teams), players=tuple(players))


def _rating(rng: np.random.Generator, spread: float) -> float:
    return float(np.clip(rng.normal(0.5, spread), 0.05, 0.95))


def _skill(
    rng: np.random.Generator, team_level: float, position: PlayerPosition
) -> float:
    position_adjustment = {
        PlayerPosition.FORWARD: 0.08,
        PlayerPosition.DEFENSE: -0.04,
        PlayerPosition.GOALIE: -0.12,
    }[position]
    return float(np.clip(rng.normal(0.5 + 0.35 * (team_level - 0.5) + position_adjustment, 0.14), 0, 1))
