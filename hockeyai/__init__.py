"""Core contracts for the HockeyAI synthetic data generator."""

from hockeyai.config import GoalProbabilityCoefficients, SimulationConfig
from hockeyai.engine import goal_probability, simulate_dataset
from hockeyai.models import (
    AnomalyType,
    Event,
    EventType,
    ManpowerSituation,
    Match,
    PenaltyEndReason,
    PenaltyType,
    Player,
    PlayerPosition,
    Shot,
    ShotGroundTruth,
    ShotOutcome,
    SimulationDataset,
    Team,
)

__all__ = [
    "AnomalyType",
    "Event",
    "EventType",
    "GoalProbabilityCoefficients",
    "ManpowerSituation",
    "Match",
    "PenaltyEndReason",
    "PenaltyType",
    "Player",
    "PlayerPosition",
    "Shot",
    "ShotGroundTruth",
    "ShotOutcome",
    "SimulationConfig",
    "SimulationDataset",
    "Team",
    "goal_probability",
    "simulate_dataset",
]
