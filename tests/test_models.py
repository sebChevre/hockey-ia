from dataclasses import fields

import pytest

from hockeyai import (
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
from hockeyai.geometry import shot_geometry


def test_team_and_player_ratings_are_bounded():
    with pytest.raises(ValueError, match="between 0 and 1"):
        Team("T1", "Team 1", attack_rating=1.1, defense_rating=0.5, goalie_rating=0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        Player("P1", "T1", PlayerPosition.FORWARD, 0.5, -0.1, 0.5)


def test_manpower_is_in_home_away_order_and_never_below_three():
    assert ManpowerSituation(5, 4).code == "5v4"
    with pytest.raises(ValueError):
        ManpowerSituation(2, 5)


def test_match_rejects_same_team_and_negative_score():
    with pytest.raises(ValueError, match="must differ"):
        Match("M1", "T1", "T1", 0, 0, 42)
    with pytest.raises(ValueError, match="non-negative"):
        Match("M1", "T1", "T2", -1, 0, 42)


def test_event_checks_period_and_coordinate_pair():
    with pytest.raises(ValueError, match="period"):
        Event(
            "E1", "M1", 0, 1200, 1, EventType.FACEOFF, "T1", None,
            None, None, None, 0, 0, ManpowerSituation(),
        )
    with pytest.raises(ValueError, match="both be set"):
        Event(
            "E1", "M1", 0, 0, 1, EventType.FACEOFF, "T1", None,
            12, None, None, 0, 0, ManpowerSituation(),
        )
    with pytest.raises(ValueError, match="period"):
        Event(
            "E1", "M1", 0, 0, True, EventType.FACEOFF, "T1", None,
            None, None, None, 0, 0, ManpowerSituation(),
        )


def test_shot_outcome_agrees_with_label_and_geometry():
    geometry = shot_geometry(50, 15)
    shot = Shot(
        "S1", "E1", "M1", "T1", "P1", 10, 50, 15,
        geometry.distance_m, geometry.angle_deg, "WRIST",
        ManpowerSituation(), 0.7, 0.5, 0, True, ShotOutcome.GOAL,
    )
    assert shot.is_goal

    with pytest.raises(ValueError, match="does not match"):
        Shot(
            "S2", "E2", "M1", "T1", "P1", 10, 50, 15,
            1, geometry.angle_deg, "WRIST", ManpowerSituation(), 0.7, 0.5,
            0, False, ShotOutcome.MISSED,
        )
    with pytest.raises(ValueError, match="must agree"):
        Shot(
            "S3", "E3", "M1", "T1", "P1", 10, 50, 15,
            geometry.distance_m, geometry.angle_deg, "WRIST",
            ManpowerSituation(), 0.7, 0.5, 0, True, ShotOutcome.SAVED,
        )


def test_ground_truth_is_bounded_and_separate_from_shots():
    assert ShotGroundTruth("S1", 0.25).true_goal_probability == 0.25
    with pytest.raises(ValueError):
        ShotGroundTruth("S1", 1.01)
    assert "true_goal_probability" not in {item.name for item in fields(Shot)}


def test_dataset_tables_are_immutable_tuples_of_their_row_type():
    dataset = SimulationDataset((), (), (), (), ())
    assert dataset.shot_ground_truth == ()

    with pytest.raises(ValueError, match="teams must be a tuple"):
        SimulationDataset([], (), (), (), ())
    with pytest.raises(ValueError, match="players must be a tuple"):
        SimulationDataset((), (Team("T1", "Team 1", 0.5, 0.5, 0.5),), (), (), ())
