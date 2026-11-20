from collections import Counter

import pytest

from hockeyai import (
    EventType,
    GoalProbabilityCoefficients,
    PlayerPosition,
    SimulationConfig,
    goal_probability,
    simulate_dataset,
)
from hockeyai.validation import validate_dataset


def test_goal_probability_is_bounded_and_monotonic_for_expected_inputs():
    coefficients = GoalProbabilityCoefficients()
    close_chance = goal_probability(5, 0, 0.7, 0.4, False, coefficients)
    far_chance = goal_probability(40, 0, 0.7, 0.4, False, coefficients)
    powerplay_chance = goal_probability(5, 0, 0.7, 0.4, True, coefficients)

    assert 0.005 <= far_chance < close_chance <= 0.70
    assert powerplay_chance > close_chance
    assert goal_probability(1000, 180, 0, 1, False, coefficients) == 0.005


@pytest.mark.parametrize(
    ("distance", "angle", "shooter_skill", "goalie_skill", "powerplay"),
    [
        (-1, 0, 0.5, 0.5, False),
        (1, float("nan"), 0.5, 0.5, False),
        (1, 0, -0.1, 0.5, False),
        (1, 0, 0.5, 1.1, False),
        (1, 0, 0.5, 0.5, 1),
    ],
)
def test_goal_probability_rejects_invalid_inputs(
    distance, angle, shooter_skill, goalie_skill, powerplay
):
    with pytest.raises(ValueError):
        goal_probability(distance, angle, shooter_skill, goalie_skill, powerplay)


def test_roster_and_event_tables_are_consistent_and_reproducible():
    config = SimulationConfig(n_matches=100, seed=42)
    first = simulate_dataset(config)
    second = simulate_dataset(config)

    assert first == second
    assert len(first.matches) == 100
    assert len(first.teams) == config.n_teams
    assert len(first.players) == config.n_teams * 20
    assert all(
        sum(player.position is PlayerPosition.FORWARD for player in first.players
            if player.team_id == team.team_id) == 12
        for team in first.teams
    )
    assert all(player.games_played > 0 for player in first.players)
    assert all(player.ice_time_seconds > 0 for player in first.players)
    assert validate_dataset(first, config).is_valid


def test_different_seed_changes_simulation_and_progress_reports_each_match():
    progress = []
    first = simulate_dataset(
        SimulationConfig(n_matches=100, seed=42),
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    changed = simulate_dataset(SimulationConfig(n_matches=100, seed=43))

    assert first != changed
    assert progress[0] == (1, 100)
    assert progress[-1] == (100, 100)
    assert len(progress) == 100


def test_shot_outcomes_scores_and_stable_player_reuse():
    dataset = simulate_dataset(SimulationConfig(n_matches=100, seed=7))
    events_by_shot = Counter(
        event.related_shot_id
        for event in dataset.events
        if event.event_type in (EventType.GOAL, EventType.SAVE)
    )
    shots_by_player = Counter(shot.player_id for shot in dataset.shots)

    assert all(
        events_by_shot[shot.shot_id]
        == (1 if shot.is_goal or shot.shot_outcome.value == "SAVED" else 0)
        for shot in dataset.shots
    )
    assert any(count > 1 for count in shots_by_player.values())
    assert len(dataset.shot_ground_truth) == len(dataset.shots)
    assert "true_goal_probability" not in dataset.shots[0].__dataclass_fields__


def test_1000_match_integration_smoke():
    dataset = simulate_dataset(SimulationConfig(n_matches=1000, seed=42))

    assert len(dataset.matches) == 1000
    assert validate_dataset(dataset).is_valid


def test_validation_finds_corrupted_cross_table_link():
    from dataclasses import replace

    dataset = simulate_dataset(SimulationConfig(n_matches=100, seed=11))
    bad_event = replace(dataset.events[0], team_id="UNKNOWN")
    corrupted = replace(dataset, events=(bad_event,) + dataset.events[1:])

    report = validate_dataset(corrupted)

    assert not report.is_valid
    assert any("unknown team" in error for error in report.errors)


def test_validation_warns_when_documented_baseline_calibration_is_missed():
    from hockeyai.validation import MatchStatistics, _calibration_warnings

    config = SimulationConfig(n_matches=100)
    stats = tuple(
        MatchStatistics(f"M-{index:03d}", shots=80, goals=10, penalties=9)
        for index in range(100)
    )

    warnings = _calibration_warnings(stats, config)

    assert any("shots per match" in warning for warning in warnings)
    assert any("goals per match" in warning for warning in warnings)
    assert any("penalties per match" in warning for warning in warnings)
