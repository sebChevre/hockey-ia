from collections import Counter

import pytest

from hockeyai import (
    EventType,
    GoalProbabilityCoefficients,
    PenaltyEndReason,
    PenaltyType,
    PlayerPosition,
    SimulationConfig,
    goal_probability,
    simulate_dataset,
)
from hockeyai.validation import validate_dataset
from hockeyai.engine import _faceoff_win_probability


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
    assert all(0 <= player.faceoff_skill <= 1 for player in first.players)
    penalties = [
        event for event in first.events if event.event_type is EventType.PENALTY
    ]
    assert 0.9 < sum(
        event.penalty_type is PenaltyType.MINOR for event in penalties
    ) / len(penalties) < 0.99
    assert any(event.penalty_type is PenaltyType.MAJOR for event in penalties)
    penalties_by_id = {event.event_id: event for event in penalties}
    service_events = [
        event
        for event in first.events
        if event.event_type in (EventType.PENALTY_START, EventType.PENALTY_END)
    ]
    assert service_events
    assert all(
        penalties_by_id[event.related_penalty_event_id].team_id == event.team_id
        and penalties_by_id[event.related_penalty_event_id].player_id == event.player_id
        for event in service_events
    )
    assert any(
        event.event_type is EventType.PENALTY_END
        and event.penalty_end_reason is PenaltyEndReason.GOAL
        for event in first.events
    )
    players_by_id = {player.player_id: player for player in first.players}
    matches_by_id = {match.match_id: match for match in first.matches}
    goals = [event for event in first.events if event.event_type is EventType.GOAL]
    assists_per_goal = Counter(
        int(event.assist_1_player_id is not None)
        + int(event.assist_2_player_id is not None)
        for event in goals
    )
    assert set(assists_per_goal) == {0, 1, 2}
    assert assists_per_goal[0] / len(goals) == pytest.approx(0.15, abs=0.06)
    assert assists_per_goal[1] / len(goals) == pytest.approx(0.35, abs=0.06)
    assert assists_per_goal[2] / len(goals) == pytest.approx(0.50, abs=0.06)
    for goal in goals:
        assistant_ids = tuple(
            player_id
            for player_id in (
                goal.assist_1_player_id,
                goal.assist_2_player_id,
            )
            if player_id is not None
        )
        assert goal.player_id not in assistant_ids
        assert len(set(assistant_ids)) == len(assistant_ids)
        assert all(
            players_by_id[player_id].team_id == goal.team_id
            and players_by_id[player_id].position is not PlayerPosition.GOALIE
            for player_id in assistant_ids
        )
    faceoffs = [
        event for event in first.events if event.event_type is EventType.FACEOFF
    ]
    assert faceoffs
    for event in faceoffs:
        winner = players_by_id[event.faceoff_winner_player_id]
        loser = players_by_id[event.faceoff_loser_player_id]
        match = matches_by_id[event.match_id]
        assert winner.position is PlayerPosition.FORWARD
        assert loser.position is PlayerPosition.FORWARD
        assert winner.team_id != loser.team_id
        assert {winner.team_id, loser.team_id} == {
            match.home_team_id,
            match.away_team_id,
        }
    assert validate_dataset(first, config).is_valid


def test_faceoff_win_probability_is_weighted_by_relative_skill():
    assert _faceoff_win_probability(0.5, 0.5) == 0.5
    assert _faceoff_win_probability(0.8, 0.2) == pytest.approx(0.8)
    assert _faceoff_win_probability(0.2, 0.8) == pytest.approx(0.2)


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

    faceoff = next(
        event for event in dataset.events if event.event_type is EventType.FACEOFF
    )
    bad_faceoff = replace(faceoff, faceoff_winner_player_id="UNKNOWN-PLAYER")
    bad_faceoffs = tuple(
        bad_faceoff if event.event_id == faceoff.event_id else event
        for event in dataset.events
    )
    faceoff_report = validate_dataset(replace(dataset, events=bad_faceoffs))
    assert any("unknown winner" in error for error in faceoff_report.errors)


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
