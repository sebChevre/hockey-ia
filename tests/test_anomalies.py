from collections import Counter

from hockeyai.anomalies import inject_anomalies
from hockeyai import (
    AnomalyType,
    Event,
    EventType,
    ManpowerSituation,
    SimulationConfig,
    simulate_dataset,
)
from hockeyai.validation import validate_corrupted_events, validate_dataset


def _source_events(count=50):
    return tuple(
        Event(
            event_id=f"E-{index:03d}",
            match_id="M-001",
            event_sequence=index,
            elapsed_seconds=index,
            period=1,
            event_type=EventType.SHOT,
            team_id="TEAM-01",
            player_id="TEAM-01-P-01",
            x=40.0,
            y=15.0,
            related_shot_id=None,
            home_score_before=0,
            away_score_before=0,
            manpower_situation=ManpowerSituation(),
        )
        for index in range(count)
    )


def test_injection_is_reproducible_covers_types_and_preserves_input():
    source = _source_events()
    config = SimulationConfig(n_matches=100, anomaly_rate_percent=10)

    corrupted, log = inject_anomalies(source, config)
    repeated, repeated_log = inject_anomalies(source, config)

    assert corrupted == repeated
    assert log == repeated_log
    assert len(log) == 5
    assert {record.anomaly_type for record in log} == set(AnomalyType)
    assert len(corrupted) == len(source) + sum(
        record.anomaly_type is AnomalyType.DUPLICATE for record in log
    )
    assert all(record.injected_row_id for record in log if record.anomaly_type is AnomalyType.DUPLICATE)
    assert all(event.elapsed_seconds >= 0 for event in source)


def test_zero_anomaly_rate_returns_independent_clean_rows():
    source = _source_events(3)
    config = SimulationConfig(
        n_matches=100,
        anomaly_rate_percent=0,
        anomaly_types=frozenset(),
    )

    corrupted, log = inject_anomalies(source, config)

    assert not log
    assert len(corrupted) == len(source)
    assert corrupted[0]["event_id"] == source[0].event_id
    assert corrupted[0] is not corrupted[1]


def test_corrupted_event_validator_detects_each_injected_type():
    config = SimulationConfig(n_matches=100, anomaly_rate_percent=1)
    dataset = simulate_dataset(config)
    clean_events = dataset.events
    clean_report = validate_dataset(dataset, config)
    corrupted, log = inject_anomalies(
        dataset.events,
        config,
        known_team_ids={team.team_id for team in dataset.teams},
    )

    report = validate_corrupted_events(dataset, corrupted)

    assert clean_report.is_valid
    assert not report.is_valid
    assert dict(report.detected_by_type) == Counter(
        record.anomaly_type.value for record in log
    )
    assert dataset.events == clean_events


def test_corrupted_event_validator_reports_dropped_source_rows():
    dataset = simulate_dataset(SimulationConfig(n_matches=100, anomaly_rate_percent=0))

    report = validate_corrupted_events(dataset, ())

    assert not report.is_valid
    assert dict(report.detected_by_type)["missing"] == len(dataset.events)
