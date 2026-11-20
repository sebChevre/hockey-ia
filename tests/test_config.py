import pytest

from hockeyai import (
    AnomalyType,
    GoalProbabilityCoefficients,
    SimulationConfig,
)


def test_default_config_and_path_safe_scenario_slug():
    config = SimulationConfig(scenario_name="Playoffs / Test")

    assert config.n_matches == 1000
    assert config.players_per_team == 18
    assert config.goalies_per_team == 2
    assert config.scenario_slug == "playoffs-test"
    assert config.anomaly_types == frozenset(AnomalyType)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_matches", 99),
        ("n_matches", 10_001),
        ("n_teams", 5),
        ("n_teams", 18),
        ("seed", -1),
        ("seed", 2**32),
        ("offensive_intensity", 101),
        ("team_imbalance", -1),
        ("penalties_per_match", 15.1),
        ("home_advantage", 0.21),
        ("anomaly_rate_percent", 20.1),
    ],
)
def test_config_rejects_values_outside_spec(field, value):
    with pytest.raises(ValueError):
        SimulationConfig(**{field: value})


def test_config_requires_anomaly_type_when_corruption_is_enabled():
    with pytest.raises(ValueError, match="must not be empty"):
        SimulationConfig(anomaly_types=frozenset())

    config = SimulationConfig(anomaly_rate_percent=0, anomaly_types=frozenset())
    assert config.anomaly_types == frozenset()


def test_config_rejects_non_finite_model_coefficients():
    with pytest.raises(ValueError, match="finite"):
        GoalProbabilityCoefficients(beta_distance=float("nan"))


def test_scenario_name_is_bounded_and_not_blank():
    with pytest.raises(ValueError):
        SimulationConfig(scenario_name=" ")
    with pytest.raises(ValueError):
        SimulationConfig(scenario_name="x" * 51)
