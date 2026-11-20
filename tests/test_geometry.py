import pytest

from hockeyai.geometry import (
    PERIOD_LENGTH_SECONDS,
    REGULATION_LENGTH_SECONDS,
    period_elapsed_seconds,
    period_for_elapsed_seconds,
    shot_geometry,
    validate_position,
)


@pytest.mark.parametrize(
    ("elapsed", "period", "within_period"),
    [
        (0, 1, 0),
        (PERIOD_LENGTH_SECONDS - 1, 1, PERIOD_LENGTH_SECONDS - 1),
        (PERIOD_LENGTH_SECONDS, 2, 0),
        (2 * PERIOD_LENGTH_SECONDS - 1, 2, PERIOD_LENGTH_SECONDS - 1),
        (2 * PERIOD_LENGTH_SECONDS, 3, 0),
        (REGULATION_LENGTH_SECONDS - 1, 3, PERIOD_LENGTH_SECONDS - 1),
        (REGULATION_LENGTH_SECONDS, 3, PERIOD_LENGTH_SECONDS),
    ],
)
def test_period_boundaries(elapsed, period, within_period):
    assert period_for_elapsed_seconds(elapsed) == period
    assert period_elapsed_seconds(elapsed) == within_period


@pytest.mark.parametrize("elapsed", [-1, REGULATION_LENGTH_SECONDS + 1, 1.5, True])
def test_invalid_elapsed_time_is_rejected(elapsed):
    with pytest.raises(ValueError):
        period_for_elapsed_seconds(elapsed)


def test_shot_geometry_at_center_and_on_both_sides_of_goal():
    center = shot_geometry(50, 15)
    assert center.distance_m == 10
    assert center.angle_deg == 0
    assert shot_geometry(50, 20).angle_deg > 0
    assert shot_geometry(50, 10).angle_deg < 0


@pytest.mark.parametrize("point", [(-0.1, 15), (60.1, 15), (30, -0.1), (30, 30.1)])
def test_position_outside_rink_is_rejected(point):
    with pytest.raises(ValueError):
        validate_position(*point)


def test_rink_edges_are_inclusive():
    validate_position(0, 0)
    validate_position(60, 30)
