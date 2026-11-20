"""Rink, game-time, and shot-angle conventions used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math

RINK_LENGTH_M = 60.0
RINK_WIDTH_M = 30.0
REGULATION_PERIODS = 3
PERIOD_LENGTH_SECONDS = 20 * 60
REGULATION_LENGTH_SECONDS = REGULATION_PERIODS * PERIOD_LENGTH_SECONDS
TARGET_GOAL_X_M = RINK_LENGTH_M
TARGET_GOAL_Y_M = RINK_WIDTH_M / 2


@dataclass(frozen=True, slots=True)
class ShotGeometry:
    distance_m: float
    angle_deg: float


def period_for_elapsed_seconds(elapsed_seconds: int) -> int:
    """Map global time to period; exact boundaries start the next period."""
    _validate_elapsed_seconds(elapsed_seconds)
    if elapsed_seconds == REGULATION_LENGTH_SECONDS:
        return REGULATION_PERIODS
    return elapsed_seconds // PERIOD_LENGTH_SECONDS + 1


def period_elapsed_seconds(elapsed_seconds: int) -> int:
    """Return seconds elapsed within the derived period (end may be 1200)."""
    period = period_for_elapsed_seconds(elapsed_seconds)
    return elapsed_seconds - (period - 1) * PERIOD_LENGTH_SECONDS


def shot_geometry(x: float, y: float) -> ShotGeometry:
    """Measure distance and signed angle to the goal at (60, 15).

    The angle is measured from the attack direction (positive x axis); positive
    values are toward increasing y. At the exact goal center, angle is defined
    as zero because the direction vector has zero length.
    """
    _validate_position_coordinate("x", x, RINK_LENGTH_M)
    _validate_position_coordinate("y", y, RINK_WIDTH_M)
    dx = TARGET_GOAL_X_M - x
    dy = y - TARGET_GOAL_Y_M
    return ShotGeometry(
        distance_m=math.hypot(dx, dy),
        angle_deg=math.degrees(math.atan2(dy, dx)),
    )


def validate_position(x: float, y: float) -> None:
    """Raise ValueError unless a point lies within the rink, including edges."""
    _validate_position_coordinate("x", x, RINK_LENGTH_M)
    _validate_position_coordinate("y", y, RINK_WIDTH_M)


def _validate_elapsed_seconds(elapsed_seconds: int) -> None:
    if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, int):
        raise ValueError("elapsed_seconds must be an integer")
    if not 0 <= elapsed_seconds <= REGULATION_LENGTH_SECONDS:
        raise ValueError(
            f"elapsed_seconds must be between 0 and {REGULATION_LENGTH_SECONDS}"
        )


def _validate_position_coordinate(name: str, value: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
