"""Reproducible, logged corruption of a copy of the clean event table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Sequence

import numpy as np

from hockeyai.config import SimulationConfig
from hockeyai.models import AnomalyType, Event


@dataclass(frozen=True, slots=True)
class AnomalyRecord:
    source_event_id: str
    anomaly_type: AnomalyType
    field: str
    original_value: object
    corrupted_value: object
    injected_row_id: str | None = None


def inject_anomalies(
    events: Sequence[Event],
    config: SimulationConfig,
    *,
    known_team_ids: Collection[str] | None = None,
) -> tuple[tuple[dict[str, object], ...], tuple[AnomalyRecord, ...]]:
    """Corrupt selected source rows while leaving the supplied events untouched.

    The selected count is the nearest integer to the configured percentage of
    source rows (at least one row when the rate is positive and data exists).
    A source row receives at most one anomaly.
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    rows = [event_to_row(event) for event in events]
    if not rows or config.anomaly_rate_percent == 0:
        return tuple(rows), ()

    requested_types = tuple(sorted(config.anomaly_types, key=lambda item: item.value))
    eligible_by_type = {
        anomaly_type: [
            index
            for index, row in enumerate(rows)
            if _supports(row, anomaly_type)
        ]
        for anomaly_type in requested_types
    }
    eligible_rows = sorted(
        {index for indices in eligible_by_type.values() for index in indices}
    )
    if not eligible_rows:
        raise ValueError("no event row supports the configured anomaly types")

    target_count = min(
        len(eligible_rows),
        max(1, int(len(rows) * config.anomaly_rate_percent / 100 + 0.5)),
    )
    rng = np.random.Generator(np.random.PCG64(config.seed))
    assignments: dict[int, AnomalyType] = {}
    selected: set[int] = set()

    # Reserve distinct source rows so every requested type is represented
    # whenever there are enough target rows and eligible source rows.
    type_order = sorted(
        requested_types,
        key=lambda item: (len(eligible_by_type[item]), item.value),
    )
    for anomaly_type in type_order:
        if len(assignments) >= target_count:
            break
        available = [
            index
            for index in eligible_by_type[anomaly_type]
            if index not in selected
        ]
        if available:
            index = int(rng.choice(available))
            assignments[index] = anomaly_type
            selected.add(index)

    remaining = [index for index in eligible_rows if index not in selected]
    additional_count = target_count - len(selected)
    if additional_count:
        selected.update(
            int(index)
            for index in rng.choice(remaining, size=additional_count, replace=False)
        )

    counts = {anomaly_type: 0 for anomaly_type in requested_types}
    for index in sorted(selected):
        if index in assignments:
            counts[assignments[index]] += 1
            continue
        applicable = [
            anomaly_type
            for anomaly_type in requested_types
            if _supports(rows[index], anomaly_type)
        ]
        least_used = min(counts[item] for item in applicable)
        balanced = [item for item in applicable if counts[item] == least_used]
        anomaly_type = balanced[int(rng.integers(0, len(balanced)))]
        assignments[index] = anomaly_type
        counts[anomaly_type] += 1

    team_ids = set(known_team_ids or ())
    team_ids.update(
        row["team_id"] for row in rows if isinstance(row.get("team_id"), str)
    )
    unknown_team_id = f"UNKNOWN-TEAM-{config.seed:08X}"
    suffix = 1
    while unknown_team_id in team_ids:
        unknown_team_id = f"UNKNOWN-TEAM-{config.seed:08X}-{suffix}"
        suffix += 1

    corrupted: list[dict[str, object]] = []
    log: list[AnomalyRecord] = []
    duplicate_number = 0
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        anomaly_type = assignments.get(index)
        if anomaly_type is not None:
            source_event_id = str(source_row["event_id"])
            field, original, changed = _corrupt(
                row, anomaly_type, rng, unknown_team_id
            )
            injected_row_id = None
            if anomaly_type is AnomalyType.DUPLICATE:
                duplicate_number += 1
                injected_row_id = f"INJECTED-ROW-{duplicate_number:06d}"
            log.append(
                AnomalyRecord(
                    source_event_id=source_event_id,
                    anomaly_type=anomaly_type,
                    field=field,
                    original_value=original,
                    corrupted_value=changed,
                    injected_row_id=injected_row_id,
                )
            )
        corrupted.append(row)
        if anomaly_type is AnomalyType.DUPLICATE:
            corrupted.append(dict(row))

    return tuple(corrupted), tuple(log)


def event_to_row(event: Event) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "match_id": event.match_id,
        "event_sequence": event.event_sequence,
        "elapsed_seconds": event.elapsed_seconds,
        "period": event.period,
        "event_type": event.event_type.value,
        "team_id": event.team_id,
        "player_id": event.player_id,
        "faceoff_winner_player_id": event.faceoff_winner_player_id,
        "faceoff_loser_player_id": event.faceoff_loser_player_id,
        "penalty_type": event.penalty_type.value if event.penalty_type else None,
        "penalty_duration_seconds": event.penalty_duration_seconds,
        "related_penalty_event_id": event.related_penalty_event_id,
        "penalty_end_reason": (
            event.penalty_end_reason.value if event.penalty_end_reason else None
        ),
        "x": event.x,
        "y": event.y,
        "related_shot_id": event.related_shot_id,
        "home_score_before": event.home_score_before,
        "away_score_before": event.away_score_before,
        "manpower_situation": event.manpower_situation.code,
    }


def _supports(row: dict[str, object], anomaly_type: AnomalyType) -> bool:
    if anomaly_type in (AnomalyType.DUPLICATE, AnomalyType.INVALID_TIME):
        return True
    if anomaly_type is AnomalyType.MISSING:
        return any(row.get(field) is not None for field in ("team_id", "player_id", "x"))
    if anomaly_type is AnomalyType.UNKNOWN_TEAM:
        return row.get("team_id") is not None
    if anomaly_type is AnomalyType.INVALID_POSITION:
        return row.get("x") is not None and row.get("y") is not None
    return False


def _corrupt(
    row: dict[str, object],
    anomaly_type: AnomalyType,
    rng: np.random.Generator,
    unknown_team_id: str,
) -> tuple[str, object, object]:
    if anomaly_type is AnomalyType.MISSING:
        applicable = [
            field
            for field in ("team_id", "player_id", "x")
            if row.get(field) is not None
        ]
        field = applicable[int(rng.integers(0, len(applicable)))]
        original = row[field]
        row[field] = None
        return field, original, None
    if anomaly_type is AnomalyType.DUPLICATE:
        event_id = row["event_id"]
        return "event_id", event_id, event_id
    if anomaly_type is AnomalyType.INVALID_TIME:
        original = row["elapsed_seconds"]
        corrupted = -1 if rng.integers(0, 2) == 0 else 3601
        row["elapsed_seconds"] = corrupted
        return "elapsed_seconds", original, corrupted
    if anomaly_type is AnomalyType.UNKNOWN_TEAM:
        original = row["team_id"]
        row["team_id"] = unknown_team_id
        return "team_id", original, unknown_team_id
    if anomaly_type is AnomalyType.INVALID_POSITION:
        original = row["x"]
        row["x"] = 60.1
        return "x", original, 60.1
    raise ValueError(f"unsupported anomaly type: {anomaly_type}")
