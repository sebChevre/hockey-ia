"""Deterministic CSV/JSON serialization and in-memory ZIP export."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
from statistics import mean
from typing import Iterable, Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from hockeyai.anomalies import AnomalyRecord, event_to_row
from hockeyai.config import SimulationConfig
from hockeyai.models import Shot, SimulationDataset
from hockeyai.validation import (
    CorruptedEventsReport,
    ValidationReport,
    validate_corrupted_events,
)

GENERATOR_VERSION = "0.4.0"
SCHEMA_VERSION = "1.3.0"

TEAMS_COLUMNS = (
    "team_id",
    "team_name",
    "attack_rating",
    "defense_rating",
    "goalie_rating",
)
PLAYERS_COLUMNS = (
    "player_id",
    "team_id",
    "position",
    "shooting_skill",
    "passing_skill",
    "defensive_skill",
    "faceoff_skill",
    "games_played",
    "ice_time_seconds",
)
MATCHES_COLUMNS = (
    "match_id",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "simulation_seed",
)
EVENTS_COLUMNS = (
    "event_id",
    "match_id",
    "event_sequence",
    "elapsed_seconds",
    "period",
    "event_type",
    "team_id",
    "player_id",
    "assist_1_player_id",
    "assist_2_player_id",
    "faceoff_winner_player_id",
    "faceoff_loser_player_id",
    "penalty_type",
    "penalty_duration_seconds",
    "related_penalty_event_id",
    "penalty_end_reason",
    "x",
    "y",
    "related_shot_id",
    "home_score_before",
    "away_score_before",
    "manpower_situation",
)
SHOTS_COLUMNS = (
    "shot_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "elapsed_seconds",
    "x",
    "y",
    "distance_m",
    "angle_deg",
    "shot_type",
    "manpower_situation",
    "shooting_skill",
    "goalie_skill",
    "score_difference_before",
    "is_goal",
    "shot_outcome",
)
GROUND_TRUTH_COLUMNS = ("shot_id", "true_goal_probability")
ANOMALY_COLUMNS = (
    "source_event_id",
    "anomaly_type",
    "field",
    "original_value",
    "corrupted_value",
    "injected_row_id",
)

DATA_DICTIONARY = """# HockeyAI data dictionary

All data is synthetic. Coordinates use the 60 m × 30 m rink in a common
offensive frame; the target goal center is `(60, 15)`. Time is elapsed
regulation seconds in `[0, 3600]`; period boundaries are 0, 1200, and 2400
seconds. Empty CSV cells represent nulls.

## `teams.csv`

| Column | Type / unit | Nullable | Meaning |
|---|---|---|---|
| `team_id` | string | no | Stable team key. |
| `team_name` | string | no | Synthetic team name. |
| `attack_rating` | float, [0,1] | no | Latent synthetic attack strength. |
| `defense_rating` | float, [0,1] | no | Latent synthetic defense strength. |
| `goalie_rating` | float, [0,1] | no | Team-level synthetic goalie skill. |

## `players.csv`

| Column | Type / unit | Nullable | Meaning |
|---|---|---|---|
| `player_id` | string | no | Stable player key. |
| `team_id` | string | no | Foreign key to `teams.team_id`. |
| `position` | enum F/D/G | no | Forward, defense, or goalie. |
| `shooting_skill` | float, [0,1] | no | Synthetic shooting skill. |
| `passing_skill` | float, [0,1] | no | Synthetic passing skill. |
| `defensive_skill` | float, [0,1] | no | Synthetic defensive skill. |
| `faceoff_skill` | float, [0,1] | no | Synthetic skill used to weight faceoff outcomes. |
| `games_played` | integer, games | no | Team games for skaters; goalie starts for goalies. |
| `ice_time_seconds` | integer, seconds | no | Estimated exposure; see simulation conventions. |

## `matches.csv`

| Column | Type / unit | Nullable | Meaning |
|---|---|---|---|
| `match_id` | string | no | Stable match key. |
| `home_team_id`, `away_team_id` | string | no | Team foreign keys. |
| `home_score`, `away_score` | integer, goals | no | Final score from `GOAL` events. |
| `simulation_seed` | integer | no | Per-match PCG64 seed. |

## `events.csv` and `events_corrupted.csv`

| Column | Type / unit | Nullable | Meaning |
|---|---|---|---|
| `event_id` | string | no | Event key; duplicates are deliberate in corrupted data. |
| `match_id` | string | no | Match foreign key. |
| `event_sequence` | integer | no | Deterministic within-match sequence. |
| `elapsed_seconds` | integer, seconds | no | Global regulation elapsed time. |
| `period` | integer, 1–3 | no | Period derived from elapsed time. |
| `event_type` | enum | no | FACEOFF, SHOT, SAVE, GOAL, PENALTY, PENALTY_START, PENALTY_END. |
| `team_id`, `player_id` | string | event-dependent | Related team/player foreign keys. |
| `assist_1_player_id`, `assist_2_player_id` | string | GOAL only | Zero, one, or two distinct skaters on the scoring team; the scorer is not an assist. |
| `faceoff_winner_player_id`, `faceoff_loser_player_id` | string | FACEOFF only | Opposing forwards who won and lost the engagement. |
| `penalty_type` | enum | PENALTY only | `MINEURE` (120 s) or `MAJEURE` (300 s). |
| `penalty_duration_seconds` | integer, seconds | PENALTY only | Service duration; the clock starts at `PENALTY_START`. |
| `related_penalty_event_id` | string | PENALTY_START/END only | Source `PENALTY` event key. |
| `penalty_end_reason` | enum | PENALTY_END only | `TIME_EXPIRED` or `GOAL` (a qualifying power-play goal). |
| `x`, `y` | float, metres | event-dependent | Rink coordinates. |
| `related_shot_id` | string | outcome only | Shot key for GOAL or SAVE. |
| `home_score_before`, `away_score_before` | integer, goals | no | Score immediately before this event. |
| `manpower_situation` | string, skaters | no | Home-away skater counts (for example `5v4`). |

## `shots.csv`

| Column | Type / unit | Nullable | Meaning |
|---|---|---|---|
| `shot_id` | string | no | Stable shot key. |
| `event_id`, `match_id`, `team_id`, `player_id` | string | no | Event, match, team, and shooter keys. |
| `elapsed_seconds` | integer, seconds | no | Time of the shot. |
| `x`, `y` | float, metres | no | Shot location in the offensive frame. |
| `distance_m` | float, metres | no | Distance to target goal center. |
| `angle_deg` | float, degrees | no | Signed angle relative to attack direction. |
| `shot_type` | enum | no | Synthetic shot type. |
| `manpower_situation` | string, skaters | no | Home-away counts at shot time. |
| `shooting_skill`, `goalie_skill` | float, [0,1] | no | Skills used by the generator. |
| `score_difference_before` | integer, goals | no | Shooter-team score minus opponent score before shot. |
| `is_goal` | boolean target | no | Whether the shot became a goal. |
| `shot_outcome` | enum target | no | GOAL, SAVED, or MISSED; never a model feature. |

## `shot_ground_truth.csv` (reference only)

| Column | Type / unit | Nullable | Meaning |
|---|---|---|---|
| `shot_id` | string | no | Shot key. |
| `true_goal_probability` | float, [0,1] | no | Hidden probability used by the simulator; exclude from training. |

## `anomaly_log.csv` (evaluation only)

| Column | Type / unit | Nullable | Meaning |
|---|---|---|---|
| `source_event_id` | string | no | Original event key. |
| `anomaly_type` | enum | no | Applied corruption type. |
| `field` | string | no | Corrupted field. |
| `original_value`, `corrupted_value` | string representation | yes | Before/after values. |
| `injected_row_id` | string | duplicate only | Technical identifier for an injected duplicate row. |
"""


def build_export_zip(
    dataset: SimulationDataset,
    corrupted_events: Sequence[Mapping[str, object]],
    anomaly_log: Sequence[AnomalyRecord],
    config: SimulationConfig,
    clean_report: ValidationReport,
) -> bytes:
    """Build the complete export in memory; no server-side files are written."""
    if not isinstance(dataset, SimulationDataset):
        raise TypeError("dataset must be a SimulationDataset")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if not clean_report.is_valid:
        raise ValueError("refusing to export a clean dataset with validation errors")

    corrupted_report = validate_corrupted_events(dataset, corrupted_events)
    source_count = len(dataset.events)
    requested_rows = (
        min(
            source_count,
            max(1, int(source_count * config.anomaly_rate_percent / 100 + 0.5)),
        )
        if source_count and config.anomaly_rate_percent > 0
        else 0
    )
    injected_counts = Counter(record.anomaly_type.value for record in anomaly_log)
    quality_data = _quality_report(
        dataset,
        clean_report,
        corrupted_report,
        config,
        requested_rows,
        len(corrupted_events),
        anomaly_log,
        injected_counts,
    )

    data_files = {
        "teams.csv": _csv_bytes(
            TEAMS_COLUMNS, (asdict(row) for row in dataset.teams)
        ),
        "players.csv": _csv_bytes(
            PLAYERS_COLUMNS, (asdict(row) for row in dataset.players)
        ),
        "matches.csv": _csv_bytes(
            MATCHES_COLUMNS, (asdict(row) for row in dataset.matches)
        ),
        "events.csv": _csv_bytes(
            EVENTS_COLUMNS, (event_to_row(row) for row in dataset.events)
        ),
        "events_corrupted.csv": _csv_bytes(EVENTS_COLUMNS, corrupted_events),
        "shots.csv": _csv_bytes(
            SHOTS_COLUMNS, (_shot_to_row(row) for row in dataset.shots)
        ),
        "shot_ground_truth.csv": _csv_bytes(
            GROUND_TRUTH_COLUMNS,
            (asdict(row) for row in dataset.shot_ground_truth),
        ),
        "anomaly_log.csv": _csv_bytes(
            ANOMALY_COLUMNS, (_anomaly_to_row(row) for row in anomaly_log)
        ),
        "config.json": config_json_bytes(config),
        "quality_report.json": _json_bytes(quality_data),
        "data_dictionary.md": DATA_DICTIONARY.encode("utf-8"),
    }
    metadata_data = _metadata(
        dataset,
        config,
        data_files,
        len(corrupted_events),
        len(anomaly_log),
    )
    archive_files = dict(data_files)
    archive_files["metadata.json"] = _json_bytes(metadata_data)
    return _zip_bytes(archive_files)


def config_json_bytes(config: SimulationConfig) -> bytes:
    """Serialize a complete config using the same stable schema as the ZIP."""
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    config_data = asdict(config)
    config_data["anomaly_types"] = sorted(item.value for item in config.anomaly_types)
    return _json_bytes(config_data)


def _quality_report(
    dataset: SimulationDataset,
    clean_report: ValidationReport,
    corrupted_report: CorruptedEventsReport,
    config: SimulationConfig,
    requested_rows: int,
    corrupted_event_count: int,
    anomaly_log: Sequence[AnomalyRecord],
    injected_counts: Counter,
) -> dict[str, object]:
    statistics = clean_report.match_statistics
    detected_counts = dict(corrupted_report.detected_by_type)
    return {
        "anomalies": {
            "requested_rate_percent": config.anomaly_rate_percent,
            "requested_types": sorted(item.value for item in config.anomaly_types),
            "requested_source_rows": requested_rows,
            "selected_source_rows": len({row.source_event_id for row in anomaly_log}),
            "injected_rows": len(anomaly_log),
            "injected_by_type": dict(sorted(injected_counts.items())),
            "detected_by_type": dict(sorted(detected_counts.items())),
        },
        "clean_data": {
            "valid": clean_report.is_valid,
            "errors": list(clean_report.errors),
            "warnings": list(clean_report.warnings),
        },
        "corrupted_data": {
            "valid": corrupted_report.is_valid,
            "errors": list(corrupted_report.errors),
        },
        "distributions": {
            "matches": len(statistics),
            "shots_per_match": _distribution(row.shots for row in statistics),
            "goals_per_match": _distribution(row.goals for row in statistics),
            "penalties_per_match": _distribution(row.penalties for row in statistics),
            "by_match": [
                {
                    "match_id": row.match_id,
                    "shots": row.shots,
                    "goals": row.goals,
                    "penalties": row.penalties,
                }
                for row in statistics
            ],
        },
        "table_rows": {
            "teams": len(dataset.teams),
            "players": len(dataset.players),
            "matches": len(dataset.matches),
            "events_clean": len(dataset.events),
            "events_corrupted": corrupted_event_count,
            "shots": len(dataset.shots),
            "shot_ground_truth": len(dataset.shot_ground_truth),
            "anomaly_log": len(anomaly_log),
        },
    }


def _distribution(values: Iterable[int]) -> dict[str, float | int]:
    items = list(values)
    if not items:
        return {"mean": 0.0, "min": 0, "max": 0}
    return {
        "mean": round(mean(items), 3),
        "min": min(items),
        "max": max(items),
    }


def _metadata(
    dataset: SimulationDataset,
    config: SimulationConfig,
    data_files: Mapping[str, bytes],
    corrupted_event_count: int,
    anomaly_log_count: int,
) -> dict[str, object]:
    return {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": config.seed,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "volumes": {
            "teams": len(dataset.teams),
            "players": len(dataset.players),
            "matches": len(dataset.matches),
            "events_clean": len(dataset.events),
            "events_corrupted": corrupted_event_count,
            "shots": len(dataset.shots),
            "shot_ground_truth": len(dataset.shot_ground_truth),
            "anomaly_log": anomaly_log_count,
        },
        "coordinate_convention": {
            "rink_length_m": 60,
            "rink_width_m": 30,
            "frame": "common offensive frame",
            "target_goal_center_m": [60, 15],
        },
        "time_convention": {
            "unit": "seconds elapsed in regulation",
            "range": [0, 3600],
            "period_boundaries": [0, 1200, 2400, 3600],
        },
        "sha256": {
            filename: hashlib.sha256(payload).hexdigest()
            for filename, payload in sorted(data_files.items())
        },
        "sha256_scope": "All archive files except metadata.json (self-reference).",
    }


def _anomaly_to_row(record: AnomalyRecord) -> dict[str, object]:
    return {
        "source_event_id": record.source_event_id,
        "anomaly_type": record.anomaly_type.value,
        "field": record.field,
        "original_value": record.original_value,
        "corrupted_value": record.corrupted_value,
        "injected_row_id": record.injected_row_id,
    }


def _shot_to_row(shot: Shot) -> dict[str, object]:
    row = asdict(shot)
    row["manpower_situation"] = shot.manpower_situation.code
    return row


def _csv_bytes(
    columns: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> bytes:
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_csv_value(row.get(column)) for column in columns])
    return stream.getvalue().encode("utf-8")


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return value.value
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _zip_bytes(files: Mapping[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for filename, payload in sorted(files.items()):
            info = ZipInfo(filename, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()
