import csv
import hashlib
from io import BytesIO, StringIO
import json
from zipfile import ZipFile

import pandas as pd
import pytest

from hockeyai.anomalies import inject_anomalies
from hockeyai.config import SimulationConfig
from hockeyai.engine import simulate_dataset
from hockeyai.export import build_export_zip
from hockeyai.validation import validate_dataset


def test_zip_contains_required_readable_tables_metadata_and_quality_report():
    config = SimulationConfig(n_matches=100, seed=9, anomaly_rate_percent=0.1)
    dataset = simulate_dataset(config)
    clean_report = validate_dataset(dataset, config)
    corrupted, log = inject_anomalies(
        dataset.events,
        config,
        known_team_ids={team.team_id for team in dataset.teams},
    )

    archive_bytes = build_export_zip(
        dataset, corrupted, log, config, clean_report
    )
    repeated_archive_bytes = build_export_zip(
        dataset, corrupted, log, config, clean_report
    )

    with ZipFile(BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        assert {
            "teams.csv",
            "players.csv",
            "matches.csv",
            "events.csv",
            "events_corrupted.csv",
            "shots.csv",
            "shot_ground_truth.csv",
            "anomaly_log.csv",
            "config.json",
            "metadata.json",
            "quality_report.json",
            "data_dictionary.md",
        } <= names

        for filename in names:
            if filename.endswith(".csv"):
                rows = list(
                    csv.reader(StringIO(archive.read(filename).decode("utf-8")))
                )
                assert rows
            elif filename.endswith(".json"):
                json.loads(archive.read(filename))

        shots = list(csv.DictReader(StringIO(archive.read("shots.csv").decode())))
        assert len(shots) == len(dataset.shots)
        assert "true_goal_probability" not in shots[0]
        assert shots[0]["is_goal"] in ("True", "False")
        assert shots[0]["manpower_situation"] in ("5v5", "5v4", "4v5", "4v4")
        assert len(list(csv.DictReader(StringIO(archive.read("matches.csv").decode())))) == 100
        pandas_matches = pd.read_csv(BytesIO(archive.read("matches.csv")))
        pandas_shots = pd.read_csv(BytesIO(archive.read("shots.csv")))
        assert len(pandas_matches) == 100
        assert pandas_shots["is_goal"].dtype == bool

        config_data = json.loads(archive.read("config.json"))
        metadata = json.loads(archive.read("metadata.json"))
        quality = json.loads(archive.read("quality_report.json"))
        assert config_data["seed"] == 9
        assert metadata["volumes"]["events_clean"] == len(dataset.events)
        assert metadata["volumes"]["events_corrupted"] == len(corrupted)
        assert metadata["sha256"]["teams.csv"] == hashlib.sha256(
            archive.read("teams.csv")
        ).hexdigest()
        assert quality["clean_data"]["valid"]
        assert not quality["corrupted_data"]["valid"]
        assert quality["anomalies"]["selected_source_rows"] == len(log)
        assert quality["distributions"]["matches"] == 100
        assert quality["table_rows"]["events_corrupted"] == len(corrupted)
    with ZipFile(BytesIO(archive_bytes)) as first_archive, ZipFile(
        BytesIO(repeated_archive_bytes)
    ) as second_archive:
        for filename in set(first_archive.namelist()) - {"metadata.json"}:
            assert first_archive.read(filename) == second_archive.read(filename)


def test_export_rejects_a_clean_validation_report_with_errors():
    config = SimulationConfig(n_matches=100, anomaly_rate_percent=0)
    dataset = simulate_dataset(config)
    clean_report = validate_dataset(dataset, config)
    invalid_report = type(clean_report)(
        errors=("invalid",),
        warnings=(),
        match_statistics=clean_report.match_statistics,
    )

    with pytest.raises(ValueError, match="refusing to export"):
        build_export_zip(dataset, (), (), config, invalid_report)
