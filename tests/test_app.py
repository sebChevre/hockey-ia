from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import _memory_warning_message, _rink_figure
from hockeyai.geometry import shot_geometry
from hockeyai.models import ManpowerSituation, Shot, ShotOutcome


def test_rink_plot_distinguishes_teams_and_goal_outcomes():
    geometry = shot_geometry(40, 15)
    shots = (
        Shot(
            "SHOT-1",
            "EVENT-1",
            "MATCH-1",
            "TEAM-01",
            "PLAYER-01",
            100,
            40,
            15,
            geometry.distance_m,
            geometry.angle_deg,
            "WRIST",
            ManpowerSituation(),
            0.5,
            0.5,
            0,
            True,
            ShotOutcome.GOAL,
        ),
        Shot(
            "SHOT-2",
            "EVENT-2",
            "MATCH-1",
            "TEAM-02",
            "PLAYER-02",
            200,
            40,
            15,
            geometry.distance_m,
            geometry.angle_deg,
            "WRIST",
            ManpowerSituation(),
            0.5,
            0.5,
            0,
            False,
            ShotOutcome.SAVED,
        ),
    )

    figure = _rink_figure(shots, "TEAM-01", "TEAM-02")
    traces = {trace.name: trace for trace in figure.data}

    assert len(figure.data) == 4
    assert traces["Domicile (TEAM-01) — But"].marker.color == "#dc2626"
    assert traces["Domicile (TEAM-01) — But"].marker.symbol == "circle"
    assert traces["Extérieur (TEAM-02) — Tir non converti"].marker.color == "#2563eb"
    assert (
        traces["Extérieur (TEAM-02) — Tir non converti"].marker.symbol == "triangle-up"
    )
    assert list(traces["Domicile (TEAM-01) — But"].x) == [40]
    assert list(traces["Extérieur (TEAM-02) — Tir non converti"].x) == [40]


def test_streamlit_pages_load_and_navigation_works():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert "Générateur de matchs synthétiques" in [title.value for title in app.title]
    assert app.sidebar.radio[0].value == "Générateur"

    for page, expected_title in (
        ("Explorer", "Explorer les données"),
        ("Exporter", "Exporter les données"),
    ):
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not app.exception
        assert expected_title in [title.value for title in app.title]
        assert app.info


def test_memory_warning_only_appears_when_estimate_exceeds_available_memory():
    assert _memory_warning_message(1000, 2_000_000_000) is None
    warning = _memory_warning_message(10_000, 100_000_000)

    assert warning is not None
    assert "Mio disponibles" in warning
    assert "ne garantit pas" in warning
