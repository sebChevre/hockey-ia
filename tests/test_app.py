from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import _memory_warning_message


def test_streamlit_pages_load_and_navigation_works():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert "Générateur de matchs synthétiques" in [
        title.value for title in app.title
    ]
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
