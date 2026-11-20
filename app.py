"""Streamlit presentation layer for the HockeyAI simulator."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from io import BytesIO
import time
from typing import Mapping, Sequence
from zipfile import ZipFile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st

from hockeyai.anomalies import event_to_row, inject_anomalies
from hockeyai.config import SimulationConfig
from hockeyai.engine import simulate_dataset
from hockeyai.export import config_json_bytes, build_export_zip
from hockeyai.models import AnomalyType, EventType, Shot, SimulationDataset
from hockeyai.validation import (
    CorruptedEventsReport,
    ValidationReport,
    validate_corrupted_events,
    validate_dataset,
)

_PAGES = ("Générateur", "Explorer", "Exporter")
_TABLE_LABELS = ("Équipes", "Joueurs", "Matchs", "Événements", "Tirs")
_ANOMALY_LABELS = {
    AnomalyType.DUPLICATE: "Doublon",
    AnomalyType.INVALID_POSITION: "Position hors patinoire",
    AnomalyType.INVALID_TIME: "Temps invalide",
    AnomalyType.MISSING: "Valeur manquante",
    AnomalyType.UNKNOWN_TEAM: "Équipe inconnue",
}
_MEMORY_ESTIMATE_PER_MATCH = 256 * 1024
_MEMORY_ESTIMATE_OVERHEAD = 128 * 1024 * 1024


def main() -> None:
    st.set_page_config(
        page_title="HockeyAI — Générateur", page_icon="🏒", layout="wide"
    )
    st.sidebar.title("HockeyAI")
    page = st.sidebar.radio("Navigation", _PAGES, key="active_page")
    st.sidebar.caption("Données synthétiques — moteur Python indépendant de Streamlit")
    if page == "Générateur":
        _render_generator()
    elif page == "Explorer":
        _render_explorer()
    else:
        _render_export()


def _render_generator() -> None:
    st.title("Générateur de matchs synthétiques")
    st.markdown(
        "Configurez puis générez des données synthétiques de hockey. "
        "Le simulateur n’entraîne aucun modèle d’IA."
    )
    st.caption(
        "Ouvrez les infobulles « ? » à côté des paramètres pour comprendre "
        "leur effet sur les données."
    )

    with st.form("simulation_config"):
        st.subheader("Scénario et volume")
        scenario_name = st.text_input(
            "Nom du scénario",
            value="reference",
            max_chars=50,
            help=(
                "Nom affiché dans la configuration et utilisé pour nommer les "
                "fichiers exportés. Il ne change pas les données simulées."
            ),
        )
        first_row = st.columns(3)
        with first_row[0]:
            n_matches = st.number_input(
                "Nombre de matchs",
                min_value=100,
                max_value=10_000,
                value=1000,
                step=100,
                help=(
                    "Nombre de matchs simulés (de 100 à 10 000). Plus de matchs "
                    "produisent davantage de lignes et stabilisent les moyennes, "
                    "au prix d'un temps de génération et d'une mémoire accrus."
                ),
            )
        with first_row[1]:
            n_teams = st.selectbox(
                "Nombre d’équipes",
                options=tuple(range(4, 17, 2)),
                index=4,
                help=(
                    "Nombre d'équipes synthétiques (4 à 16, par nombre pair). "
                    "Chaque match en tire deux distinctes. En augmenter le nombre "
                    "diversifie les oppositions et ajoute des équipes et joueurs."
                ),
            )
        with first_row[2]:
            st.text_input(
                "Effectif par équipe",
                value="18 joueurs de champ + 2 gardiens (fixe)",
                disabled=True,
                help=(
                    "Effectif fixé par le modèle : 12 attaquants, 6 défenseurs "
                    "et 2 gardiens par équipe."
                ),
            )

        st.subheader("Aléatoire et forces sportives")
        second_row = st.columns(3)
        with second_row[0]:
            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=2**32 - 1,
                value=42,
                step=1,
                help=(
                    "Graine pseudo-aléatoire. À paramètres identiques, la même "
                    "graine reproduit le même roster, les mêmes matchs et les "
                    "mêmes anomalies. La changer produit une autre simulation."
                ),
            )
        with second_row[1]:
            offensive_intensity = st.slider(
                "Intensité offensive",
                min_value=0,
                max_value=100,
                value=50,
                help=(
                    "Règle la fréquence des tirs : 50 correspond au niveau de "
                    "base, 0 la réduit de moitié et 100 l'augmente d'environ 50 %. "
                    "Cela modifie le nombre de tirs, de buts potentiels et "
                    "d'événements, pas directement la probabilité de but d'un tir."
                ),
            )
        with second_row[2]:
            team_imbalance = st.slider(
                "Déséquilibre des équipes",
                min_value=0,
                max_value=100,
                value=30,
                help=(
                    "Règle la diversité des forces générées (attaque, défense, "
                    "gardiens et compétences des joueurs). Une valeur élevée rend "
                    "les équipes plus différentes et déséquilibre davantage les "
                    "tirs et résultats entre adversaires."
                ),
            )

        st.subheader("Pénalités et avantage domicile")
        third_row = st.columns(2)
        with third_row[0]:
            penalties_per_match = st.number_input(
                "Espérance des pénalités par match",
                min_value=0.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                help=(
                    "Moyenne visée par match, échantillonnée selon une loi de "
                    "Poisson : le nombre exact varie d'un match à l'autre. Les "
                    "appels donnent 95 % de mineures (2 min) et 5 % de majeures "
                    "(5 min). Deux pénalités au plus sont servies simultanément "
                    "par équipe ; les suivantes attendent. Un but adverse en "
                    "supériorité peut terminer une mineure, jamais une majeure."
                ),
            )
        with third_row[1]:
            home_advantage = st.number_input(
                "Avantage domicile",
                min_value=0.0,
                max_value=0.20,
                value=0.05,
                step=0.01,
                format="%.2f",
                help=(
                    "Bonus multiplicatif au facteur d'attaque de l'équipe à "
                    "domicile (0 à 0,20; 0,05 = +5 %). Il influence son rythme "
                    "et sa part des tirs, sans garantir une victoire."
                ),
            )

        st.subheader("Anomalies contrôlées")
        anomaly_rate_percent = st.slider(
            "Lignes sources ciblées (%)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
            help=(
                "Pourcentage approximatif des lignes d'événements sélectionnées "
                "pour corruption. À 0 %, aucune anomalie n'est injectée. La table "
                "propre reste intacte; les doublons peuvent augmenter le nombre "
                "final de lignes de la copie corrompue."
            ),
        )
        anomaly_types = st.multiselect(
            "Types d’anomalies",
            options=tuple(sorted(AnomalyType, key=lambda item: item.value)),
            default=tuple(sorted(AnomalyType, key=lambda item: item.value)),
            format_func=lambda anomaly: _ANOMALY_LABELS[anomaly],
            help=(
                "Choisissez les corruptions à appliquer à la copie des événements. "
                "Une ligne source reçoit au plus un type d'anomalie; la table "
                "events.csv reste propre et les changements sont consignés dans "
                "anomaly_log.csv."
            ),
        )
        st.caption(
            "**Types :** Doublon (ajoute une ligne copiée) · Position hors "
            "patinoire (coordonnée x invalide) · Temps invalide (hors du match) · "
            "Valeur manquante (champ vidé) · Équipe inconnue (ID remplacé)."
        )
        submitted = st.form_submit_button(
            "Générer", type="primary", use_container_width=True
        )

    if submitted:
        try:
            config = SimulationConfig(
                scenario_name=scenario_name,
                n_matches=int(n_matches),
                n_teams=int(n_teams),
                seed=int(seed),
                offensive_intensity=int(offensive_intensity),
                team_imbalance=int(team_imbalance),
                penalties_per_match=float(penalties_per_match),
                home_advantage=float(home_advantage),
                anomaly_rate_percent=float(anomaly_rate_percent),
                anomaly_types=frozenset(anomaly_types),
            )
        except ValueError as error:
            st.error(f"Configuration invalide : {error}")
        else:
            st.session_state.pop("simulation_bundle", None)
            warning = _memory_warning_message(
                config.n_matches,
                psutil.virtual_memory().available,
            )
            if warning:
                st.warning(warning)
            progress_bar = st.progress(0.0, text="Préparation de la simulation…")
            last_progress_update = [0]
            started_at = time.perf_counter()

            def update_progress(completed: int, total: int) -> None:
                update_every = max(1, total // 100)
                if (
                    completed == total
                    or completed - last_progress_update[0] >= update_every
                ):
                    progress_bar.progress(
                        completed / total,
                        text=f"Matchs simulés : {completed:,} / {total:,}",
                    )
                    last_progress_update[0] = completed

            dataset = simulate_dataset(config, progress_callback=update_progress)
            clean_report = validate_dataset(dataset, config)
            if not clean_report.is_valid:
                st.error("Le validateur a trouvé des erreurs dans les données propres.")
                st.json(list(clean_report.errors))
                st.stop()

            corrupted_events, anomaly_log = inject_anomalies(
                dataset.events,
                config,
                known_team_ids={team.team_id for team in dataset.teams},
            )
            corrupted_report = validate_corrupted_events(dataset, corrupted_events)
            archive_bytes = build_export_zip(
                dataset,
                corrupted_events,
                anomaly_log,
                config,
                clean_report,
            )
            elapsed = time.perf_counter() - started_at
            st.session_state["simulation_bundle"] = {
                "config": config,
                "dataset": dataset,
                "clean_report": clean_report,
                "corrupted_events": corrupted_events,
                "anomaly_log": anomaly_log,
                "corrupted_report": corrupted_report,
                "archive_bytes": archive_bytes,
                "duration_seconds": elapsed,
            }
            st.success(f"Génération terminée en {elapsed:.2f} s.")

    bundle = _get_bundle()
    if bundle is not None:
        _render_generation_summary(bundle)


def _render_generation_summary(bundle: Mapping[str, object]) -> None:
    dataset: SimulationDataset = bundle["dataset"]
    report: ValidationReport = bundle["clean_report"]
    goals = sum(match.home_score + match.away_score for match in dataset.matches)
    penalties = sum(event.event_type is EventType.PENALTY for event in dataset.events)
    shots = len(dataset.shots)

    st.subheader("Résultat de la génération")
    metrics = st.columns(5)
    metrics[0].metric("Matchs", f"{len(dataset.matches):,}")
    metrics[1].metric("Événements", f"{len(dataset.events):,}")
    metrics[2].metric("Tirs", f"{shots:,}")
    metrics[3].metric("Buts", f"{goals:,}")
    metrics[4].metric("Pénalités", f"{penalties:,}")
    st.caption(
        f"Durée : {bundle['duration_seconds']:.2f} s · "
        f"Conversion des tirs : {goals / shots:.1%}"
        if shots
        else "Aucun tir généré."
    )
    if report.warnings:
        for warning in report.warnings:
            st.warning(warning)
    else:
        st.info(
            "Contrôles des données propres réussis. Les résultats restent synthétiques."
        )


def _render_explorer() -> None:
    st.title("Explorer les données")
    st.warning(
        "Toutes les données et visualisations sont synthétiques; elles ne décrivent "
        "pas des observations sportives réelles."
    )
    bundle = _get_bundle()
    if bundle is None:
        st.info("Lancez d’abord une génération depuis la page Générateur.")
        return

    dataset: SimulationDataset = bundle["dataset"]
    matches_by_id = {match.match_id: match for match in dataset.matches}
    event_counts = Counter(event.event_type.value for event in dataset.events)
    total_goals = sum(match.home_score + match.away_score for match in dataset.matches)
    total_penalties = event_counts[EventType.PENALTY.value]
    metrics = st.columns(6)
    metrics[0].metric("Matchs", f"{len(dataset.matches):,}")
    metrics[1].metric("Tirs", f"{len(dataset.shots):,}")
    metrics[2].metric("Buts", f"{total_goals:,}")
    metrics[3].metric(
        "Conversion",
        f"{total_goals / len(dataset.shots):.1%}" if dataset.shots else "—",
    )
    metrics[4].metric("Pénalités", f"{total_penalties:,}")
    metrics[5].metric("Événements", f"{len(dataset.events):,}")

    st.subheader("Statistiques globales")
    chart_columns = st.columns(2)
    goals_by_match = [match.home_score + match.away_score for match in dataset.matches]
    with chart_columns[0]:
        goals_frame = pd.DataFrame({"Buts par match": goals_by_match})
        st.plotly_chart(
            px.histogram(
                goals_frame,
                x="Buts par match",
                nbins=max(1, max(goals_by_match, default=1) + 1),
                title="Histogramme des buts par match",
            ),
            use_container_width=True,
        )
    with chart_columns[1]:
        event_frame = pd.DataFrame(
            {
                "Type d’événement": list(event_counts),
                "Nombre": list(event_counts.values()),
            }
        )
        st.plotly_chart(
            px.bar(event_frame, x="Type d’événement", y="Nombre", title="Événements"),
            use_container_width=True,
        )

    match_id = st.selectbox(
        "Match à explorer", options=tuple(matches_by_id), format_func=str
    )
    match = matches_by_id[match_id]
    st.markdown(
        f"**{match.match_id}** — {match.home_team_id} {match.home_score} : "
        f"{match.away_score} {match.away_team_id}"
    )
    match_events = [event for event in dataset.events if event.match_id == match_id]
    timeline = pd.DataFrame(
        [
            {
                "elapsed_seconds": event.elapsed_seconds,
                "period": event.period,
                "event_type": event.event_type.value,
                "team_id": event.team_id,
                "player_id": event.player_id,
                "penalty_type": (
                    event.penalty_type.value if event.penalty_type else None
                ),
                "penalty_duration_seconds": event.penalty_duration_seconds,
                "related_penalty_event_id": event.related_penalty_event_id,
                "penalty_end_reason": (
                    event.penalty_end_reason.value
                    if event.penalty_end_reason
                    else None
                ),
                "home_score_before": event.home_score_before,
                "away_score_before": event.away_score_before,
                "manpower_situation": event.manpower_situation.code,
            }
            for event in match_events
        ]
    )
    st.dataframe(timeline, hide_index=True, use_container_width=True)

    match_shots = [shot for shot in dataset.shots if shot.match_id == match_id]
    st.caption(
        "Couleur : bleu = tir non converti, rouge = but. Symbole : cercle = "
        "domicile, triangle = extérieur. Les deux équipes attaquent dans le "
        "même repère vers (60, 15)."
    )
    st.plotly_chart(
        _rink_figure(match_shots, match.home_team_id, match.away_team_id),
        use_container_width=True,
    )

    st.subheader("Aperçu des cinq datasets propres")
    tabs = st.tabs(_TABLE_LABELS)
    for tab, (label, rows) in zip(tabs, _table_rows(dataset).items()):
        with tab:
            st.caption(f"{label} — aperçu limité aux 100 premières lignes")
            st.dataframe(
                pd.DataFrame(rows).head(100),
                hide_index=True,
                use_container_width=True,
            )

    st.subheader("Rapports de cohérence")
    clean_report: ValidationReport = bundle["clean_report"]
    corrupted_report: CorruptedEventsReport = bundle["corrupted_report"]
    if clean_report.is_valid:
        st.success("Données propres : tous les contrôles d’intégrité passent.")
    else:
        st.error("Données propres : des violations ont été détectées.")
    if clean_report.warnings:
        st.warning("\n".join(clean_report.warnings))
    with st.expander("Détails des rapports"):
        st.write("Données propres")
        st.json({"errors": clean_report.errors, "warnings": clean_report.warnings})
        st.write("Copie corrompue")
        st.json(
            {
                "valid": corrupted_report.is_valid,
                "detected_by_type": dict(corrupted_report.detected_by_type),
                "errors_sample": corrupted_report.errors[:20],
                "errors_total": len(corrupted_report.errors),
            }
        )


def _render_export() -> None:
    st.title("Exporter les données")
    bundle = _get_bundle()
    if bundle is None:
        st.info("Lancez d’abord une génération depuis la page Générateur.")
        return

    config: SimulationConfig = bundle["config"]
    dataset: SimulationDataset = bundle["dataset"]
    corrupted_events = bundle["corrupted_events"]
    anomaly_log = bundle["anomaly_log"]
    st.markdown(
        "L’archive contient les données propres et leur copie dégradée, "
        "la configuration et les rapports d’évaluation."
    )
    columns = st.columns(2)
    columns[0].download_button(
        "Télécharger le ZIP",
        data=bundle["archive_bytes"],
        file_name=f"{config.scenario_slug}.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )
    columns[1].download_button(
        "Télécharger la configuration JSON",
        data=config_json_bytes(config),
        file_name=f"{config.scenario_slug}-config.json",
        mime="application/json",
        use_container_width=True,
    )
    st.subheader("Contenu de l’archive")
    st.dataframe(
        pd.DataFrame(_archive_manifest(bundle)),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"{len(dataset.events):,} événements propres · "
        f"{len(corrupted_events):,} événements dans la copie · "
        f"{len(anomaly_log):,} opérations d’anomalie"
    )


def _rink_figure(
    shots: Sequence[Shot], home_team_id: str, away_team_id: str
) -> go.Figure:
    figure = go.Figure()
    for team_id, team_label, symbol in (
        (home_team_id, "Domicile", "circle"),
        (away_team_id, "Extérieur", "triangle-up"),
    ):
        for is_goal, outcome_label, color in (
            (False, "Tir non converti", "#2563eb"),
            (True, "But", "#dc2626"),
        ):
            selected = [
                shot
                for shot in shots
                if shot.team_id == team_id and shot.is_goal is is_goal
            ]
            figure.add_trace(
                go.Scatter(
                    x=[shot.x for shot in selected],
                    y=[shot.y for shot in selected],
                    mode="markers",
                    name=f"{team_label} ({team_id}) — {outcome_label}",
                    legendgroup=team_id,
                    marker={
                        "color": color,
                        "symbol": symbol,
                        "size": 10,
                        "opacity": 0.8,
                    },
                    text=[shot.shot_id for shot in selected],
                    hovertemplate=(
                        "%{text}<br>"
                        f"{team_label} ({team_id}) · {outcome_label}"
                        "<br>x=%{x:.1f} m, y=%{y:.1f} m<extra></extra>"
                    ),
                )
            )
    figure.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=60,
        y1=30,
        line={"color": "#64748b", "width": 2},
    )
    figure.add_vline(x=30, line_dash="dash", line_color="#94a3b8")
    figure.add_shape(
        type="circle",
        x0=59.4,
        x1=60,
        y0=14.4,
        y1=15.6,
        line={"color": "#dc2626", "width": 3},
    )
    figure.update_layout(
        title="Tirs du match dans le repère offensif commun",
        xaxis={"title": "x (m)", "range": [0, 60], "constrain": "domain"},
        yaxis={"title": "y (m)", "range": [0, 30], "scaleanchor": "x"},
        height=440,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return figure


def _table_rows(dataset: SimulationDataset) -> dict[str, list[dict[str, object]]]:
    teams = [asdict(row) for row in dataset.teams]
    players = []
    for player in dataset.players:
        row = asdict(player)
        row["position"] = player.position.value
        players.append(row)
    matches = [asdict(row) for row in dataset.matches]
    events = [event_to_row(row) for row in dataset.events]
    shots = []
    for shot in dataset.shots:
        row = asdict(shot)
        row["manpower_situation"] = shot.manpower_situation.code
        row["shot_outcome"] = shot.shot_outcome.value
        shots.append(row)
    return {
        "Équipes": teams,
        "Joueurs": players,
        "Matchs": matches,
        "Événements": events,
        "Tirs": shots,
    }


def _archive_manifest(bundle: Mapping[str, object]) -> list[dict[str, object]]:
    dataset: SimulationDataset = bundle["dataset"]
    row_counts: dict[str, int | None] = {
        "teams.csv": len(dataset.teams),
        "players.csv": len(dataset.players),
        "matches.csv": len(dataset.matches),
        "events.csv": len(dataset.events),
        "events_corrupted.csv": len(bundle["corrupted_events"]),
        "shots.csv": len(dataset.shots),
        "shot_ground_truth.csv": len(dataset.shot_ground_truth),
        "anomaly_log.csv": len(bundle["anomaly_log"]),
    }
    manifest = []
    with ZipFile(BytesIO(bundle["archive_bytes"])) as archive:
        for info in archive.infolist():
            manifest.append(
                {
                    "Fichier": info.filename,
                    "Lignes de données": row_counts.get(info.filename),
                    "Taille décompressée (octets)": info.file_size,
                }
            )
    return manifest


def _get_bundle() -> Mapping[str, object] | None:
    value = st.session_state.get("simulation_bundle")
    return value if isinstance(value, dict) else None


def _memory_warning_message(n_matches: int, available_memory_bytes: int) -> str | None:
    estimated = _MEMORY_ESTIMATE_OVERHEAD + n_matches * _MEMORY_ESTIMATE_PER_MATCH
    if estimated <= available_memory_bytes:
        return None
    estimated_mib = estimated / (1024 * 1024)
    available_mib = available_memory_bytes / (1024 * 1024)
    return (
        f"Estimation mémoire indicative : environ {estimated_mib:.0f} Mio requis "
        f"contre {available_mib:.0f} Mio disponibles. Réduisez le nombre de matchs "
        "si la machine manque de mémoire. Cette estimation heuristique ne garantit "
        "pas la réussite ni la durée de la génération."
    )


if __name__ == "__main__":
    main()
