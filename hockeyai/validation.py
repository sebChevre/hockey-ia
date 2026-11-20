"""Cross-table integrity checks for clean simulator datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from hockeyai.config import SimulationConfig
from hockeyai.geometry import (
    PERIOD_LENGTH_SECONDS,
    REGULATION_LENGTH_SECONDS,
    period_for_elapsed_seconds,
    shot_geometry,
    validate_position,
)
from hockeyai.models import (
    AnomalyType,
    Event,
    EventType,
    PlayerPosition,
    PenaltyEndReason,
    PenaltyType,
    ShotOutcome,
    SimulationDataset,
)


@dataclass(frozen=True, slots=True)
class MatchStatistics:
    match_id: str
    shots: int
    goals: int
    penalties: int


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    match_statistics: tuple[MatchStatistics, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class CorruptedEventsReport:
    errors: tuple[str, ...]
    detected_by_type: tuple[tuple[str, int], ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dataset(
    dataset: SimulationDataset, config: SimulationConfig | None = None
) -> ValidationReport:
    """Check keys, event/shot links, scores, timing, geometry, and penalties."""
    errors: list[str] = []
    warnings: list[str] = []
    teams = {team.team_id: team for team in dataset.teams}
    players = {player.player_id: player for player in dataset.players}
    matches = {match.match_id: match for match in dataset.matches}
    shots = {shot.shot_id: shot for shot in dataset.shots}
    events_by_id = {event.event_id: event for event in dataset.events}
    events_by_match: dict[str, list] = defaultdict(list)

    _check_unique_ids("team_id", (team.team_id for team in dataset.teams), errors)
    _check_unique_ids("player_id", (player.player_id for player in dataset.players), errors)
    _check_unique_ids("match_id", (match.match_id for match in dataset.matches), errors)
    _check_unique_ids("event_id", (event.event_id for event in dataset.events), errors)
    _check_unique_ids("shot_id", (shot.shot_id for shot in dataset.shots), errors)
    _check_unique_ids(
        "ground-truth shot_id",
        (row.shot_id for row in dataset.shot_ground_truth),
        errors,
    )
    if list(dataset.events) != sorted(
        dataset.events,
        key=lambda event: (event.match_id, event.elapsed_seconds, event.event_sequence),
    ):
        errors.append("events are not globally ordered by match, time, and sequence")

    players_per_team: dict[str, Counter] = defaultdict(Counter)
    for player in dataset.players:
        if player.team_id not in teams:
            errors.append(f"player {player.player_id} references unknown team {player.team_id}")
        players_per_team[player.team_id][player.position] += 1
    for team_id in teams:
        counts = players_per_team[team_id]
        if (
            counts[PlayerPosition.FORWARD] != 12
            or counts[PlayerPosition.DEFENSE] != 6
            or counts[PlayerPosition.GOALIE] != 2
        ):
            errors.append(
                f"team {team_id} roster must contain 12 forwards, 6 defensemen, and 2 goalies"
            )

    for match in dataset.matches:
        if match.home_team_id not in teams or match.away_team_id not in teams:
            errors.append(f"match {match.match_id} references an unknown team")

    for event in dataset.events:
        events_by_match[event.match_id].append(event)
        if event.match_id not in matches:
            errors.append(f"event {event.event_id} references unknown match {event.match_id}")
        if event.team_id is not None and event.team_id not in teams:
            errors.append(f"event {event.event_id} references unknown team {event.team_id}")
        if event.player_id is not None:
            player = players.get(event.player_id)
            if player is None:
                errors.append(f"event {event.event_id} references unknown player {event.player_id}")
            elif event.team_id != player.team_id:
                errors.append(f"event {event.event_id} player does not belong to its team")
            elif event.event_type is EventType.SAVE and player.position is not PlayerPosition.GOALIE:
                errors.append(f"save {event.event_id} must identify a goalie")
        if event.event_type is EventType.FACEOFF:
            winner = players.get(event.faceoff_winner_player_id or "")
            loser = players.get(event.faceoff_loser_player_id or "")
            if winner is None:
                errors.append(
                    f"faceoff {event.event_id} references an unknown winner"
                )
            elif winner.position is not PlayerPosition.FORWARD:
                errors.append(f"faceoff {event.event_id} winner must be a forward")
            if loser is None:
                errors.append(
                    f"faceoff {event.event_id} references an unknown loser"
                )
            elif loser.position is not PlayerPosition.FORWARD:
                errors.append(f"faceoff {event.event_id} loser must be a forward")
            match = matches.get(event.match_id)
            if winner is not None and loser is not None:
                if winner.team_id == loser.team_id:
                    errors.append(
                        f"faceoff {event.event_id} winner and loser must be opponents"
                    )
                if match is not None and (
                    winner.team_id not in (match.home_team_id, match.away_team_id)
                    or loser.team_id not in (match.home_team_id, match.away_team_id)
                ):
                    errors.append(
                        f"faceoff {event.event_id} players must belong to match teams"
                    )
        if event.event_type is EventType.SHOT and (
            event.team_id is None or event.player_id is None
        ):
            errors.append(f"shot event {event.event_id} must identify team and shooter")
        if event.event_type not in (EventType.GOAL, EventType.SAVE) and (
            event.related_shot_id is not None
        ):
            errors.append(f"event {event.event_id} has an unexpected related_shot_id")
        if event.period != period_for_elapsed_seconds(event.elapsed_seconds):
            errors.append(f"event {event.event_id} has an inconsistent period")
        if event.x is not None and event.y is not None:
            try:
                validate_position(event.x, event.y)
            except ValueError:
                errors.append(f"event {event.event_id} has an invalid position")
        if event.event_type in (EventType.GOAL, EventType.SAVE):
            if event.related_shot_id not in shots:
                errors.append(f"event {event.event_id} references an unknown shot")

    shots_by_event: Counter = Counter()
    for shot in dataset.shots:
        event = events_by_id.get(shot.event_id)
        if event is None:
            errors.append(f"shot {shot.shot_id} has no SHOT event {shot.event_id}")
        elif event.event_type is not EventType.SHOT:
            errors.append(f"shot {shot.shot_id} is linked to a non-SHOT event")
        else:
            shots_by_event[event.event_id] += 1
            if (
                event.match_id != shot.match_id
                or event.team_id != shot.team_id
                or event.player_id != shot.player_id
                or event.elapsed_seconds != shot.elapsed_seconds
            ):
                errors.append(f"shot {shot.shot_id} does not match its event")
        player = players.get(shot.player_id)
        if player is None:
            errors.append(f"shot {shot.shot_id} references unknown player {shot.player_id}")
        elif player.team_id != shot.team_id:
            errors.append(f"shot {shot.shot_id} shooter does not belong to its team")
        elif player.position is PlayerPosition.GOALIE:
            errors.append(f"shot {shot.shot_id} is attributed to a goalie")
        if shot.match_id not in matches:
            errors.append(f"shot {shot.shot_id} references unknown match {shot.match_id}")
        else:
            match = matches[shot.match_id]
            if shot.team_id not in (match.home_team_id, match.away_team_id):
                errors.append(f"shot {shot.shot_id} team is not playing in its match")
            expected_difference = (
                event.home_score_before - event.away_score_before
                if event is not None and shot.team_id == match.home_team_id
                else event.away_score_before - event.home_score_before
                if event is not None
                else None
            )
            if (
                expected_difference is not None
                and shot.score_difference_before != expected_difference
            ):
                errors.append(f"shot {shot.shot_id} has an inconsistent score difference")
        try:
            validate_position(shot.x, shot.y)
            geometry = shot_geometry(shot.x, shot.y)
            if abs(geometry.distance_m - shot.distance_m) > 1e-9:
                errors.append(f"shot {shot.shot_id} has an inconsistent distance")
            if abs(geometry.angle_deg - shot.angle_deg) > 1e-9:
                errors.append(f"shot {shot.shot_id} has an inconsistent angle")
        except ValueError:
            errors.append(f"shot {shot.shot_id} has invalid geometry")

    shot_event_ids = {
        event.event_id for event in dataset.events if event.event_type is EventType.SHOT
    }
    for event_id in shot_event_ids:
        if shots_by_event[event_id] != 1:
            errors.append(f"SHOT event {event_id} must have exactly one shot row")

    outcomes_by_shot: Counter = Counter()
    for event in dataset.events:
        if event.event_type in (EventType.GOAL, EventType.SAVE):
            outcomes_by_shot[event.related_shot_id] += 1
            shot = shots.get(event.related_shot_id)
            if shot is not None:
                match = matches.get(shot.match_id)
                if match is None:
                    continue
                shot_event = events_by_id.get(shot.event_id)
                expected_type = (
                    EventType.GOAL
                    if shot.shot_outcome is ShotOutcome.GOAL
                    else EventType.SAVE
                    if shot.shot_outcome is ShotOutcome.SAVED
                    else None
                )
                if event.event_type is not expected_type:
                    errors.append(
                        f"outcome event {event.event_id} does not match shot {shot.shot_id}"
                    )
                if (
                    event.match_id != shot.match_id
                    or event.elapsed_seconds != shot.elapsed_seconds
                    or shot_event is None
                    or event.event_sequence <= shot_event.event_sequence
                ):
                    errors.append(
                        f"outcome event {event.event_id} is not ordered with its shot"
                    )
                expected_team = (
                    shot.team_id
                    if event.event_type is EventType.GOAL
                    else match.away_team_id
                    if shot.team_id == match.home_team_id
                    else match.home_team_id
                )
                if event.team_id != expected_team:
                    errors.append(f"outcome event {event.event_id} has the wrong team")

    for shot in dataset.shots:
        expected_outcomes = 1 if shot.shot_outcome in (ShotOutcome.GOAL, ShotOutcome.SAVED) else 0
        if outcomes_by_shot[shot.shot_id] != expected_outcomes:
            errors.append(f"shot {shot.shot_id} has an invalid number of outcome events")
        if shot.is_goal != (shot.shot_outcome is ShotOutcome.GOAL):
            errors.append(f"shot {shot.shot_id} has an inconsistent goal label")

    stats: list[MatchStatistics] = []
    shots_per_match = Counter(shot.match_id for shot in dataset.shots)
    for match in dataset.matches:
        match_events = events_by_match[match.match_id]
        goals = [event for event in match_events if event.event_type is EventType.GOAL]
        penalties = [event for event in match_events if event.event_type is EventType.PENALTY]
        stats.append(
            MatchStatistics(
                match.match_id,
                shots_per_match[match.match_id],
                len(goals),
                len(penalties),
            )
        )
        expected_home = sum(event.team_id == match.home_team_id for event in goals)
        expected_away = sum(event.team_id == match.away_team_id for event in goals)
        if (match.home_score, match.away_score) != (expected_home, expected_away):
            errors.append(f"match {match.match_id} final score does not match GOAL events")
        _check_match_events(match, match_events, errors)

    ground_truth_shot_ids = {
        truth.shot_id for truth in dataset.shot_ground_truth
    }
    if ground_truth_shot_ids != set(shots):
        errors.append("shot ground truth must contain exactly one row per shot")

    if config is not None:
        warnings.extend(_calibration_warnings(stats, config))

    return ValidationReport(tuple(errors), tuple(warnings), tuple(stats))


def validate_corrupted_events(
    dataset: SimulationDataset, event_rows: Sequence[Mapping[str, object]]
) -> CorruptedEventsReport:
    """Report corruption in a copied events table without changing clean data."""
    errors: list[str] = []
    detected: Counter[str] = Counter()
    teams = {team.team_id for team in dataset.teams}
    players = {player.player_id: player for player in dataset.players}
    matches = {match.match_id: match for match in dataset.matches}
    source_penalties = {
        event.event_id: event
        for event in dataset.events
        if event.event_type is EventType.PENALTY
    }
    event_id_counts = Counter(
        row.get("event_id")
        for row in event_rows
        if isinstance(row.get("event_id"), str)
    )
    source_event_ids = {event.event_id for event in dataset.events}
    for event_id in sorted(source_event_ids - set(event_id_counts)):
        detected[AnomalyType.MISSING.value] += 1
        errors.append(f"[missing] source event {event_id} is absent from the copied table")
    for event_id in sorted(set(event_id_counts) - source_event_ids):
        errors.append(f"[reference] copied table contains unknown event_id {event_id}")
    for event_id, count in event_id_counts.items():
        if count > 1:
            detected[AnomalyType.DUPLICATE.value] += count - 1
            errors.append(f"[duplicate] event_id {event_id} occurs {count} times")

    required_by_type = {
        EventType.FACEOFF.value: (
            "faceoff_winner_player_id",
            "faceoff_loser_player_id",
        ),
        EventType.SHOT.value: ("team_id", "player_id", "x", "y"),
        EventType.SAVE.value: ("team_id", "player_id", "x", "y"),
        EventType.GOAL.value: ("team_id", "player_id", "x", "y"),
        EventType.PENALTY.value: (
            "team_id",
            "player_id",
            "penalty_type",
            "penalty_duration_seconds",
        ),
        EventType.PENALTY_START.value: (
            "team_id",
            "player_id",
            "related_penalty_event_id",
        ),
        EventType.PENALTY_END.value: (
            "team_id",
            "player_id",
            "related_penalty_event_id",
            "penalty_end_reason",
        ),
    }

    for row in event_rows:
        event_id = row.get("event_id")
        label = str(event_id) if event_id is not None else "<missing event_id>"
        event_type = str(row.get("event_type", ""))
        required_fields = required_by_type.get(event_type, ())
        for field in required_fields:
            if row.get(field) is None:
                detected[AnomalyType.MISSING.value] += 1
                errors.append(f"[missing] event {label} has no {field}")

        match_id = row.get("match_id")
        if not isinstance(match_id, str) or match_id not in matches:
            errors.append(f"[reference] event {label} references unknown match {match_id}")

        elapsed = row.get("elapsed_seconds")
        elapsed_is_valid = (
            isinstance(elapsed, int)
            and not isinstance(elapsed, bool)
            and 0 <= elapsed <= REGULATION_LENGTH_SECONDS
        )
        if not elapsed_is_valid:
            detected[AnomalyType.INVALID_TIME.value] += 1
            errors.append(f"[invalid_time] event {label} has invalid elapsed_seconds")
        elif row.get("period") != period_for_elapsed_seconds(elapsed):
            errors.append(f"[time] event {label} has inconsistent period")

        team_id = row.get("team_id")
        if team_id is not None and (
            not isinstance(team_id, str) or team_id not in teams
        ):
            detected[AnomalyType.UNKNOWN_TEAM.value] += 1
            errors.append(f"[unknown_team] event {label} references team {team_id}")
        player_id = row.get("player_id")
        if player_id is not None:
            player = players.get(str(player_id))
            if player is None:
                errors.append(f"[reference] event {label} references unknown player {player_id}")
            elif team_id is not None and player.team_id != team_id:
                errors.append(f"[reference] event {label} player does not belong to team")
            elif event_type == EventType.SAVE.value and player.position is not PlayerPosition.GOALIE:
                errors.append(f"[reference] save {label} does not identify a goalie")

        if event_type in (EventType.PENALTY_START.value, EventType.PENALTY_END.value):
            penalty_id = row.get("related_penalty_event_id")
            source_penalty = (
                source_penalties.get(penalty_id)
                if isinstance(penalty_id, str)
                else None
            )
            if source_penalty is None:
                errors.append(
                    f"[reference] event {label} references unknown penalty {penalty_id}"
                )
            elif (
                source_penalty.team_id != team_id
                or source_penalty.player_id != player_id
            ):
                errors.append(
                    f"[reference] event {label} does not match its source penalty"
                )

        if event_type == EventType.PENALTY.value:
            penalty_type = row.get("penalty_type")
            expected_duration = {
                PenaltyType.MINOR.value: PenaltyType.MINOR.duration_seconds,
                PenaltyType.MAJOR.value: PenaltyType.MAJOR.duration_seconds,
            }.get(str(penalty_type))
            if (
                expected_duration is None
                or row.get("penalty_duration_seconds") != expected_duration
            ):
                errors.append(f"[penalty] event {label} has invalid type or duration")
        elif event_type == EventType.PENALTY_END.value and row.get(
            "penalty_end_reason"
        ) not in {reason.value for reason in PenaltyEndReason}:
            errors.append(f"[penalty] event {label} has invalid end reason")

        if event_type == EventType.FACEOFF.value:
            faceoff_players = {}
            for role, field in (
                ("winner", "faceoff_winner_player_id"),
                ("loser", "faceoff_loser_player_id"),
            ):
                faceoff_player_id = row.get(field)
                if faceoff_player_id is None:
                    continue
                faceoff_player = (
                    players.get(faceoff_player_id)
                    if isinstance(faceoff_player_id, str)
                    else None
                )
                if faceoff_player is None:
                    errors.append(
                        f"[reference] faceoff {label} has an unknown {role} player"
                    )
                elif faceoff_player.position is not PlayerPosition.FORWARD:
                    errors.append(
                        f"[reference] faceoff {label} {role} must be a forward"
                    )
                else:
                    faceoff_players[role] = faceoff_player
            winner = faceoff_players.get("winner")
            loser = faceoff_players.get("loser")
            if winner is not None and loser is not None:
                if winner.team_id == loser.team_id:
                    errors.append(
                        f"[reference] faceoff {label} players must be opponents"
                    )
                match = matches.get(match_id)
                if match is not None and (
                    winner.team_id not in (match.home_team_id, match.away_team_id)
                    or loser.team_id not in (match.home_team_id, match.away_team_id)
                ):
                    errors.append(
                        f"[reference] faceoff {label} players must belong to match teams"
                    )

        x, y = row.get("x"), row.get("y")
        if (x is None) != (y is None):
            if "x" in required_fields or "y" in required_fields:
                errors.append(f"[missing] event {label} has incomplete coordinates")
        elif x is not None and y is not None:
            try:
                validate_position(x, y)
            except (TypeError, ValueError):
                detected[AnomalyType.INVALID_POSITION.value] += 1
                errors.append(f"[invalid_position] event {label} has out-of-rink coordinates")

    return CorruptedEventsReport(
        errors=tuple(errors),
        detected_by_type=tuple(sorted(detected.items())),
    )


def _check_match_events(match, events, errors) -> None:
    ordered = sorted(events, key=lambda event: (event.elapsed_seconds, event.event_sequence))
    if list(events) != ordered:
        errors.append(f"events for match {match.match_id} are not time-ordered")
    if len({event.event_sequence for event in events}) != len(events):
        errors.append(f"match {match.match_id} has duplicate event_sequence values")
    faceoffs = [event for event in ordered if event.event_type is EventType.FACEOFF]
    for period_start in (
        0,
        PERIOD_LENGTH_SECONDS,
        2 * PERIOD_LENGTH_SECONDS,
    ):
        if not any(event.elapsed_seconds == period_start for event in faceoffs):
            errors.append(
                f"match {match.match_id} has no faceoff at period start {period_start}"
            )
    for goal in (event for event in ordered if event.event_type is EventType.GOAL):
        if not any(
            event.elapsed_seconds == goal.elapsed_seconds
            and event.event_sequence > goal.event_sequence
            for event in faceoffs
        ):
            errors.append(f"goal {goal.event_id} is not followed by a faceoff")

    home_score = 0
    away_score = 0
    assessed_penalties: dict[str, Event] = {}
    penalty_status: dict[str, str] = {}
    active_penalties: dict[str, tuple[Event, Event]] = {}
    queued_penalties: dict[str, list[str]] = defaultdict(list)
    expected_start: str | None = None
    expected_goal_end: str | None = None
    expected_goal_time: int | None = None

    for event in ordered:
        if expected_goal_end is not None and not (
            event.event_type is EventType.PENALTY_END
            and event.related_penalty_event_id == expected_goal_end
        ):
            errors.append(
                f"goal penalty end for {expected_goal_end} is not the next event"
            )
            expected_goal_end = None
            expected_goal_time = None
        if expected_start is not None and not (
            event.event_type is EventType.PENALTY_START
            and event.related_penalty_event_id == expected_start
        ):
            errors.append(
                f"penalty start for {expected_start} is missing or out of order"
            )
            expected_start = None

        if (event.home_score_before, event.away_score_before) != (home_score, away_score):
            errors.append(f"event {event.event_id} has an inconsistent score-before snapshot")
        if event.event_type is EventType.PENALTY:
            if event.team_id is None or event.player_id is None:
                errors.append(f"penalty {event.event_id} must identify team and player")
            if event.penalty_type is None or event.penalty_duration_seconds is None:
                errors.append(
                    f"penalty {event.event_id} must identify type and duration"
                )
            else:
                assessed_penalties[event.event_id] = event
                penalty_status[event.event_id] = "queued"
                if (
                    len(queued_penalties[event.team_id]) == 0
                    and sum(
                        penalty[0].team_id == event.team_id
                        for penalty in active_penalties.values()
                    )
                    < 2
                ):
                    expected_start = event.event_id
                else:
                    queued_penalties[event.team_id].append(event.event_id)
        elif event.event_type is EventType.PENALTY_START:
            penalty_id = event.related_penalty_event_id
            source = assessed_penalties.get(penalty_id or "")
            if source is None:
                errors.append(f"penalty start {event.event_id} has no assessment")
            else:
                if expected_start != penalty_id:
                    errors.append(f"penalty start {event.event_id} is unexpected")
                if (event.team_id, event.player_id) != (
                    source.team_id,
                    source.player_id,
                ):
                    errors.append(
                        f"penalty start {event.event_id} does not match its assessment"
                    )
                if penalty_status.get(penalty_id) != "queued":
                    errors.append(f"penalty {penalty_id} starts more than once")
                active_for_team = sum(
                    active[0].team_id == source.team_id
                    for active in active_penalties.values()
                )
                if active_for_team >= 2:
                    errors.append(f"team {source.team_id} exceeds two active penalties")
                active_penalties[penalty_id] = (source, event)
                penalty_status[penalty_id] = "active"
            if expected_start == penalty_id:
                expected_start = None
        elif event.event_type is EventType.PENALTY_END:
            penalty_id = event.related_penalty_event_id
            active = active_penalties.get(penalty_id or "")
            if active is None:
                errors.append(f"penalty end {event.event_id} has no active penalty")
            else:
                source, start_event = active
                if (event.team_id, event.player_id) != (
                    source.team_id,
                    source.player_id,
                ):
                    errors.append(
                        f"penalty end {event.event_id} does not match its assessment"
                    )
                if event.penalty_end_reason is PenaltyEndReason.TIME_EXPIRED:
                    expected_expiry = _penalty_expiry(source, start_event)
                    if event.elapsed_seconds != expected_expiry:
                        errors.append(
                            f"penalty end {event.event_id} has an inconsistent expiry time"
                        )
                elif event.penalty_end_reason is PenaltyEndReason.GOAL:
                    if (
                        expected_goal_end != penalty_id
                        or expected_goal_time != event.elapsed_seconds
                        or source.penalty_type is not PenaltyType.MINOR
                    ):
                        errors.append(
                            f"penalty end {event.event_id} is not a valid power-play goal end"
                        )
                else:
                    errors.append(
                        f"penalty end {event.event_id} has no valid end reason"
                    )
                active_penalties.pop(penalty_id, None)
                penalty_status[penalty_id] = "ended"
                if expected_goal_end == penalty_id:
                    expected_goal_end = None
                    expected_goal_time = None
                queue = queued_penalties[source.team_id]
                if (
                    queue
                    and sum(
                        penalty[0].team_id == source.team_id
                        for penalty in active_penalties.values()
                    )
                    < 2
                ):
                    expected_start = queue.pop(0)

        elif event.event_type is EventType.GOAL:
            penalized_team_id = (
                match.away_team_id
                if event.team_id == match.home_team_id
                and event.manpower_situation.away_skaters
                < event.manpower_situation.home_skaters
                else match.home_team_id
                if event.team_id == match.away_team_id
                and event.manpower_situation.home_skaters
                < event.manpower_situation.away_skaters
                else None
            )
            if penalized_team_id is not None:
                eligible_minors = [
                    (penalty_id, source, start)
                    for penalty_id, (source, start) in active_penalties.items()
                    if source.team_id == penalized_team_id
                    and source.penalty_type is PenaltyType.MINOR
                ]
                if eligible_minors:
                    penalty_id, _, _ = min(
                        eligible_minors,
                        key=lambda penalty: (
                            _penalty_expiry(penalty[1], penalty[2]),
                            penalty[1].event_sequence,
                        ),
                    )
                    expected_goal_end = penalty_id
                    expected_goal_time = event.elapsed_seconds

        expected_home = max(
            3,
            5
            - sum(
                penalty[0].team_id == match.home_team_id
                for penalty in active_penalties.values()
            ),
        )
        expected_away = max(
            3,
            5
            - sum(
                penalty[0].team_id == match.away_team_id
                for penalty in active_penalties.values()
            ),
        )
        if (event.manpower_situation.home_skaters, event.manpower_situation.away_skaters) != (
            expected_home,
            expected_away,
        ):
            errors.append(f"event {event.event_id} has an inconsistent manpower snapshot")
        if event.event_type is EventType.GOAL:
            if event.team_id == match.home_team_id:
                home_score += 1
            elif event.team_id == match.away_team_id:
                away_score += 1
            else:
                errors.append(f"goal {event.event_id} is not attributed to a match team")
    if expected_start is not None:
        errors.append(f"penalty start for {expected_start} is missing")
    if expected_goal_end is not None:
        errors.append(f"goal penalty end for {expected_goal_end} is missing")
    for penalty_id, (source, start_event) in active_penalties.items():
        expected_expiry = _penalty_expiry(source, start_event)
        if expected_expiry <= REGULATION_LENGTH_SECONDS:
            errors.append(f"penalty {penalty_id} expired without a penalty-end event")


def _penalty_expiry(source: Event, start_event: Event) -> int:
    if source.penalty_duration_seconds is None:
        return start_event.elapsed_seconds
    return start_event.elapsed_seconds + source.penalty_duration_seconds


def _calibration_warnings(
    stats: Sequence[MatchStatistics], config: SimulationConfig
) -> tuple[str, ...]:
    if not stats:
        return ()
    warnings: list[str] = []
    average_shots = sum(row.shots for row in stats) / len(stats)
    average_goals = sum(row.goals for row in stats) / len(stats)
    average_penalties = sum(row.penalties for row in stats) / len(stats)
    if config.offensive_intensity == 50 and not 55 <= average_shots <= 70:
        warnings.append(
            f"mean shots per match is {average_shots:.2f}; nominal calibration range is 55-70"
        )
    if (
        config.offensive_intensity == 50
        and config.team_imbalance == 30
        and math.isclose(config.home_advantage, 0.05)
        and not 4 <= average_goals <= 8
    ):
        warnings.append(
            f"mean goals per match is {average_goals:.2f}; documented baseline range is 4-8"
        )
    penalty_tolerance = 3 * math.sqrt(config.penalties_per_match / len(stats))
    if abs(average_penalties - config.penalties_per_match) > penalty_tolerance:
        warnings.append(
            f"mean penalties per match is {average_penalties:.2f}; "
            f"configured Poisson expectation is {config.penalties_per_match:.2f} "
            f"(three-standard-error tolerance ±{penalty_tolerance:.2f})"
        )
    return tuple(warnings)


def _check_unique_ids(name: str, identifiers, errors: list[str]) -> None:
    duplicates = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicates:
        errors.append(f"{name} values must be unique; duplicates: {duplicates[:5]}")
