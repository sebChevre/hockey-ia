from hockeyai import (
    Event,
    EventType,
    ManpowerSituation,
    Match,
    PenaltyEndReason,
    PenaltyType,
)
from hockeyai.geometry import period_for_elapsed_seconds
from hockeyai.validation import _check_match_events


HOME_TEAM = "HOME"
AWAY_TEAM = "AWAY"


def _event(
    sequence: int,
    elapsed: int,
    event_type: EventType,
    *,
    team_id: str | None = None,
    player_id: str | None = None,
    situation: ManpowerSituation = ManpowerSituation(),
    home_score: int = 0,
    away_score: int = 0,
    penalty_type: PenaltyType | None = None,
    related_penalty_event_id: str | None = None,
    penalty_end_reason: PenaltyEndReason | None = None,
) -> Event:
    is_faceoff = event_type is EventType.FACEOFF
    return Event(
        event_id=f"EVENT-{sequence:03d}",
        match_id="MATCH-1",
        event_sequence=sequence,
        elapsed_seconds=elapsed,
        period=period_for_elapsed_seconds(elapsed),
        event_type=event_type,
        team_id=team_id,
        player_id=player_id,
        x=None,
        y=None,
        related_shot_id=None,
        home_score_before=home_score,
        away_score_before=away_score,
        manpower_situation=situation,
        faceoff_winner_player_id="HOME-FORWARD" if is_faceoff else None,
        faceoff_loser_player_id="AWAY-FORWARD" if is_faceoff else None,
        penalty_type=penalty_type,
        penalty_duration_seconds=(
            penalty_type.duration_seconds if event_type is EventType.PENALTY else None
        ),
        related_penalty_event_id=related_penalty_event_id,
        penalty_end_reason=penalty_end_reason,
    )


def _assert_valid_match_events(events: list[Event], *, home_score: int = 0) -> None:
    errors: list[str] = []
    _check_match_events(
        Match("MATCH-1", HOME_TEAM, AWAY_TEAM, home_score, 0, 42),
        events,
        errors,
    )
    assert errors == []


def test_third_penalty_waits_for_a_service_slot_without_dropping_below_three():
    five_v_five = ManpowerSituation(5, 5)
    five_v_four = ManpowerSituation(5, 4)
    five_v_three = ManpowerSituation(5, 3)
    events = [
        _event(1, 0, EventType.FACEOFF, situation=five_v_five),
        _event(
            2,
            100,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-1",
            penalty_type=PenaltyType.MINOR,
            situation=five_v_five,
        ),
        _event(
            3,
            100,
            EventType.PENALTY_START,
            team_id=AWAY_TEAM,
            player_id="AWAY-1",
            related_penalty_event_id="EVENT-002",
            situation=five_v_four,
        ),
        _event(
            4,
            101,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-2",
            penalty_type=PenaltyType.MAJOR,
            situation=five_v_four,
        ),
        _event(
            5,
            101,
            EventType.PENALTY_START,
            team_id=AWAY_TEAM,
            player_id="AWAY-2",
            related_penalty_event_id="EVENT-004",
            situation=five_v_three,
        ),
        _event(
            6,
            102,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-3",
            penalty_type=PenaltyType.MINOR,
            situation=five_v_three,
        ),
        _event(
            7,
            220,
            EventType.PENALTY_END,
            team_id=AWAY_TEAM,
            player_id="AWAY-1",
            related_penalty_event_id="EVENT-002",
            penalty_end_reason=PenaltyEndReason.TIME_EXPIRED,
            situation=five_v_four,
        ),
        _event(
            8,
            220,
            EventType.PENALTY_START,
            team_id=AWAY_TEAM,
            player_id="AWAY-3",
            related_penalty_event_id="EVENT-006",
            situation=five_v_three,
        ),
        _event(
            9,
            340,
            EventType.PENALTY_END,
            team_id=AWAY_TEAM,
            player_id="AWAY-3",
            related_penalty_event_id="EVENT-006",
            penalty_end_reason=PenaltyEndReason.TIME_EXPIRED,
            situation=five_v_four,
        ),
        _event(
            10,
            401,
            EventType.PENALTY_END,
            team_id=AWAY_TEAM,
            player_id="AWAY-2",
            related_penalty_event_id="EVENT-004",
            penalty_end_reason=PenaltyEndReason.TIME_EXPIRED,
            situation=five_v_five,
        ),
        _event(11, 1200, EventType.FACEOFF, situation=five_v_five),
        _event(12, 2400, EventType.FACEOFF, situation=five_v_five),
    ]

    _assert_valid_match_events(events)


def test_power_play_goal_ends_one_minor_and_starts_the_queued_penalty():
    five_v_five = ManpowerSituation(5, 5)
    five_v_four = ManpowerSituation(5, 4)
    five_v_three = ManpowerSituation(5, 3)
    events = [
        _event(1, 0, EventType.FACEOFF, situation=five_v_five),
        _event(
            2,
            100,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-1",
            penalty_type=PenaltyType.MINOR,
            situation=five_v_five,
        ),
        _event(
            3,
            100,
            EventType.PENALTY_START,
            team_id=AWAY_TEAM,
            player_id="AWAY-1",
            related_penalty_event_id="EVENT-002",
            situation=five_v_four,
        ),
        _event(
            4,
            101,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-2",
            penalty_type=PenaltyType.MAJOR,
            situation=five_v_four,
        ),
        _event(
            5,
            101,
            EventType.PENALTY_START,
            team_id=AWAY_TEAM,
            player_id="AWAY-2",
            related_penalty_event_id="EVENT-004",
            situation=five_v_three,
        ),
        _event(
            6,
            102,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-3",
            penalty_type=PenaltyType.MINOR,
            situation=five_v_three,
        ),
        _event(
            7,
            120,
            EventType.GOAL,
            team_id=HOME_TEAM,
            player_id="HOME-1",
            situation=five_v_three,
        ),
        _event(
            8,
            120,
            EventType.PENALTY_END,
            team_id=AWAY_TEAM,
            player_id="AWAY-1",
            related_penalty_event_id="EVENT-002",
            penalty_end_reason=PenaltyEndReason.GOAL,
            situation=five_v_four,
            home_score=1,
        ),
        _event(
            9,
            120,
            EventType.PENALTY_START,
            team_id=AWAY_TEAM,
            player_id="AWAY-3",
            related_penalty_event_id="EVENT-006",
            situation=five_v_three,
            home_score=1,
        ),
        _event(
            10,
            120,
            EventType.FACEOFF,
            situation=five_v_three,
            home_score=1,
        ),
        _event(
            11,
            240,
            EventType.PENALTY_END,
            team_id=AWAY_TEAM,
            player_id="AWAY-3",
            related_penalty_event_id="EVENT-006",
            penalty_end_reason=PenaltyEndReason.TIME_EXPIRED,
            situation=five_v_four,
            home_score=1,
        ),
        _event(
            12,
            401,
            EventType.PENALTY_END,
            team_id=AWAY_TEAM,
            player_id="AWAY-2",
            related_penalty_event_id="EVENT-004",
            penalty_end_reason=PenaltyEndReason.TIME_EXPIRED,
            situation=five_v_five,
            home_score=1,
        ),
        _event(13, 1200, EventType.FACEOFF, situation=five_v_five, home_score=1),
        _event(14, 2400, EventType.FACEOFF, situation=five_v_five, home_score=1),
    ]

    _assert_valid_match_events(events, home_score=1)


def test_active_and_queued_penalties_may_extend_beyond_regulation():
    five_v_five = ManpowerSituation(5, 5)
    five_v_four = ManpowerSituation(5, 4)
    five_v_three = ManpowerSituation(5, 3)
    events = [
        _event(1, 0, EventType.FACEOFF, situation=five_v_five),
        _event(2, 1200, EventType.FACEOFF, situation=five_v_five),
        _event(3, 2400, EventType.FACEOFF, situation=five_v_five),
        _event(
            4,
            3500,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-1",
            penalty_type=PenaltyType.MINOR,
            situation=five_v_five,
        ),
        _event(
            5,
            3500,
            EventType.PENALTY_START,
            team_id=AWAY_TEAM,
            player_id="AWAY-1",
            related_penalty_event_id="EVENT-004",
            situation=five_v_four,
        ),
        _event(
            6,
            3501,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-2",
            penalty_type=PenaltyType.MAJOR,
            situation=five_v_four,
        ),
        _event(
            7,
            3501,
            EventType.PENALTY_START,
            team_id=AWAY_TEAM,
            player_id="AWAY-2",
            related_penalty_event_id="EVENT-006",
            situation=five_v_three,
        ),
        _event(
            8,
            3502,
            EventType.PENALTY,
            team_id=AWAY_TEAM,
            player_id="AWAY-3",
            penalty_type=PenaltyType.MINOR,
            situation=five_v_three,
        ),
    ]

    _assert_valid_match_events(events)
