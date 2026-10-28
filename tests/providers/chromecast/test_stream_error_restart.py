"""
Tests for restarting the current track after a Cast receiver drops the flow stream on an error.

Regression tests for https://github.com/music-assistant/support/issues/6512. A Google Home
Speaker failed to decode the flow stream mid-track (MEDIA_DECODE, error 102), went
IDLE/ERROR and loaded the 'next' command item queued behind the stream by itself. Music
Assistant took that fetch for a press on Next, so the rest of the track was skipped (and in
the last track the queue jumped back, support #6658). The interrupted track must be loaded
again where it was, and the command fetch of the dropped session must be ignored.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import PlaybackState

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.providers.chromecast.constants import MASS_APP_ID
from music_assistant.providers.chromecast.player import ChromecastPlayer

if TYPE_CHECKING:
    from music_assistant_models.player_queue import PlayerQueue

COMMAND_URL = "http://192.168.1.58:8097/command/session-1/queue-1/next.mp3"
FLOW_URL = "http://192.168.1.58:8097/flow/session-1/queue-1/item.flac"


# -- Cast player ---------------------------------------------------------------------------


def _fake_player(*, flow_underrun: bool = False) -> Any:
    """
    Build a playing, ungrouped Cast player on mocked collaborators.

    :param flow_underrun: Whether the queue's flow stream has been fully consumed.
    """
    fake = cast("Any", ChromecastPlayer.__new__(ChromecastPlayer))
    fake.mass = MagicMock()
    fake.mass.streams.get_command_url = MagicMock(return_value=COMMAND_URL)
    fake.mass.players.get_active_queue = MagicMock(return_value=MagicMock(queue_id="queue-1"))
    fake.logger = MagicMock()
    fake.cc = MagicMock(app_id=MASS_APP_ID)
    fake.update_state = MagicMock()
    fake._flow_stream_underrun = MagicMock(return_value=flow_underrun)
    fake._media_error_reported = False
    fake._player_id = "cast-1"
    fake._stream_error_handled = False
    fake._app_quit_task_id = "cast_quit_app_test"
    fake.active_cast_group = None
    fake._cache = {"display_name": "Test Cast"}
    fake._attr_playback_state = PlaybackState.PLAYING
    return fake


def _status(*, idle_reason: str | None, content_id: str) -> MagicMock:
    status = MagicMock()
    status.content_id = content_id
    status.player_is_playing = idle_reason is None
    status.player_is_paused = False
    status.player_is_idle = idle_reason is not None
    status.idle_reason = idle_reason
    return status


def test_decode_error_moving_on_to_the_command_item_restarts_the_track() -> None:
    """IDLE/ERROR naming the command item (the C30 shape) restarts the current track."""
    fake = _fake_player()

    fake._handle_media_status(_status(idle_reason="ERROR", content_id=COMMAND_URL))

    fake.mass.player_queues.restart_after_stream_error.assert_called_once_with("queue-1")


def test_restart_is_requested_before_the_idle_state_is_published() -> None:
    """The queue must still hold the playing position when the restart is requested."""
    fake = _fake_player()
    states: list[PlaybackState] = []
    fake.mass.player_queues.restart_after_stream_error.side_effect = lambda _qid: states.append(
        fake._attr_playback_state
    )

    fake._handle_media_status(_status(idle_reason="ERROR", content_id=COMMAND_URL))

    assert states == [PlaybackState.PLAYING]


def test_repeated_error_status_restarts_once() -> None:
    """The receiver repeats the error status; only the first one restarts the track."""
    fake = _fake_player()

    for _ in range(3):
        fake._handle_media_status(_status(idle_reason="ERROR", content_id=COMMAND_URL))

    fake.mass.player_queues.restart_after_stream_error.assert_called_once()


@pytest.mark.parametrize(
    ("idle_reason", "content_id", "flow_underrun"),
    [
        # an error on the stream itself, without moving on (e.g. a failed LOAD)
        ("ERROR", FLOW_URL, False),
        # the device's own Next button: the stream is interrupted, not in error
        ("INTERRUPTED", COMMAND_URL, False),
        # the flow stream was fully played: the end of the queue, not a dropped stream
        ("ERROR", COMMAND_URL, True),
    ],
)
def test_other_idle_statuses_do_not_restart(
    idle_reason: str, content_id: str, flow_underrun: bool
) -> None:
    """Only a dropped stream followed by the command item restarts the track."""
    fake = _fake_player(flow_underrun=flow_underrun)

    fake._handle_media_status(_status(idle_reason=idle_reason, content_id=content_id))

    fake.mass.player_queues.restart_after_stream_error.assert_not_called()


# -- player queues controller --------------------------------------------------------------


def _fake_queues(*, state: PlaybackState = PlaybackState.PLAYING, duration: int = 200) -> Any:
    """Build a controller stand-in owning one playing queue, 37 s into its current item."""

    def _close_coro(coro: Any, **_kwargs: Any) -> None:
        coro.close()

    queue = MagicMock(queue_id="queue-1", state=state, corrected_elapsed_time=37.6)
    queue.current_item = MagicMock(queue_item_id="item-1", duration=duration)
    queue.current_item.name = "Track 1"
    data = PlayerQueueData(queue=cast("PlayerQueue", queue))
    data.session_id = "session-1"
    fake = MagicMock()
    fake._queue_data = {"queue-1": data}
    fake.queue_data_or_none = MagicMock(side_effect=lambda qid: fake._queue_data.get(qid))
    fake.mass.create_task = MagicMock(side_effect=_close_coro)
    return fake


def _restart(fake: Any) -> bool:
    return PlayerQueuesController.restart_after_stream_error(
        cast("PlayerQueuesController", fake), "queue-1"
    )


def test_restart_reloads_the_current_item_at_its_position() -> None:
    """The interrupted item is played again from where it was, and the session is marked."""
    fake = _fake_queues()

    assert _restart(fake) is True

    fake.play_index.assert_called_once_with("queue-1", "item-1", seek_position=37)
    fake.mass.create_task.assert_called_once()
    assert fake._queue_data["queue-1"].errored_session_id == "session-1"


def test_restart_of_an_item_without_duration_starts_it_over() -> None:
    """A radio stream cannot be sought, so it is loaded again from its start."""
    fake = _fake_queues(duration=0)

    assert _restart(fake) is True

    fake.play_index.assert_called_once_with("queue-1", "item-1", seek_position=0)


def test_no_restart_when_the_queue_is_not_playing() -> None:
    """A stopped or paused queue is left alone."""
    fake = _fake_queues(state=PlaybackState.IDLE)

    assert _restart(fake) is False

    fake.mass.create_task.assert_not_called()
    assert fake._queue_data["queue-1"].errored_session_id is None


def test_an_item_failing_again_right_away_is_not_restarted_again() -> None:
    """A stream that keeps failing must not be restarted over and over."""
    fake = _fake_queues()
    assert _restart(fake) is True
    fake._queue_data["queue-1"].session_id = "session-2"

    assert _restart(fake) is False

    fake.mass.create_task.assert_called_once()
    assert fake._queue_data["queue-1"].errored_session_id == "session-1"


def test_an_item_failing_again_much_later_is_restarted_again() -> None:
    """The cooldown only covers errors that follow each other closely."""
    fake = _fake_queues()
    fake._queue_data["queue-1"].last_error_restart = ("item-1", time.time() - 120)

    assert _restart(fake) is True
