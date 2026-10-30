"""
Tests for what a failed start on a queue stops.

A play on a member of a playing group loads its item while the player still plays the
group's stream. When that load fails, for example because the group holds the only stream
slot of the provider, stopping the player would silence the whole group.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.models.music_provider import MusicProvider, ProviderStreamLimitError

QUEUE_ID = "member"


def _capacity_error() -> ProviderStreamLimitError:
    """Return the error a provider with its only stream slot taken raises."""
    provider = MagicMock(spec=MusicProvider)
    provider.max_concurrent_streams = 1
    provider.name = "Spotify"
    provider.instance_id = "spotify--1"
    return ProviderStreamLimitError(provider, 15)


class _Harness(NamedTuple):
    """The controller under test and the mocks it reports to."""

    ctrl: PlayerQueuesController
    queue_data: PlayerQueueData
    device_stop: AsyncMock
    device_play: AsyncMock
    audio_cleanup: Mock


def _controller(active_group: str | None = None, synced_to: str | None = None) -> _Harness:
    """
    Build a bare controller with one queue, whose player is in the given group.

    :param active_group: The group player whose stream the player plays, if any.
    :param synced_to: The player the player is synced to, if any.
    """
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    # the queue counts itself active, as it does while its player reports no source
    queue = PlayerQueue(
        queue_id=QUEUE_ID, active=True, display_name="Member", available=True, items=1
    )
    queue_data = PlayerQueueData(queue=queue)
    queue_data.items = [QueueItem(queue_id=QUEUE_ID, queue_item_id="i1", name="i1", duration=200)]
    ctrl._queue_data = {QUEUE_ID: queue_data}
    ctrl.logger = MagicMock()
    ctrl.signal_update = Mock()  # type: ignore[method-assign]
    ctrl.on_player_update = Mock()  # type: ignore[method-assign]
    ctrl._check_player_permission = Mock()  # type: ignore[method-assign]
    ctrl._set_transitioning = Mock()  # type: ignore[method-assign]
    ctrl._get_resume_position = AsyncMock(return_value=0)  # type: ignore[method-assign]
    ctrl._get_next_index = Mock(return_value=None)  # type: ignore[method-assign]
    audio_cleanup = Mock()
    ctrl._cleanup_queue_audio_data = audio_cleanup  # type: ignore[method-assign]
    ctrl.mass = MagicMock()
    ctrl.mass.players.get_player.return_value = SimpleNamespace(
        player_id=QUEUE_ID,
        state=SimpleNamespace(active_group=active_group, synced_to=synced_to),
    )
    device_stop = AsyncMock()
    device_play = AsyncMock()
    ctrl.mass.players._handle_cmd_stop = device_stop
    ctrl.mass.players.play_media = device_play

    @contextlib.asynccontextmanager
    async def _no_lock(*_args: Any, **_kwargs: Any) -> AsyncIterator[None]:
        """Stand in for the playback lock the play and stop actions take."""
        yield

    ctrl.mass.players.get_group_and_player_lock = _no_lock
    return _Harness(ctrl, queue_data, device_stop, device_play, audio_cleanup)


async def _fail_start(harness: _Harness, failure: str) -> None:
    """
    Start the queue so that it fails at the given point, and expect its error.

    :param failure: stream-limit (provider at capacity), no-more-tracks (the item cannot be
        played and none follows) or no-item (the index has no item).
    """
    ctrl = harness.ctrl
    if failure == "stream-limit":
        ctrl._load_item = AsyncMock(side_effect=_capacity_error())  # type: ignore[method-assign]
        with pytest.raises(ProviderStreamLimitError):
            await ctrl.play_index(QUEUE_ID, 0)
    elif failure == "no-more-tracks":
        ctrl._load_item = AsyncMock(side_effect=MediaNotFoundError("unplayable"))  # type: ignore[method-assign]
        with pytest.raises(MediaNotFoundError, match="no more tracks"):
            await ctrl.play_index(QUEUE_ID, 0)
    else:
        ctrl._load_item = AsyncMock()  # type: ignore[method-assign]
        with pytest.raises(MediaNotFoundError, match="No playable item"):
            await ctrl.play_index(QUEUE_ID, 5)


FAILURES = ["stream-limit", "no-more-tracks", "no-item"]


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize(
    ("active_group", "synced_to"),
    [("syncgroup_casa", None), (None, "leader")],
    ids=["group-member", "synced-member"],
)
async def test_failed_start_leaves_the_group_a_member_plays_alone(
    failure: str, active_group: str | None, synced_to: str | None
) -> None:
    """A member still playing its group's stream is not stopped when its own start fails."""
    harness = _controller(active_group=active_group, synced_to=synced_to)

    await _fail_start(harness, failure)

    harness.device_stop.assert_not_awaited()
    harness.device_play.assert_not_awaited()
    # the session the failed start opened still ends, with whatever audio it attached
    assert harness.queue_data.session_id is None
    harness.audio_cleanup.assert_called_once()
    assert harness.audio_cleanup.call_args.args[0] == QUEUE_ID


@pytest.mark.parametrize("failure", FAILURES)
async def test_failed_start_still_stops_a_player_outside_a_group(failure: str) -> None:
    """A player that is in no group is stopped on a failed start, as before."""
    harness = _controller()

    await _fail_start(harness, failure)

    harness.device_stop.assert_awaited_once_with(QUEUE_ID)
    assert harness.queue_data.session_id is None
    harness.audio_cleanup.assert_called_once()
