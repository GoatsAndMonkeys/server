"""Tests for re-reading the DLNA transport state while the device keeps sending events."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from music_assistant.providers.dlna.player import DLNAPlayer
from tests.common import MockProvider


@pytest.mark.asyncio
async def test_poll_rereads_transport_state_despite_recent_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that keeps eventing must still get its transport state polled every minute."""
    provider = MockProvider("dlna", instance_id="dlna_test")
    provider.mass.streams.base_url = "http://192.168.1.2:8097"

    device = MagicMock()
    device.profile_device.available = True
    device.name = "Onkyo receiver"
    device.async_update = AsyncMock()

    player = DLNAPlayer(
        provider,  # type: ignore[arg-type]
        "uuid:onkyo-player",
        "http://192.168.1.10/description.xml",
        device=device,
    )

    now = time.time() + 120
    # a RenderingControl event arrived 10 seconds ago, none for the transport
    player.last_seen = now - 10
    monkeypatch.setattr("music_assistant.providers.dlna.player.time.time", lambda: now)

    await player.poll()

    device.async_update.assert_awaited_once_with(do_ping=True)
