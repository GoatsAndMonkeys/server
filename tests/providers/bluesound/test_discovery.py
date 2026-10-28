"""
Tests for reaching BluOS players at the address they announce over mDNS.

A player on a LAN with an IPv6 prefix (e.g. a ULA from a Thread border router) can
re-announce itself with only its IPv6 address while it reconnects.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from ipaddress import ip_address
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from yarl import URL
from zeroconf import ServiceStateChange
from zeroconf.asyncio import AsyncServiceInfo

from music_assistant.providers.bluesound.const import MUSP_MDNS_TYPE
from music_assistant.providers.bluesound.player import BluesoundPlayer
from music_assistant.providers.bluesound.provider import (
    BluesoundDiscoveryInfo,
    BluesoundPlayerProvider,
)

SERVICE_TYPE = "_musc._tcp.local."
SERVICE_NAME = f"Kitchen.{SERVICE_TYPE}"
ZONE_SERVICE_NAME = f"Kitchen-2.{MUSP_MDNS_TYPE}"
MAC = "90:56:82:41:be:04"
IPV4 = "192.168.13.229"
IPV6_ULA = "fdec:7249:300f:4622:9256:82ff:fe41:be04"
PROVIDER_MODULE = "music_assistant.providers.bluesound.provider"


@pytest.fixture
def new_player() -> Iterator[MagicMock]:
    """Record players the provider (re-)creates, without contacting a device."""
    with (
        patch(f"{PROVIDER_MODULE}.BluesoundPlayer") as new_player,
        patch.dict(BluesoundPlayerProvider.player_map),
    ):
        new_player.return_value.setup = AsyncMock()
        yield new_player


@pytest.mark.parametrize(
    ("address", "expected_url"),
    [(IPV4, f"http://{IPV4}:11000"), (IPV6_ULA, f"http://[{IPV6_ULA}]:11000")],
)
def test_client_url_is_valid_for_the_player_address(address: str, expected_url: str) -> None:
    """An IPv6 address must be bracketed in the URL the BluOS client requests."""
    player = BluesoundPlayer(MagicMock(), MAC, _discovery_info(address), "Kitchen", address, 11000)

    url = URL(f"{player.client.base_url}/Status")

    assert player.client.base_url == expected_url
    assert url.host == address
    assert url.port == 11000


@pytest.mark.parametrize("reachable", [True, False])
async def test_ipv6_only_reannouncement_keeps_the_ipv4_player(
    new_player: MagicMock, reachable: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """A reconnect announced over IPv6 alone only refreshes the player at its IPv4 address."""
    known_player = _known_player(IPV4, reachable=reachable)
    provider = await _provider(known_player)

    with caplog.at_level(logging.DEBUG, logger=provider.logger.name):
        await provider.on_mdns_service_state_change(
            SERVICE_NAME, ServiceStateChange.Updated, _service_info(IPV6_ULA)
        )

    assert new_player.call_count == 0
    assert known_player.ip_address == IPV4
    assert known_player.refreshed_urls == [f"http://{IPV4}:11000"]  # type: ignore[attr-defined]
    assert ("Player back online" in caplog.text) is reachable


async def test_ipv6_only_zone_reannouncement_keeps_the_known_zone(
    new_player: MagicMock,
) -> None:
    """A multi-zone player is still recognised when its MAC cannot be looked up over IPv6."""
    zone_id = f"{MAC}:11000"
    known_player = _known_player(IPV4, player_id=zone_id)
    provider = await _provider(known_player)

    with patch(f"{PROVIDER_MODULE}.get_mac_address", AsyncMock(return_value=MAC)) as mac_lookup:
        for address in (IPV4, IPV6_ULA):
            info = _service_info(address, service_type=MUSP_MDNS_TYPE, name=ZONE_SERVICE_NAME)
            await provider.on_mdns_service_state_change(
                ZONE_SERVICE_NAME, ServiceStateChange.Updated, info
            )

    mac_lookup.assert_awaited_once_with(IPV4)
    assert new_player.call_count == 0
    assert known_player.refreshed_urls == [f"http://{IPV4}:11000"] * 2  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("known_address", "announced"),
    [(IPV4, ("192.168.13.230",)), (IPV6_ULA, (IPV4, IPV6_ULA)), (IPV6_ULA, ("fdec::2",))],
)
async def test_changed_address_sets_up_the_player_again(
    new_player: MagicMock, known_address: str, announced: tuple[str, ...]
) -> None:
    """A real address change, or IPv4 becoming available, still re-creates the player."""
    known_player = _known_player(known_address)
    provider = await _provider(known_player)
    BluesoundPlayerProvider.player_map[(known_address, 11000)] = MAC

    await provider.on_mdns_service_state_change(
        SERVICE_NAME, ServiceStateChange.Updated, _service_info(*announced)
    )

    new_player.assert_called_once()
    assert new_player.call_args.args[4] == announced[0]
    new_player.return_value.setup.assert_awaited_once()
    assert known_player.refreshed_urls == []  # type: ignore[attr-defined]
    assert BluesoundPlayerProvider.player_map == {(announced[0], 11000): MAC}


async def test_new_player_prefers_ipv4(new_player: MagicMock) -> None:
    """A new player is set up at its IPv4 address when the announcement has both."""
    provider = await _provider(None)

    await provider.on_mdns_service_state_change(
        SERVICE_NAME, ServiceStateChange.Added, _service_info(IPV6_ULA, IPV4)
    )

    assert new_player.call_args.args[4] == IPV4
    assert BluesoundPlayerProvider.player_map[(IPV4, 11000)] == MAC


def _discovery_info(address: str) -> BluesoundDiscoveryInfo:
    """Create the discovery info the provider hands to a new player."""
    return BluesoundDiscoveryInfo(
        _objectType="", ip_address=address, port="11000", mac=MAC, model="N130"
    )


def _service_info(
    *addresses: str, service_type: str = SERVICE_TYPE, name: str = SERVICE_NAME
) -> AsyncServiceInfo:
    """Create a BluOS mDNS announcement carrying the given addresses."""
    return AsyncServiceInfo(
        service_type,
        name,
        port=11000,
        addresses=[ip_address(address).packed for address in addresses],
        properties={"mac": MAC, "model": "N130"},
    )


def _known_player(address: str, player_id: str = MAC, reachable: bool = True) -> BluesoundPlayer:
    """Create a registered player whose refresh records the URL it would request."""
    player = BluesoundPlayer(
        MagicMock(), player_id, _discovery_info(address), "Kitchen", address, 11000
    )
    player.refreshed_urls = []  # type: ignore[attr-defined]

    async def _refresh() -> None:
        player.refreshed_urls.append(player.client.base_url)  # type: ignore[attr-defined]
        player._attr_available = reachable

    player.update_attributes = AsyncMock(side_effect=_refresh)  # type: ignore[method-assign]
    return player


async def _provider(known_player: BluesoundPlayer | None) -> BluesoundPlayerProvider:
    """Create a Bluesound provider that already knows the given player."""
    provider = BluesoundPlayerProvider.__new__(BluesoundPlayerProvider)
    provider.mass = MagicMock()
    provider.mass.players.get_player.side_effect = lambda player_id: (
        known_player if known_player and player_id == known_player.player_id else None
    )
    provider.logger = logging.getLogger("test.bluesound.discovery")
    await provider.handle_async_init()
    return provider
