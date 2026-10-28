"""Bluesound Player Provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypedDict, cast

from zeroconf import ServiceStateChange

from music_assistant.helpers.util import (
    get_mac_address,
    get_port_from_zeroconf,
    get_primary_ip_address_from_zeroconf,
    is_ipv6_address,
)
from music_assistant.models.player_provider import PlayerProvider

from .const import MUSP_MDNS_TYPE
from .player import BluesoundPlayer

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry
    from zeroconf.asyncio import AsyncServiceInfo


class BluesoundDiscoveryInfo(TypedDict):
    """Template for MDNS discovery info."""

    _objectType: str
    ip_address: str
    port: str
    mac: str
    model: str


class BluesoundPlayerProvider(PlayerProvider):
    """Bluos compatible player provider, providing support for bluesound speakers."""

    player_map: ClassVar[dict[tuple[str, int], str]] = {}
    _zone_macs: dict[str, str]

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Set up legacy BluOS devices."""
        return ()

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self._zone_macs = {}

    async def on_mdns_service_state_change(
        self, name: str, state_change: ServiceStateChange, info: AsyncServiceInfo | None
    ) -> None:
        """Handle MDNS service state callback for BluOS."""
        if state_change == ServiceStateChange.Removed:
            # Wait for connection to fail, same as sonos.
            return
        name = name.split(".", 1)[0]
        assert info is not None

        ip_address = get_primary_ip_address_from_zeroconf(info)
        port = get_port_from_zeroconf(info)

        if not ip_address or not port:
            self.logger.debug("Ignoring incomplete mdns discovery for Bluesound player: %s", name)
            return

        player_id: str | None
        mac_address: str | None
        if info.type == MUSP_MDNS_TYPE:
            # this is a multi-zone device, we need to fetch the mac address of the main device.
            # That lookup only works over IPv4, so an IPv6-only announcement of a known zone
            # reuses the mac address found for it before.
            if is_ipv6_address(ip_address):
                mac_address = self._zone_macs.get(info.name)
            else:
                mac_address = await get_mac_address(ip_address)
            player_id = f"{mac_address}:{port}"
        else:
            mac_address = info.decoded_properties.get("mac")
            player_id = mac_address

        if not mac_address:
            self.logger.debug(
                "Ignoring mdns discovery for Bluesound player without MAC address: %s",
                name,
            )
            return

        # Handle update of existing player
        assert player_id is not None  # for type checker
        if info.type == MUSP_MDNS_TYPE:
            self._zone_macs[info.name] = mac_address
        if bluos_player := self.mass.players.get_player(player_id):
            bluos_player = cast("BluesoundPlayer", bluos_player)
            # An announcement that only carries an IPv6 address (e.g. while the device
            # re-announces after a network blip) does not move a player known at an IPv4
            # address: BluOS groups players by their IPv4 address.
            if ip_address == bluos_player.ip_address or (
                is_ipv6_address(ip_address) and not is_ipv6_address(bluos_player.ip_address)
            ):
                # Refresh the player at the address it already has
                bluos_player._attr_available = True
                await bluos_player.update_attributes()
                if bluos_player._attr_available:
                    self.logger.debug("Player back online: %s", bluos_player.name)
                return
            self.logger.debug(
                "IP address for player %s updated to %s", bluos_player.name, ip_address
            )
            self.player_map.pop((bluos_player.ip_address, bluos_player.port), None)

        # New player discovered
        self.logger.debug("Discovered player: %s", name)

        discovery_info = BluesoundDiscoveryInfo(
            _objectType=info.decoded_properties.get("_objectType") or "",
            ip_address=ip_address,
            port=str(port),
            mac=mac_address,
            model=info.decoded_properties.get("model") or "",
        )

        # Create BluOS player
        bluos_player = BluesoundPlayer(self, player_id, discovery_info, name, ip_address, port)
        self.player_map[(ip_address, port)] = player_id

        # Register with Music Assistant
        await bluos_player.setup()
