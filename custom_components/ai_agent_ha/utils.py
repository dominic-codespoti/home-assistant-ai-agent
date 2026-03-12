"""Utilities for AI Agent HA."""
from __future__ import annotations

import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, SYSTEM_ENTRY_UNIQUE_ID

_LOGGER = logging.getLogger(__name__)

def get_system_entry(hass: HomeAssistant) -> ConfigEntry | None:
    """Get the system configuration entry."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID:
            return entry
    return None
