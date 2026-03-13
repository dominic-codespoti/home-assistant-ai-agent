import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import aiohttp
import yaml  # type: ignore[import-untyped]
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_WEATHER_ENTITY, DOMAIN
from .utils import sanitize_for_logging
from .models import (
    BaseAIClient,
    LocalClient,
    LlamaClient,
    OpenAIClient,
    GeminiClient,
    AnthropicClient,
    OpenRouterClient,
    AlterClient,
    ZaiClient,
)

_LOGGER = logging.getLogger(__name__)


# === Main Agent ===
class AiAgentHaAgent:
    """Agent for handling queries with dynamic data requests and multiple AI providers."""

    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant integrated with Home Assistant.\n"
            "You have access to a set of capability-rich tools via the Model Context Protocol (MCP).\n"
            "Use these tools to discover entities, check their states, and perform actions.\n\n"
            "CORE CAPABILITIES:\n"
            "- 'discover_entities': Find entities by name, domain, area, device_class, or current state (e.g., state='on').\n"
            "- 'get_entity_details': Get the full status and attributes for specific entity IDs.\n"
            "- 'perform_action': Control devices (lights, switches, etc.) by calling Home Assistant services.\n"
            "- 'list_areas' / 'list_domains': Get an overview of how the home is organized.\n"
            "- 'get_index': Get a high-level overview of the entire system structure.\n\n"
            "DASHBOARD & AUTOMATION CREATION:\n"
            "If the user asks to create an automation or dashboard, use the same tool-calling mechanism to gather data, "
            "then respond with the appropriate JSON format in your final response.\n\n"
            "RESPONSE FORMATS:\n"
            "If you are not using a native tool-calling function (e.g., on a local or fallback model), "
            "you must still provide your request as a JSON object with a 'request_type' field.\n"
            "Example: {\"request_type\": \"_mcp_tool_calls\", \"tool_calls\": [{\"name\": \"discover_entities\", \"arguments\": {\"state\": \"on\"}}]}\n\n"
            "Always be concise and helpful. When asked 'what is on?', first discover entities with state 'on'.\n"
            "STRICT RULES:\n"
            "1. You MUST use the provided tools to gather data before providing a final answer.\n"
            "2. If a tool call returns no results, try alternative search terms or broader domains immediately.\n"
            "3. DO NOT explain what you are doing or what you plan to do between tool calls.\n"
            "4. Stay in the tool-calling loop until you have successfully gathered all necessary information or exhausted all search possibilities."
        ),
    }

    SYSTEM_PROMPT_LOCAL = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant integrated with Home Assistant.\n"
            "You must use the provided tools to interact with the environment.\n"
            "If you cannot call tools directly, you MUST respond with a JSON object like:\n"
            "{\"request_type\": \"_mcp_tool_calls\", \"tool_calls\": [{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}]}\n\n"
            "AVAILABLE TOOLS:\n"
            "- 'discover_entities(state, area, domain, device_class)': Search for devices.\n"
            "- 'perform_action(domain, action, target, data)': Control devices.\n"
            "- 'get_entity_details(entity_ids)': Get status of specific devices.\n"
            "- 'get_index()': Get system overview."
        ),
    }

    def __init__(self, hass: HomeAssistant, config: Dict[str, Any]):
        """Initialize the agent with provider selection."""
        self.hass = hass
        self.config = config
        self.conversation_history: List[Dict[str, Any]] = []
        self._cache: Dict[str, Any] = {}
        self.ai_client: BaseAIClient
        self._cache_timeout = 300  # 5 minutes
        self._max_retries = 10
        self._retry_delay = 1  # seconds
        self._rate_limit = 60  # requests per minute
        self._last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()

        provider = config.get("ai_provider", "openai")
        models_config = config.get("models", {})

        _LOGGER.debug("Initializing AiAgentHaAgent with provider: %s", provider)
        _LOGGER.debug("Models config loaded: %s", models_config)

        # Set the appropriate system prompt based on provider
        if provider == "local":
            self.system_prompt = self.SYSTEM_PROMPT_LOCAL
            _LOGGER.debug("Using local-optimized system prompt")
        else:
            self.system_prompt = self.SYSTEM_PROMPT
            _LOGGER.debug("Using standard system prompt")

        # Initialize the appropriate AI client with model selection
        if provider == "openai":
            model = models_config.get("openai", "gpt-3.5-turbo")
            self.ai_client = OpenAIClient(config.get("openai_token"), model)
        elif provider == "gemini":
            model = models_config.get("gemini", "gemini-2.5-flash")
            self.ai_client = GeminiClient(config.get("gemini_token"), model)
        elif provider == "openrouter":
            model = models_config.get("openrouter", "openai/gpt-4o")
            self.ai_client = OpenRouterClient(config.get("openrouter_token"), model)
        elif provider == "anthropic":
            model = models_config.get("anthropic", "claude-sonnet-4-5-20250929")
            self.ai_client = AnthropicClient(config.get("anthropic_token"), model)
        elif provider == "alter":
            model = models_config.get("alter", "")
            self.ai_client = AlterClient(config.get("alter_token"), model)
        elif provider == "zai":
            model = models_config.get("zai", "glm-4.7")
            endpoint_type = config.get("zai_endpoint", "general")
            self.ai_client = ZaiClient(config.get("zai_token"), model, endpoint_type)
        elif provider == "local":
            model = models_config.get("local", "")
            url = config.get("local_url")
            if not url:
                _LOGGER.error("Missing local_url for local provider")
                raise Exception("Missing local_url configuration for local provider")
            self.ai_client = LocalClient(url, model)
        else:  # default to llama if somehow specified
            model = models_config.get("llama", "Llama-4-Maverick-17B-128E-Instruct-FP8")
            self.ai_client = LlamaClient(config.get("llama_token"), model)

        _LOGGER.debug(
            "AiAgentHaAgent initialized successfully with provider: %s, model: %s",
            provider,
            model,
        )

    def _validate_api_key(self) -> bool:
        """Validate the API key format."""
        provider = self.config.get("ai_provider", "openai")

        if provider == "openai":
            token = self.config.get("openai_token")
        elif provider == "gemini":
            token = self.config.get("gemini_token")
        elif provider == "openrouter":
            token = self.config.get("openrouter_token")
        elif provider == "anthropic":
            token = self.config.get("anthropic_token")
        elif provider == "alter":
            token = self.config.get("alter_token")
        elif provider == "zai":
            token = self.config.get("zai_token")
        elif provider == "local":
            token = self.config.get("local_url")
        else:
            token = self.config.get("llama_token")

        if not token or not isinstance(token, str):
            return False

        # For local provider, validate URL format
        if provider == "local":
            return bool(token.startswith(("http://", "https://")))

        # Add more specific validation based on your API key format
        return len(token) >= 32

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        if current_time - self._request_window_start >= 60:
            self._request_count = 0
            self._request_window_start = current_time

        if self._request_count >= self._rate_limit:
            return False

        self._request_count += 1
        return True

    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if it's still valid."""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if time.time() - timestamp < self._cache_timeout:
                return data
            del self._cache[key]
        return None

    def _set_cached_data(self, key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._cache[key] = (time.time(), data)

    def _sanitize_automation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize automation configuration to prevent injection attacks."""
        sanitized: Dict[str, Any] = {}
        for key, value in config.items():
            if key in ["alias", "description"]:
                # Sanitize strings
                sanitized[key] = str(value).strip()[:100]  # Limit length
            elif key in ["trigger", "condition", "action"]:
                # Validate arrays
                if isinstance(value, list):
                    sanitized[key] = value
            elif key == "mode":
                # Validate mode
                if value in ["single", "restart", "queued", "parallel"]:
                    sanitized[key] = value
        return sanitized

    async def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Get the state of a specific entity."""
        try:
            _LOGGER.debug("Requesting entity state for: %s", entity_id)
            state = self.hass.states.get(entity_id)
            if not state:
                _LOGGER.warning("Entity not found: %s", entity_id)
                return {"error": f"Entity {entity_id} not found"}

            # Get area information from entity/device registry
            # Wrapped in try-except to handle cases where registries aren't available (e.g., in tests)
            area_id = None
            area_name = None

            try:
                from homeassistant.helpers import area_registry as ar
                from homeassistant.helpers import device_registry as dr
                from homeassistant.helpers import entity_registry as er

                entity_registry = er.async_get(self.hass)
                device_registry = dr.async_get(self.hass)
                area_registry = ar.async_get(self.hass)

                if entity_registry and hasattr(entity_registry, "async_get"):
                    # Try to find the entity in the registry
                    entity_entry = entity_registry.async_get(entity_id)
                    if entity_entry:
                        _LOGGER.debug("Entity %s found in registry", entity_id)
                        # Check if entity has a direct area assignment
                        if hasattr(entity_entry, "area_id") and entity_entry.area_id:
                            area_id = entity_entry.area_id
                            _LOGGER.debug(
                                "Entity %s has direct area assignment: %s",
                                entity_id,
                                area_id,
                            )
                        # Otherwise check if the entity's device has an area
                        elif (
                            hasattr(entity_entry, "device_id")
                            and entity_entry.device_id
                            and device_registry
                            and hasattr(device_registry, "async_get")
                        ):
                            _LOGGER.debug(
                                "Entity %s has device_id: %s, checking device area",
                                entity_id,
                                entity_entry.device_id,
                            )
                            device_entry = device_registry.async_get(
                                entity_entry.device_id
                            )
                            if device_entry:
                                if (
                                    hasattr(device_entry, "area_id")
                                    and device_entry.area_id
                                ):
                                    area_id = device_entry.area_id
                                    _LOGGER.debug(
                                        "Device %s has area: %s",
                                        entity_entry.device_id,
                                        area_id,
                                    )
                                else:
                                    _LOGGER.debug(
                                        "Device %s has no area assigned",
                                        entity_entry.device_id,
                                    )
                            else:
                                _LOGGER.debug(
                                    "Device %s not found in registry",
                                    entity_entry.device_id,
                                )
                        else:
                            _LOGGER.debug(
                                "Entity %s has no area_id and no device_id", entity_id
                            )
                    else:
                        _LOGGER.debug(
                            "Entity %s not found in entity registry", entity_id
                        )
                else:
                    _LOGGER.debug("Entity registry not available for %s", entity_id)

                # Get area name from area_id
                if (
                    area_id
                    and area_registry
                    and hasattr(area_registry, "async_get_area")
                ):
                    area_entry = area_registry.async_get_area(area_id)
                    if area_entry and hasattr(area_entry, "name"):
                        area_name = area_entry.name
                        _LOGGER.debug(
                            "Resolved area_id %s to area_name: %s", area_id, area_name
                        )
                    else:
                        _LOGGER.debug("Could not resolve area_id %s to name", area_id)
                elif area_id:
                    _LOGGER.debug(
                        "Have area_id %s but area_registry not available", area_id
                    )
            except Exception as e:
                # Registries not available (likely in test environment) - skip area information
                _LOGGER.warning(
                    "Exception retrieving area information for %s: %s",
                    entity_id,
                    str(e),
                )

            result = {
                "entity_id": state.entity_id,
                "state": state.state,
                "last_changed": (
                    state.last_changed.isoformat() if state.last_changed else None
                ),
                "friendly_name": state.attributes.get("friendly_name"),
                "area_id": area_id,
                "area_name": area_name,
                "attributes": {
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in state.attributes.items()
                },
            }
            _LOGGER.debug(
                "Retrieved entity state for %s: area_id=%s, area_name=%s",
                entity_id,
                area_id,
                area_name,
            )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting entity state: %s", str(e))
            return {"error": f"Error getting entity state: {str(e)}"}

    async def get_entities_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all entities for a specific domain."""
        try:
            _LOGGER.debug("Requesting all entities for domain: %s", domain)
            states = [
                state
                for state in self.hass.states.async_all()
                if state.entity_id.startswith(f"{domain}.")
            ]
            _LOGGER.debug("Found %d entities in domain %s", len(states), domain)
            return [await self.get_entity_state(state.entity_id) for state in states]
        except Exception as e:
            _LOGGER.exception("Error getting entities by domain: %s", str(e))
            return [{"error": f"Error getting entities for domain {domain}: {str(e)}"}]

    async def get_entities_by_device_class(
        self, device_class: str, domain: str = None
    ) -> List[Dict[str, Any]]:
        """Get all entities with a specific device_class.

        Args:
            device_class: The device class to filter by (e.g., 'temperature', 'humidity', 'motion')
            domain: Optional domain to restrict search (e.g., 'sensor', 'binary_sensor')

        Returns:
            List of entity state dictionaries that match the device_class
        """
        try:
            _LOGGER.debug(
                "Requesting all entities with device_class: %s (domain: %s)",
                device_class,
                domain or "all",
            )
            matching_entities = []

            for state in self.hass.states.async_all():
                # Filter by domain if specified
                if domain and not state.entity_id.startswith(f"{domain}."):
                    continue

                # Check if this entity has the matching device_class
                entity_device_class = state.attributes.get("device_class")
                if entity_device_class == device_class:
                    matching_entities.append(state.entity_id)

            _LOGGER.debug(
                "Found %d entities with device_class %s",
                len(matching_entities),
                device_class,
            )

            # Get full state information for each matching entity
            return [
                await self.get_entity_state(entity_id)
                for entity_id in matching_entities
            ]

        except Exception as e:
            _LOGGER.exception("Error getting entities by device_class: %s", str(e))
            return [
                {
                    "error": f"Error getting entities with device_class {device_class}: {str(e)}"
                }
            ]

    async def get_climate_related_entities(self) -> List[Dict[str, Any]]:
        """Get all climate-related entities including climate domain and temperature/humidity sensors.

        Returns:
            List of entity state dictionaries for:
            - All climate.* entities (thermostats, HVAC systems)
            - All sensor.* entities with device_class: temperature
            - All sensor.* entities with device_class: humidity
        """
        try:
            _LOGGER.debug("Requesting all climate-related entities")
            climate_entities = []

            # Get all climate domain entities (thermostats, HVAC)
            climate_domain = await self.get_entities_by_domain("climate")
            climate_entities.extend(climate_domain)

            # Get temperature sensors
            temp_sensors = await self.get_entities_by_device_class(
                "temperature", "sensor"
            )
            climate_entities.extend(temp_sensors)

            # Get humidity sensors
            humidity_sensors = await self.get_entities_by_device_class(
                "humidity", "sensor"
            )
            climate_entities.extend(humidity_sensors)

            # Deduplicate by entity_id (edge case: if an entity appears in multiple categories)
            seen_entity_ids = set()
            unique_entities = []
            for entity in climate_entities:
                entity_id = entity.get("entity_id")
                if entity_id and entity_id not in seen_entity_ids:
                    seen_entity_ids.add(entity_id)
                    unique_entities.append(entity)

            _LOGGER.debug(
                "Found %d total climate-related entities (deduplicated from %d)",
                len(unique_entities),
                len(climate_entities),
            )
            return unique_entities

        except Exception as e:
            _LOGGER.exception("Error getting climate-related entities: %s", str(e))
            return [{"error": f"Error getting climate-related entities: {str(e)}"}]

    async def get_entities_by_area(self, area_id: str) -> List[Dict[str, Any]]:
        """Get all entities for a specific area."""
        try:
            _LOGGER.debug("Requesting all entities for area: %s", area_id)

            # Get entity registry to find entities assigned to the area
            from homeassistant.helpers import device_registry as dr
            from homeassistant.helpers import entity_registry as er

            entity_registry = er.async_get(self.hass)
            device_registry = dr.async_get(self.hass)

            entities_in_area = []

            # Find entities assigned to the area (directly or through their device)
            for entity in entity_registry.entities.values():
                # Check if entity is directly assigned to the area
                if entity.area_id == area_id:
                    entities_in_area.append(entity.entity_id)
                # Check if entity's device is assigned to the area
                elif entity.device_id:
                    device = device_registry.devices.get(entity.device_id)
                    if device and device.area_id == area_id:
                        entities_in_area.append(entity.entity_id)

            _LOGGER.debug(
                "Found %d entities in area %s", len(entities_in_area), area_id
            )

            # Get state information for each entity
            result = []
            for entity_id in entities_in_area:
                state_info = await self.get_entity_state(entity_id)
                if not state_info.get("error"):  # Only include entities that exist
                    result.append(state_info)

            return result

        except Exception as e:
            _LOGGER.exception("Error getting entities by area: %s", str(e))
            return [{"error": f"Error getting entities for area {area_id}: {str(e)}"}]

    async def get_entities(self, area_id=None, area_ids=None) -> List[Dict[str, Any]]:
        """Get entities by area(s) - flexible method that supports single area or multiple areas."""
        try:
            # Handle different parameter formats
            areas_to_process = []

            if area_ids:
                # Multiple areas provided
                if isinstance(area_ids, list):
                    areas_to_process = area_ids
                else:
                    areas_to_process = [area_ids]
            elif area_id:
                # Single area provided
                if isinstance(area_id, list):
                    areas_to_process = area_id
                else:
                    areas_to_process = [area_id]
            else:
                return [{"error": "No area_id or area_ids provided"}]

            _LOGGER.debug("Requesting entities for areas: %s", areas_to_process)

            all_entities = []
            for area in areas_to_process:
                entities_in_area = await self.get_entities_by_area(area)
                all_entities.extend(entities_in_area)

            # Remove duplicates based on entity_id
            seen_entities = set()
            unique_entities = []
            for entity in all_entities:
                if isinstance(entity, dict) and "entity_id" in entity:
                    if entity["entity_id"] not in seen_entities:
                        seen_entities.add(entity["entity_id"])
                        unique_entities.append(entity)
                else:
                    unique_entities.append(entity)  # Keep error messages

            _LOGGER.debug(
                "Found %d unique entities across %d areas",
                len(unique_entities),
                len(areas_to_process),
            )
            return unique_entities

        except Exception as e:
            _LOGGER.exception("Error getting entities: %s", str(e))
            return [{"error": f"Error getting entities: {str(e)}"}]

    async def get_calendar_events(
        self, entity_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get calendar events, optionally filtered by entity_id."""
        try:
            if entity_id:
                _LOGGER.debug(
                    "Requesting calendar events for specific entity: %s", entity_id
                )
                return [await self.get_entity_state(entity_id)]

            _LOGGER.debug("Requesting all calendar events")
            return await self.get_entities_by_domain("calendar")
        except Exception as e:
            _LOGGER.exception("Error getting calendar events: %s", str(e))
            return [{"error": f"Error getting calendar events: {str(e)}"}]

    async def get_automations(self) -> List[Dict[str, Any]]:
        """Get all automations."""
        try:
            _LOGGER.debug("Requesting all automations")
            return await self.get_entities_by_domain("automation")
        except Exception as e:
            _LOGGER.exception("Error getting automations: %s", str(e))
            return [{"error": f"Error getting automations: {str(e)}"}]

    async def get_entity_registry(self) -> List[Dict]:
        """Get entity registry entries with device_class and other metadata.

        Area information is resolved from the entity or its device.
        """
        _LOGGER.debug("Requesting all entity registry entries")
        try:
            from homeassistant.helpers import area_registry as ar
            from homeassistant.helpers import device_registry as dr
            from homeassistant.helpers import entity_registry as er

            entity_registry = er.async_get(self.hass)
            if not entity_registry:
                return []

            device_registry = dr.async_get(self.hass)
            area_registry = ar.async_get(self.hass)

            result = []
            for entry in entity_registry.entities.values():
                # Get the current state to access device_class and other attributes
                state = self.hass.states.get(entry.entity_id)
                device_class = state.attributes.get("device_class") if state else None
                state_class = state.attributes.get("state_class") if state else None
                unit_of_measurement = (
                    state.attributes.get("unit_of_measurement") if state else None
                )

                # Resolve area_id and area_name
                # First check entity's direct area assignment
                area_id = entry.area_id
                area_name = None

                # If entity doesn't have area, check device's area
                if not area_id and entry.device_id and device_registry:
                    device_entry = device_registry.async_get(entry.device_id)
                    if device_entry and hasattr(device_entry, "area_id"):
                        area_id = device_entry.area_id

                # Resolve area_name from area_id
                if area_id and area_registry:
                    area_entry = area_registry.async_get_area(area_id)
                    if area_entry and hasattr(area_entry, "name"):
                        area_name = area_entry.name

                result.append(
                    {
                        "entity_id": entry.entity_id,
                        "device_id": entry.device_id,
                        "platform": entry.platform,
                        "disabled": entry.disabled,
                        "area_id": area_id,
                        "area_name": area_name,
                        "original_name": entry.original_name,
                        "unique_id": entry.unique_id,
                        "device_class": device_class,
                        "state_class": state_class,
                        "unit_of_measurement": unit_of_measurement,
                    }
                )

            return result
        except Exception as e:
            _LOGGER.exception("Error getting entity registry entries: %s", str(e))
            return [{"error": f"Error getting entity registry entries: {str(e)}"}]

    async def get_device_registry(self) -> List[Dict]:
        """Get device registry entries"""
        _LOGGER.debug("Requesting all device registry entries")
        try:
            from homeassistant.helpers import device_registry as dr

            registry = dr.async_get(self.hass)
            if not registry:
                return []
            return [
                {
                    "id": device.id,
                    "name": device.name,
                    "model": device.model,
                    "manufacturer": device.manufacturer,
                    "sw_version": device.sw_version,
                    "hw_version": device.hw_version,
                    "connections": (
                        list(device.connections) if device.connections else []
                    ),
                    "identifiers": (
                        list(device.identifiers) if device.identifiers else []
                    ),
                    "area_id": device.area_id,
                    "disabled": device.disabled_by is not None,
                    "entry_type": (
                        device.entry_type.value if device.entry_type else None
                    ),
                    "name_by_user": device.name_by_user,
                }
                for device in registry.devices.values()
            ]
        except Exception as e:
            _LOGGER.exception("Error getting device registry entries: %s", str(e))
            return [{"error": f"Error getting device registry entries: {str(e)}"}]

    async def get_history(self, entity_id: str, hours: int = 24) -> List[Dict]:
        """Get historical state changes for an entity"""
        _LOGGER.debug("Requesting historical state changes for entity: %s", entity_id)
        try:
            from homeassistant.components.recorder.history import get_significant_states

            now = dt_util.utcnow()
            start = now - timedelta(hours=hours)

            # Get history using the recorder history module
            history_data = await self.hass.async_add_executor_job(
                get_significant_states,
                self.hass,
                start,
                now,
                [entity_id],
            )

            # Convert to serializable format
            result = []
            for entity_id_key, states in history_data.items():
                for state in states:
                    # Skip if it's a dict (mypy type narrowing)
                    if isinstance(state, dict):
                        continue
                    result.append(
                        {
                            "entity_id": state.entity_id,
                            "state": state.state,
                            "last_changed": state.last_changed.isoformat(),
                            "last_updated": state.last_updated.isoformat(),
                            "attributes": dict(state.attributes),
                        }
                    )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting history: %s", str(e))
            return [{"error": f"Error getting history: {str(e)}"}]

    async def get_area_registry(self) -> Dict[str, Any]:
        """Get area registry information"""
        _LOGGER.debug("Get area registry information")
        try:
            from homeassistant.helpers import area_registry as ar

            registry = ar.async_get(self.hass)
            if not registry:
                return {}

            result = {}
            for area in registry.areas.values():
                result[area.id] = {
                    "name": area.name,
                    "normalized_name": area.normalized_name,
                    "picture": area.picture,
                    "icon": area.icon,
                    "floor_id": area.floor_id,
                    "labels": list(area.labels) if area.labels else [],
                }
            return result
        except Exception as e:
            _LOGGER.exception("Error getting area registry: %s", str(e))
            return {"error": f"Error getting area registry: {str(e)}"}

    async def get_person_data(self) -> List[Dict]:
        """Get person tracking information"""
        _LOGGER.debug("Requesting person tracking information")
        try:
            result = []
            for state in self.hass.states.async_all("person"):
                result.append(
                    {
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "state": state.state,
                        "latitude": state.attributes.get("latitude"),
                        "longitude": state.attributes.get("longitude"),
                        "source": state.attributes.get("source"),
                        "gps_accuracy": state.attributes.get("gps_accuracy"),
                        "last_changed": (
                            state.last_changed.isoformat()
                            if state.last_changed
                            else None
                        ),
                    }
                )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting person tracking information: %s", str(e))
            return [{"error": f"Error getting person tracking information: {str(e)}"}]

    async def get_statistics(self, entity_id: str) -> Dict:
        """Get statistics for an entity"""
        _LOGGER.debug("Requesting statistics for entity: %s", entity_id)
        try:
            from homeassistant.components import recorder

            # Check if recorder is available
            if not self.hass.data.get(recorder.DATA_INSTANCE):
                return {"error": "Recorder component is not available"}

            # from homeassistant.components.recorder.statistics import get_latest_short_term_statistics
            import homeassistant.components.recorder.statistics as stats_module

            # Get latest statistics
            stats = await self.hass.async_add_executor_job(
                # get_latest_short_term_statistics,
                stats_module.get_last_short_term_statistics,
                self.hass,
                1,
                entity_id,
                True,
                set(),
            )

            if entity_id in stats:
                stat_data = stats[entity_id][0] if stats[entity_id] else {}
                return {
                    "entity_id": entity_id,
                    "start": stat_data.get("start"),
                    "mean": stat_data.get("mean"),
                    "min": stat_data.get("min"),
                    "max": stat_data.get("max"),
                    "last_reset": stat_data.get("last_reset"),
                    "state": stat_data.get("state"),
                    "sum": stat_data.get("sum"),
                }
            else:
                return {"error": f"No statistics available for entity {entity_id}"}
        except Exception as e:
            _LOGGER.exception("Error getting statistics: %s", str(e))
            return {"error": f"Error getting statistics: {str(e)}"}

    async def get_scenes(self) -> List[Dict]:
        """Get scene configurations"""
        _LOGGER.debug("Requesting scene configurations")
        try:
            result = []
            for state in self.hass.states.async_all("scene"):
                result.append(
                    {
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "last_activated": state.attributes.get("last_activated"),
                        "icon": state.attributes.get("icon"),
                        "last_changed": (
                            state.last_changed.isoformat()
                            if state.last_changed
                            else None
                        ),
                    }
                )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting scene configurations: %s", str(e))
            return [{"error": f"Error getting scene configurations: {str(e)}"}]

    async def get_weather_data(self) -> Dict[str, Any]:
        """Get weather data from any available weather entity in the system."""
        try:
            # Find all weather entities
            weather_entities = [
                state
                for state in self.hass.states.async_all()
                if state.domain == "weather"
            ]

            if not weather_entities:
                return {
                    "error": "No weather entities found in the system. Please add a weather integration."
                }

            # Use the first available weather entity
            state = weather_entities[0]
            _LOGGER.debug("Using weather entity: %s", state.entity_id)

            # Get all available attributes
            all_attributes = state.attributes
            _LOGGER.debug(
                "Available weather attributes: %s", json.dumps(all_attributes)
            )

            # Get forecast data
            forecast = all_attributes.get("forecast", [])

            # Process forecast data
            processed_forecast = []
            for day in forecast:
                forecast_entry = {
                    "datetime": day.get("datetime"),
                    "temperature": day.get("temperature"),
                    "condition": day.get("condition"),
                    "precipitation": day.get("precipitation"),
                    "precipitation_probability": day.get("precipitation_probability"),
                    "humidity": day.get("humidity"),
                    "wind_speed": day.get("wind_speed"),
                    "wind_bearing": day.get("wind_bearing"),
                }
                # Only add entries that have at least some data
                if any(v is not None for v in forecast_entry.values()):
                    processed_forecast.append(forecast_entry)

            # Get current weather data
            current = {
                "entity_id": state.entity_id,
                "temperature": all_attributes.get("temperature"),
                "humidity": all_attributes.get("humidity"),
                "pressure": all_attributes.get("pressure"),
                "wind_speed": all_attributes.get("wind_speed"),
                "wind_bearing": all_attributes.get("wind_bearing"),
                "condition": state.state,
                "forecast_available": len(processed_forecast) > 0,
            }

            # Log the processed data for debugging
            _LOGGER.debug(
                "Processed weather data: %s",
                json.dumps(
                    {"current": current, "forecast_count": len(processed_forecast)}
                ),
            )

            return {"current": current, "forecast": processed_forecast}
        except Exception as e:
            _LOGGER.exception("Error getting weather data: %s", str(e))
            return {"error": f"Error getting weather data: {str(e)}"}

    async def create_automation(
        self, automation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new automation with validation and sanitization."""
        try:
            _LOGGER.debug(
                "Creating automation with config: %s", json.dumps(automation_config)
            )

            # Validate required fields
            if not all(
                key in automation_config for key in ["alias", "trigger", "action"]
            ):
                return {"error": "Missing required fields in automation configuration"}

            # Sanitize configuration
            sanitized_config = self._sanitize_automation_config(automation_config)

            # Generate a unique ID for the automation
            automation_id = f"ai_agent_auto_{int(time.time() * 1000)}"

            # Create the automation entry
            automation_entry = {
                "id": automation_id,
                "alias": sanitized_config["alias"],
                "description": sanitized_config.get("description", ""),
                "trigger": sanitized_config["trigger"],
                "condition": sanitized_config.get("condition", []),
                "action": sanitized_config["action"],
                "mode": sanitized_config.get("mode", "single"),
            }

            # Read current automations.yaml using async executor
            automations_path = self.hass.config.path("automations.yaml")
            try:
                current_automations = await self.hass.async_add_executor_job(
                    lambda: yaml.safe_load(open(automations_path, "r")) or []
                )
            except FileNotFoundError:
                current_automations = []

            # Check for duplicate automation names
            if any(
                auto.get("alias") == automation_entry["alias"]
                for auto in current_automations
            ):
                return {
                    "error": f"An automation with the name '{automation_entry['alias']}' already exists"
                }

            # Append new automation
            current_automations.append(automation_entry)

            # Write back to file using async executor
            await self.hass.async_add_executor_job(
                lambda: yaml.dump(
                    current_automations,
                    open(automations_path, "w"),
                    default_flow_style=False,
                )
            )

            # Reload automations
            await self.hass.services.async_call("automation", "reload")

            # Clear automation-related caches
            self._cache.clear()

            return {
                "success": True,
                "message": f"Automation '{automation_entry['alias']}' created successfully",
            }

        except Exception as e:
            _LOGGER.exception("Error creating automation: %s", str(e))
            return {"error": f"Error creating automation: {str(e)}"}

    async def get_dashboards(self) -> List[Dict[str, Any]]:
        """Get list of all dashboards."""
        try:
            _LOGGER.debug("Requesting all dashboards")

            # Get dashboards via WebSocket API
            ws_api = self.hass.data.get("websocket_api")
            if not ws_api:
                return [{"error": "WebSocket API not available"}]

            # Use the lovelace service to get dashboards
            try:
                from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

                # Get lovelace data using property access (required for HA 2026.2+)
                # lovelace_data is a LovelaceData dataclass with a 'dashboards' attribute
                lovelace_data = self.hass.data.get(LOVELACE_DOMAIN)
                if lovelace_data is None:
                    return [{"error": "Lovelace not available"}]

                # Safety check for dashboards attribute (backward compatibility)
                if not hasattr(lovelace_data, "dashboards"):
                    return [{"error": "Lovelace dashboards not available"}]

                # Use property access instead of dictionary access
                dashboards = lovelace_data.dashboards

                # Get YAML dashboard configs for metadata (title, icon, etc.)
                # yaml_dashboards contains the configuration with metadata
                yaml_configs = getattr(lovelace_data, "yaml_dashboards", {}) or {}

                dashboard_list = []

                # Iterate over all dashboards (None key = default dashboard)
                for url_path, dashboard_obj in dashboards.items():
                    # Try to get metadata from yaml_dashboards first
                    yaml_config = yaml_configs.get(url_path, {}) or {}

                    # Get title - check yaml config, then use defaults
                    title = yaml_config.get("title")
                    if not title:
                        title = (
                            "Overview"
                            if url_path is None
                            else (url_path or "Dashboard")
                        )

                    # Get icon - check yaml config, then use defaults
                    icon = yaml_config.get("icon")
                    if not icon:
                        icon = "mdi:home" if url_path is None else "mdi:view-dashboard"

                    # Get sidebar/admin settings from yaml config or defaults
                    show_in_sidebar = yaml_config.get("show_in_sidebar", True)
                    require_admin = yaml_config.get("require_admin", False)

                    dashboard_list.append(
                        {
                            "url_path": url_path,
                            "title": title,
                            "icon": icon,
                            "show_in_sidebar": show_in_sidebar,
                            "require_admin": require_admin,
                        }
                    )

                _LOGGER.debug("Found %d dashboards", len(dashboard_list))
                return dashboard_list

            except Exception as e:
                _LOGGER.warning("Could not get dashboards via lovelace: %s", str(e))
                return [{"error": f"Could not retrieve dashboards: {str(e)}"}]

        except Exception as e:
            _LOGGER.exception("Error getting dashboards: %s", str(e))
            return [{"error": f"Error getting dashboards: {str(e)}"}]

    async def get_dashboard_config(
        self, dashboard_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration of a specific dashboard."""
        try:
            _LOGGER.debug(
                "Requesting dashboard config for: %s", dashboard_url or "default"
            )

            # Get dashboard configuration
            try:
                from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

                # Get lovelace data using property access (required for HA 2026.2+)
                lovelace_data = self.hass.data.get(LOVELACE_DOMAIN)
                if lovelace_data is None:
                    return {"error": "Lovelace not available"}

                # Safety check for dashboards attribute (backward compatibility)
                if not hasattr(lovelace_data, "dashboards"):
                    return {"error": "Lovelace dashboards not available"}

                # Use property access instead of dictionary access
                # The dashboards dict uses None as key for the default dashboard
                dashboards = lovelace_data.dashboards

                # Get the dashboard (None key = default dashboard)
                dashboard_key = None if dashboard_url is None else dashboard_url
                if dashboard_key in dashboards:
                    dashboard = dashboards[dashboard_key]
                    config = await dashboard.async_get_info()
                    return dict(config) if config else {"error": "No dashboard config"}
                else:
                    if dashboard_url is None:
                        return {"error": "Default dashboard not found"}
                    else:
                        return {"error": f"Dashboard '{dashboard_url}' not found"}

            except Exception as e:
                _LOGGER.warning("Could not get dashboard config: %s", str(e))
                return {"error": f"Could not retrieve dashboard config: {str(e)}"}

        except Exception as e:
            _LOGGER.exception("Error getting dashboard config: %s", str(e))
            return {"error": f"Error getting dashboard config: {str(e)}"}

    async def create_dashboard(
        self, dashboard_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new dashboard using Home Assistant's Lovelace WebSocket API."""
        try:
            _LOGGER.debug(
                "Creating dashboard with config: %s",
                json.dumps(dashboard_config, default=str),
            )

            # Validate required fields
            if not dashboard_config.get("title"):
                return {"error": "Dashboard title is required"}

            if not dashboard_config.get("url_path"):
                return {"error": "Dashboard URL path is required"}

            # Sanitize the URL path
            url_path = (
                dashboard_config["url_path"].lower().replace(" ", "-").replace("_", "-")
            )

            # Prepare dashboard configuration for Lovelace
            dashboard_data = {
                "title": dashboard_config["title"],
                "icon": dashboard_config.get("icon", "mdi:view-dashboard"),
                "show_in_sidebar": dashboard_config.get("show_in_sidebar", True),
                "require_admin": dashboard_config.get("require_admin", False),
                "views": dashboard_config.get("views", []),
            }

            try:
                # Create dashboard file directly - this is the most reliable method
                import os

                import yaml

                # Create the dashboard YAML file
                lovelace_config_file = self.hass.config.path(
                    f"ui-lovelace-{url_path}.yaml"
                )

                # Use async_add_executor_job to perform file I/O asynchronously
                def write_dashboard_file():
                    with open(lovelace_config_file, "w") as f:
                        yaml.dump(
                            dashboard_data,
                            f,
                            default_flow_style=False,
                            allow_unicode=True,
                        )

                await self.hass.async_add_executor_job(write_dashboard_file)

                _LOGGER.info(
                    "Successfully created dashboard file: %s", lovelace_config_file
                )

                # Now update configuration.yaml
                try:
                    config_file = self.hass.config.path("configuration.yaml")
                    dashboard_config_entry = {
                        url_path: {
                            "mode": "yaml",
                            "title": dashboard_config["title"],
                            "icon": dashboard_config.get("icon", "mdi:view-dashboard"),
                            "show_in_sidebar": dashboard_config.get(
                                "show_in_sidebar", True
                            ),
                            "filename": f"ui-lovelace-{url_path}.yaml",
                        }
                    }

                    def update_config_file():
                        try:
                            with open(config_file, "r") as f:
                                content = f.read()

                            # Dashboard configuration to add
                            dashboard_yaml = f"""    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml"""

                            # Check if lovelace section exists
                            if "lovelace:" not in content:
                                # Add complete lovelace section at the end
                                lovelace_section = f"""
# Lovelace dashboards configuration added by AI Agent
lovelace:
  dashboards:
{dashboard_yaml}
"""
                                with open(config_file, "a") as f:
                                    f.write(lovelace_section)
                                return True

                            # If lovelace exists, check for dashboards section
                            lines = content.split("\n")
                            new_lines = []
                            dashboard_added = False
                            in_lovelace = False
                            lovelace_indent = 0

                            for i, line in enumerate(lines):
                                new_lines.append(line)

                                # Detect lovelace section
                                if (
                                    line.strip() == "lovelace:"
                                    or line.strip().startswith("lovelace:")
                                ):
                                    in_lovelace = True
                                    lovelace_indent = len(line) - len(line.lstrip())
                                    continue

                                # If we're in lovelace section
                                if in_lovelace:
                                    current_indent = (
                                        len(line) - len(line.lstrip())
                                        if line.strip()
                                        else 0
                                    )

                                    # If we hit another top-level section, we're out of lovelace
                                    if (
                                        line.strip()
                                        and current_indent <= lovelace_indent
                                        and not line.startswith(" ")
                                    ):
                                        if line.strip() != "lovelace:":
                                            in_lovelace = False

                                    # Look for dashboards section
                                    if in_lovelace and "dashboards:" in line:
                                        # Add our dashboard after the dashboards: line
                                        new_lines.append(dashboard_yaml)
                                        dashboard_added = True
                                        in_lovelace = False  # We're done
                                        break

                            # If we found lovelace but no dashboards section, add it
                            if not dashboard_added and "lovelace:" in content:
                                # Find lovelace section and add dashboards
                                new_lines = []
                                for line in lines:
                                    new_lines.append(line)
                                    if (
                                        line.strip() == "lovelace:"
                                        or line.strip().startswith("lovelace:")
                                    ):
                                        # Add dashboards section right after lovelace
                                        new_lines.append("  dashboards:")
                                        new_lines.append(dashboard_yaml)
                                        dashboard_added = True
                                        break

                            if dashboard_added:
                                with open(config_file, "w") as f:
                                    f.write("\n".join(new_lines))
                                return True
                            else:
                                # Last resort: append to end of file
                                with open(config_file, "a") as f:
                                    f.write(f"\n  dashboards:\n{dashboard_yaml}\n")
                                return True

                        except Exception as e:
                            _LOGGER.error(
                                "Failed to update configuration.yaml: %s", str(e)
                            )
                            # Fallback to simple append method
                            try:
                                with open(config_file, "r") as f:
                                    content = f.read()

                                # Check if lovelace section exists
                                if "lovelace:" not in content:
                                    # Add lovelace section
                                    lovelace_config = f"""
# Lovelace dashboards
lovelace:
  dashboards:
    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml
"""
                                    with open(config_file, "a") as f:
                                        f.write(lovelace_config)
                                else:
                                    # Add to existing lovelace section (simple approach)
                                    dashboard_entry = f"""    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml
"""
                                    # Find the dashboards section and add to it
                                    lines = content.split("\n")
                                    new_lines = []
                                    in_dashboards = False
                                    dashboards_indented = False

                                    for line in lines:
                                        new_lines.append(line)
                                        if (
                                            "dashboards:" in line
                                            and "lovelace"
                                            in content[: content.find(line)]
                                        ):
                                            in_dashboards = True
                                            # Add our dashboard entry after dashboards:
                                            new_lines.append(dashboard_entry.rstrip())
                                            in_dashboards = False

                                    # If we couldn't find dashboards section, add it under lovelace
                                    if not any("dashboards:" in line for line in lines):
                                        for i, line in enumerate(new_lines):
                                            if line.strip() == "lovelace:":
                                                new_lines.insert(i + 1, "  dashboards:")
                                                new_lines.insert(
                                                    i + 2, dashboard_entry.rstrip()
                                                )
                                                break

                                    with open(config_file, "w") as f:
                                        f.write("\n".join(new_lines))

                                return True
                            except Exception as fallback_error:
                                _LOGGER.error(
                                    "Fallback config update also failed: %s",
                                    str(fallback_error),
                                )
                                return False

                    config_updated = await self.hass.async_add_executor_job(
                        update_config_file
                    )

                    if config_updated:
                        success_message = f"""Dashboard '{dashboard_config['title']}' created successfully!

✅ Dashboard file created: ui-lovelace-{url_path}.yaml
✅ Configuration.yaml updated automatically

🔄 Please restart Home Assistant to see your new dashboard in the sidebar."""

                        return {
                            "success": True,
                            "message": success_message,
                            "url_path": url_path,
                            "restart_required": True,
                        }
                    else:
                        # Config update failed, provide manual instructions
                        config_instructions = f"""Dashboard '{dashboard_config['title']}' created successfully!

✅ Dashboard file created: ui-lovelace-{url_path}.yaml
⚠️  Could not automatically update configuration.yaml

Please manually add this to your configuration.yaml:

lovelace:
  dashboards:
    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml

Then restart Home Assistant to see your new dashboard in the sidebar."""

                        return {
                            "success": True,
                            "message": config_instructions,
                            "url_path": url_path,
                            "restart_required": True,
                        }

                except Exception as config_error:
                    _LOGGER.error(
                        "Error updating configuration.yaml: %s", str(config_error)
                    )
                    # Provide manual instructions as fallback
                    config_instructions = f"""Dashboard '{dashboard_config['title']}' created successfully!

✅ Dashboard file created: ui-lovelace-{url_path}.yaml
⚠️  Could not automatically update configuration.yaml

Please manually add this to your configuration.yaml:

lovelace:
  dashboards:
    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml

Then restart Home Assistant to see your new dashboard in the sidebar."""

                    return {
                        "success": True,
                        "message": config_instructions,
                        "url_path": url_path,
                        "restart_required": True,
                    }

            except Exception as e:
                _LOGGER.error("Failed to create dashboard file: %s", str(e))
                return {"error": f"Failed to create dashboard file: {str(e)}"}

        except Exception as e:
            _LOGGER.exception("Error creating dashboard: %s", str(e))
            return {"error": f"Error creating dashboard: {str(e)}"}

    async def update_dashboard(
        self, dashboard_url: str, dashboard_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing dashboard using Home Assistant's Lovelace WebSocket API."""
        try:
            _LOGGER.debug(
                "Updating dashboard %s with config: %s",
                dashboard_url,
                json.dumps(dashboard_config, default=str),
            )

            # Prepare updated dashboard configuration
            dashboard_data = {
                "title": dashboard_config.get("title", "Updated Dashboard"),
                "icon": dashboard_config.get("icon", "mdi:view-dashboard"),
                "show_in_sidebar": dashboard_config.get("show_in_sidebar", True),
                "require_admin": dashboard_config.get("require_admin", False),
                "views": dashboard_config.get("views", []),
            }

            try:
                # Update dashboard file directly
                import os

                import yaml

                # Try updating the YAML file
                dashboard_file = self.hass.config.path(
                    f"ui-lovelace-{dashboard_url}.yaml"
                )

                # Check if file exists asynchronously
                def check_file_exists():
                    return os.path.exists(dashboard_file)

                file_exists = await self.hass.async_add_executor_job(check_file_exists)

                if not file_exists:
                    dashboard_file = self.hass.config.path(
                        f"dashboards/{dashboard_url}.yaml"
                    )
                    file_exists = await self.hass.async_add_executor_job(
                        lambda: os.path.exists(dashboard_file)
                    )

                if file_exists:
                    # Use async_add_executor_job to perform file I/O asynchronously
                    def update_dashboard_file():
                        with open(dashboard_file, "w") as f:
                            yaml.dump(
                                dashboard_data,
                                f,
                                default_flow_style=False,
                                allow_unicode=True,
                            )

                    await self.hass.async_add_executor_job(update_dashboard_file)

                    _LOGGER.info(
                        "Successfully updated dashboard file: %s", dashboard_file
                    )
                    return {
                        "success": True,
                        "message": f"Dashboard '{dashboard_url}' updated successfully!",
                    }
                else:
                    return {"error": f"Dashboard file for '{dashboard_url}' not found"}

            except Exception as e:
                _LOGGER.error("Failed to update dashboard file: %s", str(e))
                return {"error": f"Failed to update dashboard file: {str(e)}"}

        except Exception as e:
            _LOGGER.exception("Error updating dashboard: %s", str(e))
            return {"error": f"Error updating dashboard: {str(e)}"}

    async def process_query(
        self, user_query: str, provider: Optional[str] = None, debug: bool = False
    ) -> Dict[str, Any]:
        """Process a user query with input validation and rate limiting."""
        try:
            if not user_query or not isinstance(user_query, str):
                return {"success": False, "error": "Invalid query format"}

            # Get the correct configuration for the requested provider
            if provider and provider in self.hass.data[DOMAIN]["configs"]:
                config = self.hass.data[DOMAIN]["configs"][provider]
            else:
                config = self.config

            _LOGGER.debug(f"Processing query with provider: {provider}")
            # Log sanitized config (masks all tokens/keys for security)
            _LOGGER.debug(
                f"Using config: {json.dumps(sanitize_for_logging(config), default=str)}"
            )

            selected_provider = provider or config.get("ai_provider", "llama")
            models_config = config.get("models", {})

            provider_config = {
                "openai": {
                    "token_key": "openai_token",
                    "model": models_config.get("openai", "gpt-3.5-turbo"),
                    "client_class": OpenAIClient,
                },
                "gemini": {
                    "token_key": "gemini_token",
                    "model": models_config.get("gemini", "gemini-2.5-flash"),
                    "client_class": GeminiClient,
                },
                "openrouter": {
                    "token_key": "openrouter_token",
                    "model": models_config.get("openrouter", "openai/gpt-4o"),
                    "client_class": OpenRouterClient,
                },
                "llama": {
                    "token_key": "llama_token",
                    "model": models_config.get(
                        "llama", "Llama-4-Maverick-17B-128E-Instruct-FP8"
                    ),
                    "client_class": LlamaClient,
                },
                "anthropic": {
                    "token_key": "anthropic_token",
                    "model": models_config.get(
                        "anthropic", "claude-sonnet-4-5-20250929"
                    ),
                    "client_class": AnthropicClient,
                },
                "alter": {
                    "token_key": "alter_token",
                    "model": models_config.get("alter", ""),
                    "client_class": AlterClient,
                },
                "zai": {
                    "token_key": "zai_token",
                    "model": models_config.get("zai", ""),
                    "client_class": ZaiClient,
                },
                "local": {
                    "token_key": "local_url",
                    "model": models_config.get("local", ""),
                    "client_class": LocalClient,
                },
            }

            # Validate provider and get configuration
            if selected_provider not in provider_config:
                _LOGGER.warning(
                    f"Invalid provider {selected_provider}, falling back to llama"
                )
                selected_provider = "llama"

            provider_settings = provider_config[selected_provider]
            token = self.config.get(provider_settings["token_key"])

            def _with_debug(result: Dict[str, Any]) -> Dict[str, Any]:
                """Attach a sanitized trace when UI requests debug info."""
                if debug and "debug" not in result:
                    result["debug"] = self._build_debug_trace(
                        selected_provider,
                        provider_settings,
                        config.get("zai_endpoint", "general"),
                    )
                return result

            # Validate token/URL
            if not token:
                error_msg = f"No {'URL' if selected_provider == 'local' else 'token'} configured for provider {selected_provider}"
                _LOGGER.error(error_msg)
                return _with_debug({"success": False, "error": error_msg})

            # Initialize client
            try:
                if selected_provider == "zai":
                    # ZaiClient takes (token, model, endpoint_type)
                    endpoint_type = config.get("zai_endpoint", "general")
                    self.ai_client = provider_settings["client_class"](
                        token=token,
                        model=provider_settings["model"],
                        endpoint_type=endpoint_type,
                    )
                    _LOGGER.debug(
                        f"Initialized {selected_provider} client with model {provider_settings['model']}, endpoint_type {endpoint_type}"
                    )
                elif selected_provider == "local":
                    # LocalClient takes (url, model)
                    self.ai_client = provider_settings["client_class"](
                        url=token, model=provider_settings["model"]
                    )
                    _LOGGER.debug(
                        f"Initialized {selected_provider} client with model {provider_settings['model']}"
                    )
                else:
                    # Other clients take (token, model)
                    self.ai_client = provider_settings["client_class"](
                        token=token, model=provider_settings["model"]
                    )
                    _LOGGER.debug(
                        f"Initialized {selected_provider} client with model {provider_settings['model']}"
                    )
            except Exception as e:
                error_msg = f"Error initializing {selected_provider} client: {str(e)}"
                _LOGGER.error(error_msg)
                return _with_debug({"success": False, "error": error_msg})

            # Process the query with rate limiting and retries
            if not self._check_rate_limit():
                return _with_debug(
                    {
                        "success": False,
                        "error": "Rate limit exceeded. Please wait before trying again.",
                    }
                )

            # Sanitize user input
            user_query = user_query.strip()[:1000]  # Limit length and trim whitespace

            _LOGGER.debug("Processing new query: %s", user_query)

            # Check cache for identical query
            cache_key = f"query_{hash(user_query)}_{provider}_{debug}"
            cached_result = self._get_cached_data(cache_key)
            if cached_result:
                return (
                    dict(cached_result)
                    if isinstance(cached_result, dict)
                    else {"error": "Invalid cached result"}
                )

            # Add system message to conversation if it's the first message
            if not self.conversation_history:
                _LOGGER.debug("Adding system message to new conversation")
                self.conversation_history.append(self.system_prompt)

            # Add user query to conversation
            self.conversation_history.append({"role": "user", "content": user_query})
            _LOGGER.debug("Added user query to conversation history")

            max_iterations = 5  # Prevent infinite loops
            iteration = 0

            # Fetch MCP tools once
            available_tools = await self._get_mcp_tools()
            if available_tools:
                _LOGGER.debug(f"Retrieved {len(available_tools)} MCP tools for prompt injection.")

            while iteration < max_iterations:
                iteration += 1
                _LOGGER.debug(f"Processing iteration {iteration} of {max_iterations}")

                try:
                    # Get AI response
                    _LOGGER.debug("Requesting response from AI provider")
                    response = await self._get_ai_response(tools=available_tools)
                    _LOGGER.debug("Received response from AI provider: %s", response)

                    try:
                        # Try to parse the response as JSON with simplified approach
                        response_clean = response.strip()

                        # Remove potential BOM and other invisible characters
                        import codecs

                        if response_clean.startswith(codecs.BOM_UTF8.decode("utf-8")):
                            response_clean = response_clean[1:]

                        # Remove other common invisible characters
                        invisible_chars = [
                            "\ufeff",
                            "\u200b",
                            "\u200c",
                            "\u200d",
                            "\u2060",
                        ]
                        for char in invisible_chars:
                            response_clean = response_clean.replace(char, "")

                        _LOGGER.debug(
                            "Cleaned response length: %d", len(response_clean)
                        )
                        _LOGGER.debug(
                            "Cleaned response first 100 chars: %s", response_clean[:100]
                        )
                        _LOGGER.debug(
                            "Cleaned response last 100 chars: %s", response_clean[-100:]
                        )

                        # Simple strategy: try to parse the cleaned response directly
                        response_data = None
                        try:
                            _LOGGER.debug("Attempting basic JSON parse...")
                            response_data = json.loads(response_clean)
                            _LOGGER.debug("Basic JSON parse succeeded!")
                        except json.JSONDecodeError as e:
                            _LOGGER.warning("Basic JSON parse failed: %s", str(e))
                            _LOGGER.debug("JSON error position: %d", e.pos)
                            if e.pos < len(response_clean):
                                _LOGGER.debug(
                                    "Character at error position: %s (ord: %d)",
                                    repr(response_clean[e.pos]),
                                    ord(response_clean[e.pos]),
                                )
                                _LOGGER.debug(
                                    "Context around error: %s",
                                    repr(
                                        response_clean[max(0, e.pos - 10) : e.pos + 10]
                                    ),
                                )

                            # Fallback: try to extract JSON by finding the first { and last }
                            json_start = response_clean.find("{")
                            json_end = response_clean.rfind("}")

                            if (
                                json_start != -1
                                and json_end != -1
                                and json_end > json_start
                            ):
                                json_part = response_clean[json_start : json_end + 1]
                                _LOGGER.debug(
                                    "Trying fallback extraction from pos %d to %d",
                                    json_start,
                                    json_end,
                                )
                                _LOGGER.debug("Extracted JSON: %s", json_part[:200])

                                try:
                                    response_data = json.loads(json_part)
                                    _LOGGER.debug("Fallback JSON extraction succeeded!")
                                except json.JSONDecodeError as e2:
                                    _LOGGER.warning(
                                        "Fallback JSON extraction also failed: %s",
                                        str(e2),
                                    )
                                    raise e  # Re-raise the original error
                            else:
                                _LOGGER.warning(
                                    "Could not find JSON boundaries in response"
                                )
                                raise e  # Re-raise the original error

                        if response_data is None:
                            raise json.JSONDecodeError(
                                "All parsing strategies failed", response_clean, 0
                            )

                        _LOGGER.debug("Successfully parsed JSON response")
                        _LOGGER.debug(
                            "Parsed response type: %s",
                            response_data.get("request_type", "unknown"),
                        )

                        # Check if this is a data request (either format)
                        data_request_types = [
                            "get_entity_state",
                            "get_entities_by_domain",
                            "get_entities_by_device_class",
                            "get_climate_related_entities",
                            "get_entities_by_area",
                            "get_entities",
                            "get_calendar_events",
                            "get_automations",
                            "get_entity_registry",
                            "get_device_registry",
                            "get_weather_data",
                            "get_area_registry",
                            "get_history",
                            "get_person_data",
                            "get_statistics",
                            "get_scenes",
                            "get_dashboards",
                            "get_dashboard_config",
                            "set_entity_state",
                            "create_automation",
                            "create_dashboard",
                            "update_dashboard",
                        ]

                        if response_data.get("request_type") == "_mcp_tool_calls":
                            tool_calls = response_data.get("tool_calls", [])
                            text_content = response_data.get("content", "")
                            
                            # Log assistant's message to conversation
                            assistant_msg = {
                                "role": "assistant",
                                "content": text_content,
                                "tool_calls": tool_calls
                            }
                            self.conversation_history.append(assistant_msg)
                            
                            # Execute each tool
                            mcp_server = self.hass.data[DOMAIN].get("mcp_server")
                            for tc in tool_calls:
                                tool_name = tc.get("name")
                                arguments = tc.get("arguments", {})
                                tool_id = tc.get("id")
                                
                                _LOGGER.debug(f"Executing MCP tool {tool_name} with args {arguments}")
                                
                                # Fire HA event for UI feedback
                                self.hass.bus.async_fire("ai_agent_ha_tool_call", {
                                    "tool": tool_name,
                                    "arguments": arguments,
                                    "id": tool_id
                                })
                                
                                try:
                                    if mcp_server:
                                        # Use the MCPServer wrapper's tool call logic
                                        result = await mcp_server.handle_tool_call({"name": tool_name, "arguments": arguments})
                                        result_text = json.dumps(result, default=str)
                                    else:
                                        result_text = json.dumps({"isError": True, "content": [{"type": "text", "text": "MCP Server not running."}]})
                                except Exception as e:
                                    _LOGGER.error(f"Error executing MCP tool {tool_name}: {e}")
                                    result_text = json.dumps({"isError": True, "content": [{"type": "text", "text": str(e)}]})
                                
                                self.conversation_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": result_text
                                })
                                
                            continue

                        elif (
                            response_data.get("request_type") == "data_request"
                            or response_data.get("request_type") in data_request_types
                        ):
                            # Handle data request (both standard format and direct request type)
                            if response_data.get("request_type") == "data_request":
                                request_type = response_data.get("request")
                            else:
                                request_type = response_data.get("request_type")
                            parameters = response_data.get("parameters", {})
                            _LOGGER.debug(
                                "Processing data request: %s with parameters: %s",
                                request_type,
                                json.dumps(parameters),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Get requested data
                            data: Union[Dict[str, Any], List[Dict[str, Any]]]
                            if request_type == "get_entity_state":
                                data = await self.get_entity_state(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_entities_by_domain":
                                data = await self.get_entities_by_domain(
                                    parameters.get("domain")
                                )
                            elif request_type == "get_entities_by_area":
                                data = await self.get_entities_by_area(
                                    parameters.get("area_id")
                                )
                            elif request_type == "get_entities":
                                data = await self.get_entities(
                                    area_id=parameters.get("area_id"),
                                    area_ids=parameters.get("area_ids"),
                                )
                            elif request_type == "get_entities_by_device_class":
                                data = await self.get_entities_by_device_class(
                                    parameters.get("device_class"),
                                    parameters.get("domain"),
                                )
                            elif request_type == "get_climate_related_entities":
                                data = await self.get_climate_related_entities()
                            elif request_type == "get_calendar_events":
                                data = await self.get_calendar_events(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_automations":
                                data = await self.get_automations()
                            elif request_type == "get_entity_registry":
                                data = await self.get_entity_registry()
                            elif request_type == "get_device_registry":
                                data = await self.get_device_registry()
                            elif request_type == "get_weather_data":
                                data = await self.get_weather_data()
                            elif request_type == "get_area_registry":
                                data = await self.get_area_registry()
                            elif request_type == "get_history":
                                data = await self.get_history(
                                    parameters.get("entity_id"),
                                    parameters.get("hours", 24),
                                )
                            elif request_type == "get_person_data":
                                data = await self.get_person_data()
                            elif request_type == "get_statistics":
                                data = await self.get_statistics(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_scenes":
                                data = await self.get_scenes()
                            elif request_type == "get_dashboards":
                                data = await self.get_dashboards()
                            elif request_type == "get_dashboard_config":
                                data = await self.get_dashboard_config(
                                    parameters.get("dashboard_url")
                                )
                            elif request_type == "set_entity_state":
                                data = await self.set_entity_state(
                                    parameters.get("entity_id"),
                                    parameters.get("state"),
                                    parameters.get("attributes"),
                                )
                            elif request_type == "create_automation":
                                data = await self.create_automation(
                                    parameters.get("automation")
                                )
                            elif request_type == "create_dashboard":
                                data = await self.create_dashboard(
                                    parameters.get("dashboard_config")
                                )
                            elif request_type == "update_dashboard":
                                data = await self.update_dashboard(
                                    parameters.get("dashboard_url"),
                                    parameters.get("dashboard_config"),
                                )
                            else:
                                data = {
                                    "error": f"Unknown request type: {request_type}"
                                }
                                _LOGGER.warning(
                                    "Unknown request type: %s", request_type
                                )

                            # Check if any data request resulted in an error
                            if isinstance(data, dict) and "error" in data:
                                return _with_debug(
                                    {"success": False, "error": data["error"]}
                                )
                            elif isinstance(data, list) and any(
                                "error" in item
                                for item in data
                                if isinstance(item, dict)
                            ):
                                errors = [
                                    item["error"]
                                    for item in data
                                    if isinstance(item, dict) and "error" in item
                                ]
                                return _with_debug(
                                    {"success": False, "error": "; ".join(errors)}
                                )

                            _LOGGER.debug(
                                "Retrieved data for request: %s",
                                json.dumps(data, default=str),
                            )

                            # Add data to conversation as a user message (not system to avoid overwriting system prompt in Anthropic API)
                            self.conversation_history.append(
                                {
                                    "role": "user",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            continue

                        elif response_data.get("request_type") == "final_response":
                            # Add final response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Return final response
                            _LOGGER.debug(
                                "Received final response: %s",
                                response_data.get("response"),
                            )
                            result = {
                                "success": True,
                                "answer": response_data.get("response", ""),
                            }
                            result = _with_debug(result)
                            self._set_cached_data(cache_key, result)
                            return result
                        elif (
                            response_data.get("request_type") == "automation_suggestion"
                        ):
                            # Add automation suggestion to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Return automation suggestion
                            _LOGGER.debug(
                                "Received automation suggestion: %s",
                                json.dumps(response_data.get("automation")),
                            )
                            result = {
                                "success": True,
                                "answer": json.dumps(response_data),
                            }
                            result = _with_debug(result)
                            self._set_cached_data(cache_key, result)
                            return result
                        elif (
                            response_data.get("request_type") == "dashboard_suggestion"
                        ):
                            # Add dashboard suggestion to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Return dashboard suggestion
                            _LOGGER.debug(
                                "Received dashboard suggestion: %s",
                                json.dumps(response_data.get("dashboard")),
                            )
                            result = {
                                "success": True,
                                "answer": json.dumps(response_data),
                            }
                            result = _with_debug(result)
                            self._set_cached_data(cache_key, result)
                            return result
                        elif response_data.get("request_type") in [
                            "get_entities",
                            "get_entities_by_area",
                        ]:
                            # Handle direct get_entities request (for backward compatibility)
                            parameters = response_data.get("parameters", {})
                            _LOGGER.debug(
                                "Processing direct get_entities request with parameters: %s",
                                json.dumps(parameters),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Get entities data
                            if response_data.get("request_type") == "get_entities":
                                data = await self.get_entities(
                                    area_id=parameters.get("area_id"),
                                    area_ids=parameters.get("area_ids"),
                                )
                            else:  # get_entities_by_area
                                data = await self.get_entities_by_area(
                                    parameters.get("area_id")
                                )

                            _LOGGER.debug(
                                "Retrieved %d entities",
                                len(data) if isinstance(data, list) else 1,
                            )

                            # Add data to conversation as a user message (not system to avoid overwriting system prompt in Anthropic API)
                            self.conversation_history.append(
                                {
                                    "role": "user",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            continue
                        elif response_data.get("request_type") == "call_service":
                            # Handle service call request
                            domain = response_data.get("domain")
                            service = response_data.get("service")
                            target = response_data.get("target", {})
                            service_data = response_data.get("service_data", {})

                            # Resolve nested requests in target
                            if target and "entity_id" in target:
                                entity_id_value = target["entity_id"]
                                if (
                                    isinstance(entity_id_value, dict)
                                    and "request_type" in entity_id_value
                                ):
                                    # This is a nested request, resolve it
                                    nested_request_type = entity_id_value.get(
                                        "request_type"
                                    )
                                    nested_parameters = entity_id_value.get(
                                        "parameters", {}
                                    )

                                    _LOGGER.debug(
                                        "Resolving nested request: %s with parameters: %s",
                                        nested_request_type,
                                        json.dumps(nested_parameters),
                                    )

                                    # Resolve the nested request
                                    if nested_request_type == "get_entities":
                                        entities_data = await self.get_entities(
                                            area_id=nested_parameters.get("area_id"),
                                            area_ids=nested_parameters.get("area_ids"),
                                        )
                                    elif nested_request_type == "get_entities_by_area":
                                        entities_data = await self.get_entities_by_area(
                                            nested_parameters.get("area_id")
                                        )
                                    elif (
                                        nested_request_type == "get_entities_by_domain"
                                    ):
                                        entities_data = (
                                            await self.get_entities_by_domain(
                                                nested_parameters.get("domain")
                                            )
                                        )
                                    else:
                                        _LOGGER.error(
                                            "Unsupported nested request type: %s",
                                            nested_request_type,
                                        )
                                        return {
                                            "success": False,
                                            "error": f"Unsupported nested request type: {nested_request_type}",
                                        }

                                    # Extract entity IDs from the resolved data
                                    if isinstance(entities_data, list):
                                        entity_ids = [
                                            entity.get("entity_id")
                                            for entity in entities_data
                                            if entity.get("entity_id")
                                        ]
                                        target["entity_id"] = entity_ids
                                        _LOGGER.debug(
                                            "Resolved nested request to entity IDs: %s",
                                            entity_ids,
                                        )
                                    else:
                                        _LOGGER.error(
                                            "Nested request returned unexpected data format"
                                        )
                                        return _with_debug(
                                            {
                                                "success": False,
                                                "error": "Nested request returned unexpected data format",
                                            }
                                        )

                            # Handle backward compatibility with old format
                            if not domain or not service:
                                request = response_data.get("request")
                                parameters = response_data.get("parameters", {})

                                if request and "entity_id" in parameters:
                                    entity_id = parameters["entity_id"]
                                    # Infer domain from entity_id
                                    if "." in entity_id:
                                        domain = entity_id.split(".")[0]
                                        service = request
                                        target = {"entity_id": entity_id}
                                        # Remove entity_id from parameters to avoid duplication
                                        service_data = {
                                            k: v
                                            for k, v in parameters.items()
                                            if k != "entity_id"
                                        }
                                        _LOGGER.debug(
                                            "Converted old format: domain=%s, service=%s",
                                            domain,
                                            service,
                                        )

                            _LOGGER.debug(
                                "Processing service call: %s.%s with target: %s and data: %s",
                                domain,
                                service,
                                json.dumps(target),
                                json.dumps(service_data),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Call the service
                            data = await self.call_service(
                                domain, service, target, service_data
                            )

                            # Check if service call resulted in an error
                            if isinstance(data, dict) and "error" in data:
                                return _with_debug(
                                    {"success": False, "error": data["error"]}
                                )

                            _LOGGER.debug(
                                "Service call completed: %s",
                                json.dumps(data, default=str),
                            )

                            # Add data to conversation as a user message (not system to avoid overwriting system prompt in Anthropic API)
                            self.conversation_history.append(
                                {
                                    "role": "user",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            # Go to next iteration to continue the loop
                            continue

                        # Unknown request type
                        _LOGGER.warning(
                            "Unknown response type: %s",
                            response_data.get("request_type"),
                        )
                        return _with_debug(
                            {
                                "success": False,
                                "error": f"Unknown response type: {response_data.get('request_type')}",
                            }
                        )

                    except json.JSONDecodeError as e:
                        # Check if this is a local provider that might have already wrapped the response
                        provider = self.config.get("ai_provider", "unknown")
                        if provider == "local":
                            _LOGGER.debug(
                                "Local provider returned non-JSON response (this is normal and handled): %s",
                                response[:200],
                            )
                        else:
                            # Log more of the response to help with debugging for non-local providers
                            response_preview = (
                                response[:1000] if len(response) > 1000 else response
                            )
                            _LOGGER.warning(
                                "Failed to parse response as JSON: %s. Response length: %d. Response preview: %s",
                                str(e),
                                len(response),
                                response_preview,
                            )

                            # Log additional debugging information
                            _LOGGER.debug(
                                "First 50 characters as bytes: %s",
                                response[:50].encode("utf-8") if response else b"",
                            )
                            _LOGGER.debug(
                                "Response starts with: %s",
                                repr(response[:10]) if response else "None",
                            )

                        # Also log the response to a separate debug file for detailed analysis (non-local providers only)
                        if provider != "local":
                            try:
                                import os

                                debug_dir = "/config/ai_agent_ha_debug"

                                def write_debug_file():
                                    if not os.path.exists(debug_dir):
                                        os.makedirs(debug_dir)

                                    import datetime

                                    timestamp = datetime.datetime.now().strftime(
                                        "%Y%m%d_%H%M%S"
                                    )
                                    debug_file = os.path.join(
                                        debug_dir, f"failed_response_{timestamp}.txt"
                                    )

                                    with open(debug_file, "w", encoding="utf-8") as f:
                                        f.write(f"Timestamp: {timestamp}\n")
                                        f.write(f"Provider: {provider}\n")
                                        f.write(f"Error: {str(e)}\n")
                                        f.write(f"Response length: {len(response)}\n")
                                        f.write(
                                            f"Response bytes: {response.encode('utf-8') if response else b''}\n"
                                        )
                                        f.write(f"Response repr: {repr(response)}\n")
                                        f.write(f"Full response:\n{response}\n")

                                    return debug_file

                                # Run file operations in executor to avoid blocking
                                debug_file = await self.hass.async_add_executor_job(
                                    write_debug_file
                                )
                                _LOGGER.info(
                                    "Failed response saved to debug file: %s",
                                    debug_file,
                                )
                            except Exception as debug_error:
                                _LOGGER.debug(
                                    "Could not save debug file: %s", str(debug_error)
                                )

                        # Check if this looks like a corrupted automation suggestion
                        if (
                            response.strip().startswith(
                                '{"request_type": "automation_suggestion'
                            )
                            and len(response) > 10000
                            and response.count("for its use in various fields") > 50
                        ):
                            _LOGGER.warning(
                                "Detected corrupted automation suggestion response with repetitive text"
                            )
                            result = _with_debug(
                                {
                                    "success": False,
                                    "error": "AI generated corrupted automation response. Please try again with a more specific automation request.",
                                }
                            )
                            self._set_cached_data(cache_key, result)
                            return result

                        # If response is not valid JSON, try to wrap it as a final response
                        try:
                            # Truncate extremely long responses to prevent memory issues
                            response_to_wrap = response
                            if len(response) > 50000:
                                response_to_wrap = (
                                    response[:5000]
                                    + "... [Response truncated due to excessive length]"
                                )
                                _LOGGER.warning(
                                    "Truncated extremely long response from %d to 5000 characters",
                                    len(response),
                                )

                            wrapped_response = {
                                "request_type": "final_response",
                                "response": response_to_wrap,
                            }
                            result = {
                                "success": True,
                                "answer": json.dumps(wrapped_response),
                            }
                            _LOGGER.debug("Wrapped non-JSON response as final_response")
                        except Exception as wrap_error:
                            _LOGGER.error(
                                "Failed to wrap response: %s", str(wrap_error)
                            )
                            result = {
                                "success": False,
                                "error": f"Invalid response format: {str(e)}",
                            }

                        result = _with_debug(result)
                        self._set_cached_data(cache_key, result)
                        return result

                except Exception as e:
                    _LOGGER.exception("Error processing AI response: %s", str(e))
                    return _with_debug(
                        {
                            "success": False,
                            "error": f"Error processing AI response: {str(e)}",
                        }
                    )

            # If we've reached max iterations without a final response
            _LOGGER.warning("Reached maximum iterations without final response")
            result = {
                "success": False,
                "error": "Maximum iterations reached without final response",
            }
            result = _with_debug(result)
            self._set_cached_data(cache_key, result)
            return result

        except Exception as e:
            _LOGGER.exception("Error in process_query: %s", str(e))
            return _with_debug(
                {"success": False, "error": f"Error in process_query: {str(e)}"}
            )

    def _build_debug_trace(
        self,
        provider: Optional[str],
        provider_settings: Optional[Dict[str, Any]],
        endpoint_type: Optional[str],
    ) -> Dict[str, Any]:
        """Return a sanitized snapshot of the HA↔AI conversation for UI display."""
        history_tail = (
            self.conversation_history[-20:] if self.conversation_history else []
        )
        return {
            "provider": provider,
            "model": provider_settings.get("model") if provider_settings else None,
            "endpoint_type": endpoint_type,
            "conversation": history_tail,
        }

    async def _get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Fetch available MCP tools from the local server instance."""
        mcp_server = self.hass.data[DOMAIN].get("mcp_server")
        if not mcp_server or not hasattr(mcp_server, "mcp_server"):
            return []
            
        try:
            # Get tools from the wrapper's handle_tools_list which constructs the definitions
            res = await mcp_server.handle_tools_list()
            tools = res.get("tools", [])
            
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("inputSchema", {})
                    }
                })
            return openai_tools
        except Exception as e:
            _LOGGER.error("Error getting MCP tools from SDK: %s", str(e))
            return []

    async def _get_ai_response(self, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Get response from the selected AI provider with retries and rate limiting."""
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded. Please try again later.")
        retry_count = 0
        last_error = None
        # Limit conversation history to last 15 messages to prevent token overflow
        recent_messages = (
            self.conversation_history[-15:]
            if len(self.conversation_history) > 15
            else self.conversation_history
        )
        # Ensure system prompt is always the first message
        if not recent_messages or recent_messages[0].get("role") != "system":
            recent_messages = [self.system_prompt] + recent_messages

        _LOGGER.debug("Sending %d messages to AI provider", len(recent_messages))
        _LOGGER.debug("AI provider: %s", self.config.get("ai_provider", "unknown"))

        while retry_count < self._max_retries:
            try:
                _LOGGER.debug(
                    "Attempt %d/%d: Calling AI client",
                    retry_count + 1,
                    self._max_retries,
                )
                response = await self.ai_client.get_response(recent_messages, tools=tools)
                _LOGGER.debug(
                    "AI client returned response of length: %d", len(response or "")
                )
                _LOGGER.debug("AI response preview: %s", (response or "")[:200])

                # Check for extremely long responses that might indicate model issues
                if response and len(response) > 50000:
                    _LOGGER.warning(
                        "AI returned extremely long response (%d characters), this may indicate a model issue",
                        len(response),
                    )
                    # Check for repetitive patterns that indicate a corrupted response
                    if response.count("for its use in various fields") > 50:
                        _LOGGER.error(
                            "Detected corrupted repetitive response, aborting this iteration"
                        )
                        raise Exception(
                            "AI generated corrupted response with repetitive text. Please try again with a clearer request."
                        )

                # Check if response is empty
                if not response or response.strip() == "":
                    _LOGGER.warning(
                        "AI client returned empty response on attempt %d",
                        retry_count + 1,
                    )
                    if retry_count + 1 >= self._max_retries:
                        raise Exception(
                            "AI provider returned empty response after all retries"
                        )
                    else:
                        retry_count += 1
                        await asyncio.sleep(self._retry_delay * retry_count)
                        continue

                return str(response)
            except Exception as e:
                _LOGGER.error(
                    "AI client error on attempt %d: %s", retry_count + 1, str(e)
                )
                last_error = e
                retry_count += 1
                if retry_count < self._max_retries:
                    await asyncio.sleep(self._retry_delay * retry_count)
                continue
        raise Exception(
            f"Failed after {retry_count} retries. Last error: {str(last_error)}"
        )

    def clear_conversation_history(self) -> None:
        """Clear the conversation history and cache."""
        self.conversation_history = []
        self._cache.clear()
        _LOGGER.debug("Conversation history and cache cleared")

    async def set_entity_state(
        self, entity_id: str, state: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Set the state of an entity."""
        try:
            _LOGGER.debug(
                "Setting state for entity %s to %s with attributes: %s",
                entity_id,
                state,
                json.dumps(attributes or {}),
            )

            # Validate entity exists
            if not self.hass.states.get(entity_id):
                return {"error": f"Entity {entity_id} not found"}

            # Call the appropriate service based on the domain
            domain = entity_id.split(".")[0]

            if domain == "light":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                service_data = {"entity_id": entity_id}
                if attributes and service == "turn_on":
                    service_data.update(attributes)
                await self.hass.services.async_call("light", service, service_data)

            elif domain == "switch":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                await self.hass.services.async_call(
                    "switch", service, {"entity_id": entity_id}
                )

            elif domain == "cover":
                if state.lower() in ["open", "up"]:
                    service = "open_cover"
                elif state.lower() in ["close", "down"]:
                    service = "close_cover"
                elif state.lower() == "stop":
                    service = "stop_cover"
                else:
                    return {"error": f"Invalid state {state} for cover entity"}
                await self.hass.services.async_call(
                    "cover", service, {"entity_id": entity_id}
                )

            elif domain == "climate":
                service_data = {"entity_id": entity_id}
                if state.lower() in ["on", "true", "1"]:
                    service = "turn_on"
                elif state.lower() in ["off", "false", "0"]:
                    service = "turn_off"
                elif state.lower() in ["heat", "cool", "dry", "fan_only", "auto"]:
                    service = "set_hvac_mode"
                    service_data["hvac_mode"] = state.lower()
                else:
                    return {"error": f"Invalid state {state} for climate entity"}
                await self.hass.services.async_call("climate", service, service_data)

            elif domain == "fan":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                service_data = {"entity_id": entity_id}
                if attributes and service == "turn_on":
                    service_data.update(attributes)
                await self.hass.services.async_call("fan", service, service_data)

            else:
                # For other domains, try to set the state directly
                self.hass.states.async_set(entity_id, state, attributes or {})

            # Get the new state to confirm the change
            new_state = self.hass.states.get(entity_id)
            return {
                "success": True,
                "entity_id": entity_id,
                "new_state": new_state.state,
                "new_attributes": new_state.attributes,
            }

        except Exception as e:
            _LOGGER.exception("Error setting entity state: %s", str(e))
            return {"error": f"Error setting entity state: {str(e)}"}

    async def call_service(
        self,
        domain: str,
        service: str,
        target: Optional[Dict[str, Any]] = None,
        service_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a Home Assistant service."""
        try:
            _LOGGER.debug(
                "Calling service %s.%s with target: %s and data: %s",
                domain,
                service,
                json.dumps(target or {}),
                json.dumps(service_data or {}),
            )

            # Prepare the service call data
            call_data = {}

            # Add target entities if provided
            if target:
                if "entity_id" in target:
                    entity_ids = target["entity_id"]
                    if isinstance(entity_ids, list):
                        call_data["entity_id"] = entity_ids
                    else:
                        call_data["entity_id"] = [entity_ids]

                # Add other target properties
                for key, value in target.items():
                    if key != "entity_id":
                        call_data[key] = value

            # Add service data if provided
            if service_data:
                call_data.update(service_data)

            _LOGGER.debug("Final service call data: %s", json.dumps(call_data))

            # Call the service
            await self.hass.services.async_call(domain, service, call_data)

            # Get the updated states of affected entities
            result_entities = []
            if "entity_id" in call_data:
                for entity_id in call_data["entity_id"]:
                    state = self.hass.states.get(entity_id)
                    if state:
                        result_entities.append(
                            {
                                "entity_id": entity_id,
                                "state": state.state,
                                "attributes": dict(state.attributes),
                            }
                        )

            return {
                "success": True,
                "service": f"{domain}.{service}",
                "entities_affected": result_entities,
                "message": f"Successfully called {domain}.{service}",
            }

        except Exception as e:
            _LOGGER.exception(
                "Error calling service %s.%s: %s", domain, service, str(e)
            )
            return {"error": f"Error calling service {domain}.{service}: {str(e)}"}

    async def save_user_prompt_history(
        self, user_id: str, history: List[str]
    ) -> Dict[str, Any]:
        """Save user's prompt history to HA storage."""
        try:
            store: Store = Store(self.hass, 1, f"ai_agent_ha_history_{user_id}")
            await store.async_save({"history": history})
            return {"success": True}
        except Exception as e:
            _LOGGER.exception("Error saving prompt history: %s", str(e))
            return {"error": f"Error saving prompt history: {str(e)}"}

    async def load_user_prompt_history(self, user_id: str) -> Dict[str, Any]:
        """Load user's prompt history from HA storage."""
        try:
            store: Store = Store(self.hass, 1, f"ai_agent_ha_history_{user_id}")
            data = await store.async_load()
            history = data.get("history", []) if data else []
            return {"success": True, "history": history}
        except Exception as e:
            _LOGGER.exception("Error loading prompt history: %s", str(e))
            return {"error": f"Error loading prompt history: {str(e)}", "history": []}
