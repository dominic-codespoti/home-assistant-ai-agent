"""In-process bridge to Home Assistant's official MCP tool surface."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

_STATELESS_LLM_API = "stateless_assist"

try:
    from voluptuous_openapi import convert as _convert_openapi
except Exception:  # pragma: no cover - fallback for non-HA test environments
    _convert_openapi = None


class OfficialMCPBridge:
    """Expose official MCP tools and execution via Home Assistant LLM APIs."""

    def __init__(self, hass: HomeAssistant, mcp_entry: ConfigEntry) -> None:
        self.hass = hass
        self._mcp_entry_id = mcp_entry.entry_id

    def _resolve_llm_api_id(self, configured_api: Any) -> str:
        """Resolve the configured official MCP LLM API id."""
        if isinstance(configured_api, list) and configured_api:
            llm_api_id = configured_api[0]
        elif isinstance(configured_api, str) and configured_api:
            llm_api_id = configured_api
        else:
            llm_api_id = llm.LLM_API_ASSIST

        if llm_api_id == _STATELESS_LLM_API:
            return llm.LLM_API_ASSIST
        return llm_api_id

    async def _get_llm_api_instance(
        self, context: Context | None = None
    ) -> llm.APIInstance:
        """Get the LLM API instance selected by the official MCP config entry."""
        mcp_entry = self.hass.config_entries.async_get_entry(self._mcp_entry_id)
        if mcp_entry is None:
            raise HomeAssistantError(
                "Official MCP server config entry is not available"
            )

        llm_api_id = self._resolve_llm_api_id(mcp_entry.data.get(CONF_LLM_HASS_API))
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=context or Context(),
            language="*",
            assistant=conversation.DOMAIN,
            device_id=None,
        )
        return await llm.async_get_api(self.hass, llm_api_id, llm_context)

    def _format_tool(
        self, tool: llm.Tool, custom_serializer: Any
    ) -> dict[str, Any]:
        """Format tool specification in the official MCP shape."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        if _convert_openapi is not None:
            try:
                input_schema = _convert_openapi(
                    tool.parameters, custom_serializer=custom_serializer
                )
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])
            except Exception as err:
                _LOGGER.debug(
                    "Failed to convert MCP tool schema for %s: %s", tool.name, err
                )

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        return {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": schema,
        }

    async def async_list_tools(
        self, context: Context | None = None
    ) -> list[dict[str, Any]]:
        """Return tool definitions from Home Assistant's official MCP source."""
        llm_api = await self._get_llm_api_instance(context)
        return [
            self._format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
        ]

    async def async_call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        context: Context | None = None,
    ) -> Any:
        """Execute a tool call using the official MCP/LLM API tool executor."""
        if not tool_name:
            raise HomeAssistantError("Missing MCP tool name")

        llm_api = await self._get_llm_api_instance(context)
        tool_input = llm.ToolInput(
            tool_name=tool_name,
            tool_args=arguments if isinstance(arguments, dict) else {},
        )

        try:
            return await llm_api.async_call_tool(tool_input)
        except (HomeAssistantError, vol.Invalid) as err:
            raise HomeAssistantError(
                f"Error calling official MCP tool {tool_name}: {err}"
            ) from err
