"""Smoke tests for MCP-driven query orchestration."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the parent directory to the path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import homeassistant

    HOMEASSISTANT_AVAILABLE = True
except ImportError:
    HOMEASSISTANT_AVAILABLE = False


@pytest.fixture
def mock_hass():
    """Mock Home Assistant instance."""
    mock = MagicMock()
    mock.data = {}
    mock.states = MagicMock()
    mock.states.async_all.return_value = []
    mock.services = MagicMock()
    mock.bus = MagicMock()
    mock.config = MagicMock()
    return mock


def _agent_config() -> dict:
    return {
        "ai_provider": "openai",
        "openai_token": "test_token_123",
        "models": {"openai": "gpt-4o"},
    }


@pytest.mark.asyncio
async def test_process_query_uses_official_mcp_tool_calls(mock_hass):
    """Agent should execute _mcp_tool_calls through the official MCP bridge."""
    if not HOMEASSISTANT_AVAILABLE:
        pytest.skip("Home Assistant not available")

    from custom_components.ai_agent_ha.agent import AiAgentHaAgent
    from custom_components.ai_agent_ha.const import DOMAIN

    config = _agent_config()
    mock_hass.data[DOMAIN] = {"configs": {"openai": config}}

    mock_bridge = MagicMock()
    mock_bridge.async_list_tools = AsyncMock(
        return_value=[
            {
                "name": "get_current_temperatures",
                "description": "Fetch indoor and outdoor temperatures",
                "input_schema": {
                    "type": "object",
                    "properties": {"include_outdoor": {"type": "boolean"}},
                },
            }
        ]
    )
    mock_bridge.async_call_tool = AsyncMock(
        return_value={
            "inside": {"value": 22.1, "unit": "C"},
            "outside": {"value": 14.9, "unit": "C"},
        }
    )

    mock_client = MagicMock()
    mock_client.get_response = AsyncMock(
        side_effect=[
            json.dumps(
                {
                    "request_type": "_mcp_tool_calls",
                    "tool_calls": [
                        {
                            "name": "get_current_temperatures",
                            "arguments": {"include_outdoor": True},
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "request_type": "final_response",
                    "response": "Inside is 22.1C and outside is 14.9C.",
                }
            ),
        ]
    )

    with patch("custom_components.ai_agent_ha.agent.OpenAIClient", return_value=mock_client):
        agent = AiAgentHaAgent(mock_hass, config, mcp_bridge=mock_bridge)
        result = await agent.process_query(
            "What's the current temperature inside and outside?", provider="openai"
        )

    assert result["success"] is True
    assert "22.1C" in result["answer"]
    assert mock_bridge.async_call_tool.await_count == 1


@pytest.mark.asyncio
async def test_process_query_rejects_legacy_data_request(mock_hass):
    """Legacy helper request types should be rejected after MCP cutover."""
    if not HOMEASSISTANT_AVAILABLE:
        pytest.skip("Home Assistant not available")

    from custom_components.ai_agent_ha.agent import AiAgentHaAgent
    from custom_components.ai_agent_ha.const import DOMAIN

    config = _agent_config()
    mock_hass.data[DOMAIN] = {"configs": {"openai": config}}

    mock_bridge = MagicMock()
    mock_bridge.async_list_tools = AsyncMock(
        return_value=[
            {
                "name": "dummy_tool",
                "description": "dummy",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
    )

    mock_client = MagicMock()
    mock_client.get_response = AsyncMock(
        return_value=json.dumps(
            {
                "request_type": "data_request",
                "request": "get_entities_by_domain",
                "parameters": {"domain": "light"},
            }
        )
    )

    with patch("custom_components.ai_agent_ha.agent.OpenAIClient", return_value=mock_client):
        agent = AiAgentHaAgent(mock_hass, config, mcp_bridge=mock_bridge)
        result = await agent.process_query("List my lights", provider="openai")

    assert result["success"] is False
    assert "Legacy internal helper requests are disabled" in result["error"]


@pytest.mark.asyncio
async def test_process_query_requires_available_mcp_tools(mock_hass):
    """Queries should fail fast when official MCP tools are unavailable."""
    if not HOMEASSISTANT_AVAILABLE:
        pytest.skip("Home Assistant not available")

    from custom_components.ai_agent_ha.agent import AiAgentHaAgent
    from custom_components.ai_agent_ha.const import DOMAIN

    config = _agent_config()
    mock_hass.data[DOMAIN] = {"configs": {"openai": config}}

    mock_bridge = MagicMock()
    mock_bridge.async_list_tools = AsyncMock(return_value=[])

    mock_client = MagicMock()
    mock_client.get_response = AsyncMock(
        return_value=json.dumps(
            {"request_type": "final_response", "response": "This should not be used."}
        )
    )

    with patch("custom_components.ai_agent_ha.agent.OpenAIClient", return_value=mock_client):
        agent = AiAgentHaAgent(mock_hass, config, mcp_bridge=mock_bridge)
        result = await agent.process_query("Turn on kitchen lights", provider="openai")

    assert result["success"] is False
    assert "Official mcp_server tools are unavailable" in result["error"]
