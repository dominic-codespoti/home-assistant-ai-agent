"""Anthropic Claude API client."""

import json
import logging

import aiohttp

from .base import BaseAIClient

_LOGGER = logging.getLogger(__name__)


class AnthropicClient(BaseAIClient):
    def __init__(self, token, model="claude-sonnet-4-5-20250929"):
        self.token = token
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Anthropic API with model: %s", self.model)
        headers = {
            "x-api-key": self.token,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Convert standardized messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # Anthropic uses a separate system parameter
                system_message = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if "tool_calls" in message:
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in message["tool_calls"]:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments"]
                        })
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": content
                    }]
                })

        payload = {
            "model": self.model,
            "max_tokens": 8192,  # Maximum for Anthropic Claude models
            "temperature": 0.7,
            "messages": anthropic_messages,
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message
            
        tools = kwargs.get("tools")
        if tools:
            anthropic_tools = []
            for t in tools:
                if t.get("type") == "function":
                    func = t.get("function", {})
                    anthropic_tools.append({
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                    })
            if anthropic_tools:
                payload["tools"] = anthropic_tools

        _LOGGER.debug("Anthropic request payload size: %d", len(json.dumps(payload)))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Anthropic API error %d: %s", resp.status, error_text[:500])
                    raise Exception(f"Anthropic API error {resp.status}")
                data = await resp.json()
                
                if data.get("stop_reason") == "tool_use":
                    tool_calls = []
                    text_content = ""
                    for block in data.get("content", []):
                        if block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "arguments": block.get("input")
                            })
                        elif block.get("type") == "text":
                            text_content += block.get("text", "")
                            
                    return json.dumps({
                        "request_type": "_mcp_tool_calls",
                        "tool_calls": tool_calls,
                        "content": text_content
                    })

                # Extract text from plain response
                content_blocks = data.get("content", [])
                if content_blocks and isinstance(content_blocks, list):
                    for block in content_blocks:
                        if block.get("type") == "text":
                            return block.get("text", str(data))
                return str(data)
