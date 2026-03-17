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
                anthropic_messages.append({"role": "assistant", "content": content})

        payload = {
            "model": self.model,
            "max_tokens": 8192,  # Maximum for Anthropic Claude models
            "temperature": 0.7,
            "messages": anthropic_messages,
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message
            
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

                # Extract text from plain response
                content_blocks = data.get("content", [])
                if content_blocks and isinstance(content_blocks, list):
                    for block in content_blocks:
                        if block.get("type") == "text":
                            return block.get("text", str(data))
                return str(data)
