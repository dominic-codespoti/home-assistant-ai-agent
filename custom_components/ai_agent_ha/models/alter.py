"""Alter API client."""

import json
import logging

import aiohttp

from .base import BaseAIClient

_LOGGER = logging.getLogger(__name__)


class AlterClient(BaseAIClient):
    def __init__(self, token, model=""):
        self.token = token
        self.model = model
        self.api_url = "https://alterhq.com/api/v1/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Alter API with model: %s", self.model)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        _LOGGER.debug("Alter request payload: %s", json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Alter API error %d: %s", resp.status, error_text)
                    raise Exception(f"Alter API error {resp.status}")
                data = await resp.json()
                # Extract text from Alter response (OpenAI-compatible format)
                choices = data.get("choices", [])
                if not choices:
                    _LOGGER.warning("Alter response missing choices")
                    _LOGGER.debug("Full Alter response: %s", json.dumps(data, indent=2))
                    return str(data)
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", str(data))
                return str(data)
