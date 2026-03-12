"""Z.AI API client."""

import json
import logging

import aiohttp

from .base import BaseAIClient

_LOGGER = logging.getLogger(__name__)


class ZaiClient(BaseAIClient):
    def __init__(self, token, model="", endpoint_type="general"):
        self.token = token
        self.model = model
        self.endpoint_type = endpoint_type
        # General endpoint: https://api.z.ai/api/paas/v4/chat/completions
        # Coding endpoint: https://api.z.ai/api/coding/paas/v4/chat/completions
        if endpoint_type == "coding":
            self.api_url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
        else:
            self.api_url = "https://api.z.ai/api/paas/v4/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug(
            "Making request to z.ai API with model: %s, endpoint: %s",
            self.model,
            self.endpoint_type,
        )
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

        _LOGGER.debug("z.ai request payload: %s", json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("z.ai API error %d: %s", resp.status, error_text)
                    raise Exception(f"z.ai API error {resp.status}")
                data = await resp.json()
                # Extract text from z.ai response (OpenAI-compatible format)
                choices = data.get("choices", [])
                if not choices:
                    _LOGGER.warning("z.ai response missing choices")
                    _LOGGER.debug("Full z.ai response: %s", json.dumps(data, indent=2))
                    return str(data)
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", str(data))
                return str(data)
