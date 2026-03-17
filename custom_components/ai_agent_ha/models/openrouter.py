"""OpenRouter API client."""

import json
import logging

import aiohttp

from .base import BaseAIClient

_LOGGER = logging.getLogger(__name__)


class OpenRouterClient(BaseAIClient):
    def __init__(self, token, model="openai/gpt-4o"):
        self.token = token
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to OpenRouter API with model: %s", self.model)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://home-assistant.io",  # Optional for OpenRouter rankings
            "X-Title": "Home Assistant AI Agent",  # Optional for OpenRouter rankings
        }

        # Same mapping logic as OpenAI
        openai_messages = []
        for msg in messages:
            api_msg = {"role": msg.get("role", "user")}
            if "content" in msg and msg["content"] is not None:
                api_msg["content"] = msg["content"]
            openai_messages.append(api_msg)

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        _LOGGER.debug("OpenRouter request payload size limit: %d", len(json.dumps(payload)))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("OpenRouter API error %d: %s", resp.status, error_text[:500])
                    raise Exception(f"OpenRouter API error {resp.status}")
                data = await resp.json()

                choices = data.get("choices", [])
                if not choices:
                    return str(data)
                
                if choices and "message" in choices[0]:
                    msg = choices[0]["message"]
                    return msg.get("content", str(data))
                return str(data)
