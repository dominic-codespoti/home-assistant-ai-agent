"""OpenAI API client."""

import json
import logging

import aiohttp

from .base import BaseAIClient

_LOGGER = logging.getLogger(__name__)


class OpenAIClient(BaseAIClient):
    def __init__(self, token, model="gpt-3.5-turbo"):
        self.token = token
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def _is_restricted_model(self):
        """Check if the model has restricted parameters (no temperature, top_p, etc.)."""
        # Models that don't support temperature, top_p and other parameters
        restricted_models = ["o3-mini", "o3", "o1-mini", "o1-preview", "o1", "gpt-5"]

        model_lower = self.model.lower()
        return any(model_id in model_lower for model_id in restricted_models)

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to OpenAI API with model: %s", self.model)

        # Validate token
        if not self.token or not self.token.startswith("sk-"):
            raise Exception("Invalid OpenAI API key format")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        # Check if model has restricted parameters
        is_restricted = self._is_restricted_model()
        _LOGGER.debug(
            "Using model: %s (restricted parameters: %s)",
            self.model,
            is_restricted,
        )

        # Map internal standardized messages to OpenAI-compatible messages
        openai_messages = []
        for msg in messages:
            api_msg = {"role": msg.get("role", "user")}
            if "content" in msg and msg["content"] is not None:
                api_msg["content"] = msg["content"]
                
            if "tool_calls" in msg:
                api_msg["tool_calls"] = []
                for tc in msg["tool_calls"]:
                    api_msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                        }
                    })
            if msg.get("role") == "tool":
                api_msg["tool_call_id"] = msg.get("tool_call_id")
            
            # Ensure assistant messages with tool calls have at least null or empty content
            if api_msg["role"] == "assistant" and "tool_calls" in api_msg and "content" not in api_msg:
                api_msg["content"] = ""
                
            openai_messages.append(api_msg)

        # Build payload with model-appropriate parameters
        # Don't set max_tokens - let OpenAI use the model's maximum capacity
        payload = {"model": self.model, "messages": openai_messages}

        # Only add temperature and top_p for models that support them
        if not is_restricted:
            payload.update({"temperature": 0.7, "top_p": 0.9})
            
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools

        _LOGGER.debug("OpenAI request payload limit length: %s", len(json.dumps(payload)))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                response_text = await resp.text()
                _LOGGER.debug("OpenAI API response status: %d", resp.status)

                if resp.status != 200:
                    _LOGGER.error("OpenAI API error %d: %s", resp.status, response_text[:500])
                    raise Exception(f"OpenAI API error {resp.status}: {response_text[:200]}")

                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    _LOGGER.error("Failed to parse OpenAI response as JSON: %s", str(e))
                    raise Exception(
                        f"Invalid JSON response from OpenAI: {response_text[:200]}"
                    )

                # Extract text or tool calls from OpenAI response
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    msg = choices[0]["message"]
                    
                    if "tool_calls" in msg and msg["tool_calls"]:
                        # Return standardized internal tool_calls response
                        tool_calls = []
                        for tc in msg["tool_calls"]:
                            if tc.get("type") == "function":
                                fn = tc.get("function", {})
                                try:
                                    args = json.loads(fn.get("arguments", "{}"))
                                except json.JSONDecodeError:
                                    args = {}
                                tool_calls.append({
                                    "id": tc.get("id"),
                                    "name": fn.get("name"),
                                    "arguments": args
                                })
                        return json.dumps({
                            "request_type": "_mcp_tool_calls", 
                            "tool_calls": tool_calls,
                            "content": msg.get("content", "")
                        })

                    content = msg.get("content", "")
                    if not content:
                        _LOGGER.warning("OpenAI returned empty content in message")
                        _LOGGER.debug(
                            "Full OpenAI response: %s", json.dumps(data, indent=2)[:500]
                        )
                    return content
                else:
                    _LOGGER.warning("OpenAI response missing expected structure")
                    _LOGGER.debug(
                        "Full OpenAI response: %s", json.dumps(data, indent=2)[:500]
                    )
                    return str(data)
