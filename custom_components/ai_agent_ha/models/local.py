"""Local AI client for self-hosted models (Ollama, etc.)."""

import json
import logging

import aiohttp

from .base import BaseAIClient
from ..utils import sanitize_for_logging

_LOGGER = logging.getLogger(__name__)


class LocalClient(BaseAIClient):
    def __init__(self, url, model=""):
        self.url = url
        self.model = model

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug(
            "Making request to local API with model: '%s' at URL: %s",
            self.model or "[NO MODEL SPECIFIED]",
            self.url,
        )

        if not self.model:
            _LOGGER.warning(
                "No model specified for local API request. Some APIs (like Ollama) require a model name."
            )
        headers = {"Content-Type": "application/json"}

        # Format user prompt from messages
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Simple formatting: prefixing each message with its role
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        # Add final prompt prefix for the assistant's response
        prompt += "Assistant: "

        # Build a generic payload that works with most local API servers
        payload = {
            "prompt": prompt,
            "stream": False,  # Disable streaming to get a single complete response
            # max_tokens omitted - let local model use its default capacity
        }

        # Add model if specified
        if self.model:
            payload["model"] = self.model

        # Note: Payloads don't contain auth tokens (those are in headers), but may contain user prompts
        _LOGGER.debug("Local API request payload: %s", json.dumps(payload, indent=2))

        # Ollama-specific validation
        if "model" not in payload or not payload["model"]:
            _LOGGER.warning(
                "Missing 'model' field in request to local API. This may cause issues with Ollama."
            )
        elif self.url and "ollama" in self.url.lower():
            _LOGGER.debug(
                "Detected Ollama URL, ensuring model is specified: %s",
                payload.get("model"),
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Local API error %d: %s", resp.status, error_text)

                    # Provide more specific error messages for common Ollama issues
                    if resp.status == 404:
                        if "model" in payload and payload["model"]:
                            raise Exception(
                                f"Model '{payload['model']}' not found. Please ensure the model is installed in Ollama using: ollama pull {payload['model']}"
                            )
                        else:
                            raise Exception(
                                "Local API endpoint not found. Please check the URL and ensure Ollama is running."
                            )
                    elif resp.status == 400:
                        raise Exception(
                            f"Bad request to local API. Error: {error_text}"
                        )
                    else:
                        raise Exception(f"Local API error {resp.status}: {error_text}")

                try:
                    response_text = await resp.text()
                    _LOGGER.debug(
                        "Local API response (first 200 chars): %s", response_text[:200]
                    )
                    _LOGGER.debug("Local API response status: %d", resp.status)
                    # Sanitize headers to avoid logging any auth tokens
                    _LOGGER.debug(
                        "Local API response headers: %s",
                        sanitize_for_logging(dict(resp.headers)),
                    )

                    # Try to parse as JSON
                    try:
                        data = json.loads(response_text)

                        # Try common response formats
                        # Ollama format - return only the response text
                        if "response" in data:
                            response_content = data["response"]
                            _LOGGER.debug(
                                "Extracted response content: %s",
                                (
                                    response_content[:100]
                                    if response_content
                                    else "[EMPTY]"
                                ),
                            )

                            # Check if response is empty or None
                            if not response_content or response_content.strip() == "":
                                _LOGGER.warning(
                                    "Ollama returned empty response. Full data: %s",
                                    data,
                                )
                                # Check if this is a loading response
                                if data.get("done_reason") == "load":
                                    _LOGGER.warning(
                                        "Ollama is still loading the model. Please wait and try again."
                                    )
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI model is still loading. Please wait a moment and try again.",
                                        }
                                    )
                                elif data.get("done") is False:
                                    _LOGGER.warning(
                                        "Ollama response indicates it's not done yet."
                                    )
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI is still processing your request. Please try again.",
                                        }
                                    )
                                else:
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI returned an empty response. Please try rephrasing your question.",
                                        }
                                    )

                            # Check if the response looks like JSON
                            response_content = response_content.strip()
                            if response_content.startswith(
                                "{"
                            ) and response_content.endswith("}"):
                                try:
                                    # Validate that it's actually JSON and contains valid request_type
                                    parsed_json = json.loads(response_content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response"
                                        )
                                        return response_content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text"
                                    )
                                    pass

                            # If it's plain text, wrap it in the expected JSON format
                            wrapped_response = {
                                "request_type": "final_response",
                                "response": response_content,
                            }
                            _LOGGER.debug("Wrapped plain text response in JSON format")
                            return json.dumps(wrapped_response)

                        # OpenAI-like format
                        elif "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                content = choice["message"]["content"]
                            elif "text" in choice:
                                content = choice["text"]
                            else:
                                content = str(data)

                            # Check if it's valid JSON with request_type
                            content = content.strip()
                            if content.startswith("{") and content.endswith("}"):
                                try:
                                    parsed_json = json.loads(content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response (OpenAI format)"
                                        )
                                        return content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text (OpenAI format)"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text (OpenAI format)"
                                    )
                                    pass

                            # Wrap in expected format if plain text
                            wrapped_response = {
                                "request_type": "final_response",
                                "response": content,
                            }
                            return json.dumps(wrapped_response)

                        # Generic content field
                        elif "content" in data:
                            content = data["content"]
                            content = content.strip()
                            if content.startswith("{") and content.endswith("}"):
                                try:
                                    parsed_json = json.loads(content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response (generic format)"
                                        )
                                        return content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text (generic format)"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text (generic format)"
                                    )
                                    pass

                            wrapped_response = {
                                "request_type": "final_response",
                                "response": content,
                            }
                            return json.dumps(wrapped_response)

                        # Handle case where no standard fields are found
                        _LOGGER.warning(
                            "No standard response fields found in local API response. Full response: %s",
                            data,
                        )

                        # Check for Ollama-specific edge cases
                        if data.get("done_reason") == "load":
                            return json.dumps(
                                {
                                    "request_type": "final_response",
                                    "response": "The AI model is still loading. Please wait a moment and try again.",
                                }
                            )
                        elif data.get("done") is False:
                            return json.dumps(
                                {
                                    "request_type": "final_response",
                                    "response": "The AI is still processing your request. Please try again.",
                                }
                            )
                        elif "message" in data:
                            # Some APIs use "message" field
                            message_content = data["message"]
                            if (
                                isinstance(message_content, dict)
                                and "content" in message_content
                            ):
                                content = message_content["content"]
                            else:
                                content = str(message_content)
                            return json.dumps(
                                {"request_type": "final_response", "response": content}
                            )

                        # Return the whole data as string if we can't find a specific field
                        return json.dumps(
                            {
                                "request_type": "final_response",
                                "response": f"Received unexpected response format from local API: {str(data)}",
                            }
                        )

                    except json.JSONDecodeError:
                        # If not JSON, check if it's a JSON response that got corrupted by wrapping
                        response_text = response_text.strip()
                        if response_text.startswith("{") and response_text.endswith(
                            "}"
                        ):
                            try:
                                parsed_json = json.loads(response_text)
                                if (
                                    isinstance(parsed_json, dict)
                                    and "request_type" in parsed_json
                                ):
                                    _LOGGER.debug(
                                        "Local model provided valid JSON response (direct)"
                                    )
                                    return response_text
                            except json.JSONDecodeError:
                                pass

                        # If not valid JSON, wrap the raw text in expected format
                        _LOGGER.debug("Response is not JSON, wrapping plain text")
                        wrapped_response = {
                            "request_type": "final_response",
                            "response": response_text,
                        }
                        return json.dumps(wrapped_response)

                except Exception as e:
                    _LOGGER.error("Failed to parse local API response: %s", str(e))
                    raise Exception(f"Failed to parse local API response: {str(e)}")
