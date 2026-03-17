"""Google Gemini API client using the google-genai SDK."""

import asyncio
import json
import logging

from google import genai
from google.genai import errors, types

from .base import BaseAIClient

_LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class GeminiClient(BaseAIClient):
    """Gemini client using the official google-genai SDK."""

    def __init__(self, token, model="gemini-2.5-flash"):
        self.token = token.strip() if token else token
        self.model = model
        self._client = None
        self._timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        self._max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS

    async def get_response(self, messages, **kwargs):
        """Send messages to Gemini and return a response string."""
        if not self.token:
            raise Exception("Missing Gemini API key")

        # Lazy initialization keeps startup cheap and avoids unnecessary SDK work.
        if self._client is None:
            try:
                self._client = await asyncio.to_thread(genai.Client, api_key=self.token)
            except Exception as e:
                _LOGGER.error("Failed to initialize Gemini client: %s", e)
                raise Exception(f"Gemini client initialization failed: {e}")

        timeout_seconds = int(kwargs.get("timeout_seconds", self._timeout_seconds))
        max_output_tokens = int(kwargs.get("max_output_tokens", self._max_output_tokens))
        max_output_tokens = max(256, min(max_output_tokens, 8192))

        _LOGGER.debug(
            "Gemini: request to model %s with %d messages (timeout=%ss, max_output_tokens=%d)",
            self.model,
            len(messages),
            timeout_seconds,
            max_output_tokens,
        )

        system_instruction = None
        contents: list[types.Content] = []

        # Keep payload lean by skipping empty turns and mapping only required roles.
        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()

            if role == "system":
                if content:
                    system_instruction = content
            elif role == "user":
                if content:
                    contents.append(
                        types.UserContent(parts=[types.Part.from_text(text=content)])
                    )
            elif role == "assistant":
                if content:
                    contents.append(
                        types.ModelContent(parts=[types.Part.from_text(text=content)])
                    )
            elif role == "tool":
                # Preserve legacy tool context as plain text without re-enabling
                # function-calling transport overhead.
                if content:
                    contents.append(
                        types.UserContent(
                            parts=[
                                types.Part.from_text(text=f"Tool result: {content}")
                            ]
                        )
                    )

        if not contents:
            raise Exception("No content messages to send to Gemini")

        try:
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            )

            # Sync SDK call is wrapped to keep HA's event loop non-blocking.
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            _LOGGER.error("Gemini request timed out after %ss", timeout_seconds)
            raise Exception(f"Gemini request timed out after {timeout_seconds}s") from e
        except errors.APIError as e:
            _LOGGER.error(
                "Gemini API error (code=%s): %s", getattr(e, "code", "unknown"), e
            )
            raise
        except Exception as e:
            _LOGGER.error("Gemini SDK error: %s", e, exc_info=True)
            raise

        _LOGGER.debug(
            "Gemini: %d candidate(s)",
            len(response.candidates) if response.candidates else 0,
        )

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason and "SAFETY" in str(finish_reason).upper():
                _LOGGER.warning(
                    "Gemini: response blocked by safety filter (%s)", finish_reason
                )
                return json.dumps(
                    {
                        "request_type": "final_response",
                        "response": (
                            "I'm sorry, my response was blocked by the AI "
                            "safety filter. Please try rephrasing your request."
                        ),
                    }
                )

        text = self._extract_response_text(response)
        if text:
            return text

        _LOGGER.warning("Gemini: empty response (no usable text)")
        return json.dumps(
            {
                "request_type": "final_response",
                "response": (
                    "I'm sorry, I received an empty response from the AI "
                    "model. This may be a temporary issue — please try again."
                ),
            }
        )

    @staticmethod
    def _extract_response_text(response: types.GenerateContentResponse) -> str | None:
        """Extract text from Gemini response candidates safely."""
        try:
            text = response.text
            if text and text.strip():
                return text.strip()
        except (AttributeError, ValueError):
            pass

        if not response.candidates:
            return None

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return None

        chunks: list[str] = []
        for part in candidate.content.parts:
            part_text = getattr(part, "text", None)
            if part_text and part_text.strip():
                chunks.append(part_text.strip())

        if not chunks:
            return None

        return "\n".join(chunks)
