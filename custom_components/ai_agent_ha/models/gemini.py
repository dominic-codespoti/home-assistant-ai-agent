"""Google Gemini API client using the google-genai SDK.

Coded strictly against https://googleapis.github.io/python-genai/
See: Function Calling (manual), GenerateContentConfig, Content/Part types.
"""

import asyncio
import json
import logging

from google import genai
from google.genai import errors, types

from .base import BaseAIClient

_LOGGER = logging.getLogger(__name__)

# Counter for generating unique tool-call IDs within a session
_call_counter = 0


class GeminiClient(BaseAIClient):
    """Gemini client using the official google-genai SDK.

    Uses manual function calling (not automatic) because the agent orchestrator
    in agent.py handles MCP tool execution and conversation looping.
    """

    def __init__(self, token, model="gemini-2.5-flash"):
        self.token = token.strip() if token else token
        self.model = model
        self._client = None

    async def get_response(self, messages, **kwargs):
        """Send messages to Gemini and return a response string."""
        if not self.token:
            raise Exception("Missing Gemini API key")

        # Lazy initialization of the client if not already done
        if self._client is None:
            # Running synchronous genai.Client() in a thread to avoid blocking HA event loop
            try:
                self._client = await asyncio.to_thread(
                    genai.Client, api_key=self.token
                )
            except Exception as e:
                _LOGGER.error("Failed to initialize Gemini client: %s", e)
                raise Exception(f"Gemini client initialization failed: {e}")

        _LOGGER.debug("Gemini: request to model %s with %d messages", self.model, len(messages))

        # ── Build contents from conversation history ──
        # Per docs: list[types.Content] is the canonical format.
        # Roles: 'user' for user messages, 'model' for assistant, 'tool' for function responses.
        system_instruction = None
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                # Per docs: system_instruction goes in GenerateContentConfig, not contents
                system_instruction = content

            elif role == "user":
                contents.append(
                    types.UserContent(
                        parts=[types.Part.from_text(text=content or "")]
                    )
                )

            elif role == "assistant":
                # Per docs: assistant maps to role='model' (types.ModelContent)
                parts = []
                if content:
                    parts.append(types.Part.from_text(text=content))
                # Re-attach function call parts from previous turns
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        parts.append(
                            types.Part.from_function_call(
                                name=tc["name"],
                                args=tc["arguments"],
                            )
                        )
                if parts:
                    contents.append(types.ModelContent(parts=parts))

            elif role == "tool":
                # Per docs: role='tool' MUST contain a Part.from_function_response for 
                # EACH Part.from_function_call in the preceding 'model' turn.
                # If tool messages are adjacent in history, we group them.
                if contents and contents[-1].role == "tool":
                    contents[-1].parts.append(
                        types.Part.from_function_response(
                            name=msg.get("name"),
                            response={"result": content},
                        )
                    )
                else:
                    contents.append(
                        types.Content(
                            role="tool",
                            parts=[
                                types.Part.from_function_response(
                                    name=msg.get("name"),
                                    response={"result": content},
                                )
                            ],
                        )
                    )

        if not contents:
            raise Exception("No content messages to send to Gemini")

        # ── Build tool declarations ──
        # Per docs: types.FunctionDeclaration with parameters_json_schema for raw JSON schemas
        # Wrapped in types.Tool(function_declarations=[...])
        tools = None
        raw_tools = kwargs.get("tools")
        if raw_tools:
            declarations = []
            for t in raw_tools:
                if t.get("type") == "function":
                    func = t.get("function", {})
                    params = func.get("parameters", {"type": "object", "properties": {}})
                    try:
                        declarations.append(
                            types.FunctionDeclaration(
                                name=func["name"],
                                description=func.get("description", ""),
                                parameters_json_schema=params,
                            )
                        )
                    except Exception as e:
                        _LOGGER.warning(
                            "Gemini: skipping tool '%s' — schema error: %s",
                            func.get("name", "?"),
                            e,
                        )
            if declarations:
                tools = [types.Tool(function_declarations=declarations)]
                _LOGGER.debug("Gemini: %d function declarations", len(declarations))

        # ── Call the API ──
        # Per docs: GenerateContentConfig holds temperature, tools, system_instruction etc.
        # WORKAROUND: We use the synchronous client wrapped in asyncio.to_thread
        # to bypass a known 'grpcio' bug on Windows that causes async calls to fail
        # with "TypeError: Channel.getaddrinfo() takes 4 positional arguments but 5 were given".
        try:
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                tools=tools,
                system_instruction=system_instruction,
            )

            # Sync call executed in a separate thread
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config,
            )
        except errors.APIError as e:
            _LOGGER.error("Gemini API error (code=%s): %s", e.code, e.message)
            raise
        except Exception as e:
            _LOGGER.error("Gemini SDK error: %s", e, exc_info=True)
            raise

        # ── Process response ──
        _LOGGER.debug(
            "Gemini: %d candidate(s)",
            len(response.candidates) if response.candidates else 0,
        )

        # 1) Check for function calls — SDK convenience property
        # Per docs: response.function_calls returns list[FunctionCall] or None
        fn_calls = response.function_calls
        if fn_calls:
            global _call_counter
            tool_calls = []
            for fc in fn_calls:
                _call_counter += 1
                tool_calls.append(
                    {
                        "id": f"call_{_call_counter}",
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    }
                )
            # Check if there's actually any text to avoid SDK warnings
            text_content = ""
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                    part = candidate.content.parts[0]
                    if hasattr(part, "text") and part.text:
                        try:
                            text_content = response.text or ""
                        except (AttributeError, ValueError):
                            pass

            return json.dumps(
                {
                    "request_type": "_mcp_tool_calls",
                    "tool_calls": tool_calls,
                    "content": text_content,
                }
            )

        # 2) Check for safety blocks via finish_reason
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            fr = getattr(candidate, "finish_reason", None)
            if fr and "SAFETY" in str(fr).upper():
                _LOGGER.warning("Gemini: response blocked by safety filter (%s)", fr)
                return json.dumps(
                    {
                        "request_type": "final_response",
                        "response": (
                            "I'm sorry, my response was blocked by the AI "
                            "safety filter. Please try rephrasing your request."
                        ),
                    }
                )

        # 3) Extract text — SDK's .text handles thinking models automatically
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                part = candidate.content.parts[0]
                if hasattr(part, "text") and part.text:
                    try:
                        text = response.text
                        if text and text.strip():
                            return text
                    except (AttributeError, ValueError):
                        pass

        # 4) Nothing usable
        _LOGGER.warning("Gemini: empty response (no text or tool calls)")
        return json.dumps(
            {
                "request_type": "final_response",
                "response": (
                    "I'm sorry, I received an empty response from the AI "
                    "model. This may be a temporary issue — please try again."
                ),
            }
        )
