"""Google Gemini API client using the google-genai SDK."""

import json
import logging
import time

from google import genai
from google.genai import types

from .base import BaseAIClient

_LOGGER = logging.getLogger(__name__)


class GeminiClient(BaseAIClient):
    def __init__(self, token, model="gemini-2.0-flash"):
        self.token = token.strip() if token else token
        self.model = model
        self.client = None
        if self.token:
            self.client = genai.Client(
                api_key=self.token,
                http_options=types.HttpOptions(api_version='v1alpha'),
            )

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to google-genai SDK with model: %s", self.model)

        if not self.token or not self.client:
            raise Exception("Missing Gemini API key or client initialization failed")

        # ── Build contents list from conversation history ──
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content

            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content or "")],
                    )
                )

            elif role == "assistant":
                parts = []
                if content:
                    parts.append(types.Part.from_text(text=content))
                # Re-attach any function-call parts the model made
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        parts.append(
                            types.Part.from_function_call(
                                name=tc["name"],
                                args=tc["arguments"],
                            )
                        )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))

            elif role == "tool":
                # SDK docs: role='tool' for function-response content
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
            _LOGGER.error(
                "Gemini: No contents to send after processing %d messages",
                len(messages),
            )
            raise Exception("No content messages to send to Gemini")

        _LOGGER.debug(
            "Gemini: Sending %d content messages (system_instruction: %s)",
            len(contents),
            "yes" if system_instruction else "no",
        )

        # ── Build tool definitions ──
        tools = None
        raw_tools = kwargs.get("tools")
        if raw_tools:
            function_declarations = []
            for t in raw_tools:
                if t.get("type") == "function":
                    func = t.get("function", {})
                    params = func.get(
                        "parameters",
                        {"type": "object", "properties": {}},
                    )
                    try:
                        # Use parameters_json_schema (not parameters) so the
                        # SDK sends the schema as raw JSON instead of trying to
                        # parse it into a Pydantic Schema object.  This allows
                        # JSON-Schema keywords like oneOf / anyOf that the
                        # Pydantic model rejects.
                        function_declarations.append(
                            types.FunctionDeclaration(
                                name=func["name"],
                                description=func.get("description", ""),
                                parameters_json_schema=params,
                            )
                        )
                    except Exception as e:
                        _LOGGER.warning(
                            "Gemini: Skipping tool '%s' due to schema error: %s",
                            func.get("name", "unknown"),
                            e,
                        )
            if function_declarations:
                tools = [types.Tool(function_declarations=function_declarations)]
                _LOGGER.debug(
                    "Gemini: Sending %d function declarations",
                    len(function_declarations),
                )

        # ── Call the Gemini API ──
        try:
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                tools=tools,
                system_instruction=system_instruction,
            )

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # ── Debug logging ──
            _LOGGER.debug("Gemini raw response type: %s", type(response).__name__)

            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                _LOGGER.debug("Gemini prompt_feedback: %s", response.prompt_feedback)

            num_candidates = len(response.candidates) if response.candidates else 0
            _LOGGER.debug("Gemini: %d candidates in response", num_candidates)

            if response.candidates:
                for i, cand in enumerate(response.candidates):
                    finish_reason = getattr(cand, "finish_reason", None)
                    _LOGGER.debug(
                        "Gemini candidate[%d]: finish_reason=%s, has_content=%s",
                        i,
                        finish_reason,
                        cand.content is not None
                        if hasattr(cand, "content")
                        else "N/A",
                    )
                    if hasattr(cand, "content") and cand.content and cand.content.parts:
                        for j, part in enumerate(cand.content.parts):
                            attrs = []
                            if part.text is not None:
                                attrs.append(f"text({len(part.text)} chars)")
                            if part.function_call:
                                attrs.append(
                                    f"function_call({part.function_call.name})"
                                )
                            if part.function_response:
                                attrs.append("function_response")
                            _LOGGER.debug(
                                "Gemini candidate[%d].parts[%d]: %s",
                                i,
                                j,
                                ", ".join(attrs) or "empty/unknown part",
                            )

            # Quick peek at response.text
            try:
                _LOGGER.debug(
                    "Gemini response.text preview: %s",
                    repr(response.text[:300]) if response.text else "None",
                )
            except (AttributeError, ValueError) as e:
                _LOGGER.debug("Gemini response.text access error: %s", e)

            # ── Extract function calls (SDK convenience property) ──
            fn_calls = response.function_calls
            if fn_calls:
                tool_calls = []
                for fc in fn_calls:
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        {
                            "id": f"call_{int(time.time() * 1000)}",
                            "name": fc.name,
                            "arguments": args,
                        }
                    )
                # Grab any accompanying text
                text_content = ""
                if response.candidates:
                    cand = response.candidates[0]
                    if cand.content and cand.content.parts:
                        for part in cand.content.parts:
                            if part.text:
                                text_content += part.text
                return json.dumps(
                    {
                        "request_type": "_mcp_tool_calls",
                        "tool_calls": tool_calls,
                        "content": text_content,
                    }
                )

            # ── Extract text ──
            # Check finish_reason for safety blocks
            finish_str = ""
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None)
                finish_str = str(finish_reason).upper() if finish_reason else ""
                if "SAFETY" in finish_str:
                    _LOGGER.warning(
                        "Gemini response blocked by safety filter: %s",
                        finish_reason,
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

            # Use the SDK's .text convenience (aggregates all text parts)
            try:
                text = response.text
                if text and text.strip():
                    return text
            except (AttributeError, ValueError):
                pass

            # Nothing usable
            _LOGGER.warning(
                "Gemini returned no usable text or tool calls. "
                "candidates=%d, finish_reason=%s",
                num_candidates,
                finish_str if response.candidates else "no_candidates",
            )
            return json.dumps(
                {
                    "request_type": "final_response",
                    "response": (
                        "I'm sorry, I received an empty response from the AI "
                        "model. This may be a temporary issue — please try again."
                    ),
                }
            )

        except Exception as e:
            _LOGGER.error("google-genai SDK error: %s", str(e), exc_info=True)
            raise
