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
            self.client = genai.Client(api_key=self.token, http_options={'api_version': 'v1alpha'})

    @staticmethod
    def _sanitize_schema_for_gemini(schema: dict) -> dict:
        """Recursively remove JSON Schema keywords unsupported by Gemini.

        Gemini's function-calling API only supports a subset of JSON Schema.
        Keywords like ``oneOf``, ``anyOf``, ``allOf``, ``$ref``,
        ``additionalProperties``, ``$schema``, ``default``, ``format`` etc.
        cause silent failures or errors.
        """
        if not isinstance(schema, dict):
            return schema

        schema = schema.copy()

        # Keys that Gemini does not support at any level
        unsupported_keys = {
            "$schema", "$ref", "$id", "$defs",
            "additionalProperties", "patternProperties",
            "if", "then", "else",
            "default", "format", "examples",
            "title",
        }
        for key in unsupported_keys:
            schema.pop(key, None)

        # Handle oneOf / anyOf / allOf by flattening to the first option
        for combo_key in ("oneOf", "anyOf", "allOf"):
            if combo_key in schema:
                options = schema.pop(combo_key)
                if isinstance(options, list) and options:
                    # Pick the first option and merge it
                    first = options[0] if isinstance(options[0], dict) else {}
                    for k, v in first.items():
                        if k not in schema:
                            schema[k] = v
                # If schema has no "type" yet, default to string
                if "type" not in schema:
                    schema["type"] = "string"

        # Recurse into properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            cleaned_props = {}
            for prop_name, prop_val in schema["properties"].items():
                if isinstance(prop_val, dict):
                    cleaned_props[prop_name] = GeminiClient._sanitize_schema_for_gemini(prop_val)
                else:
                    cleaned_props[prop_name] = prop_val
            schema["properties"] = cleaned_props

        # Recurse into items (array types)
        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = GeminiClient._sanitize_schema_for_gemini(schema["items"])

        return schema

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to google-genai SDK with model: %s", self.model)

        if not self.token or not self.client:
            raise Exception("Missing Gemini API key or client initialization failed")

        # Extract system instruction and build history
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content or "")]))
            elif role == "assistant":
                parts = []
                if content:
                    parts.append(types.Part(text=content))
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        parts.append(types.Part(
                            function_call=types.FunctionCall(
                                name=tc["name"],
                                args=tc["arguments"]
                            )
                        ))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=msg.get("name"),
                            response={"result": content}
                        )
                    )]
                ))

        # Ensure we have at least one content message
        if not contents:
            _LOGGER.error("Gemini: No contents to send after processing %d messages", len(messages))
            raise Exception("No content messages to send to Gemini")

        _LOGGER.debug(
            "Gemini: Sending %d content messages (system_instruction: %s)",
            len(contents),
            "yes" if system_instruction else "no",
        )

        # Tool definitions
        tools = []
        raw_tools = kwargs.get("tools")
        if raw_tools:
            function_declarations = []
            for t in raw_tools:
                if t.get("type") == "function":
                    func = t.get("function", {})
                    params = func.get("parameters", {"type": "object", "properties": {}})
                    if params:
                        params = self._sanitize_schema_for_gemini(params)

                    try:
                        function_declarations.append(types.FunctionDeclaration(
                            name=func["name"],
                            description=func.get("description", ""),
                            parameters=params
                        ))
                    except Exception as e:
                        _LOGGER.warning(
                            "Gemini: Skipping tool '%s' due to schema error: %s",
                            func.get("name", "unknown"), e,
                        )
            if function_declarations:
                tools = [types.Tool(function_declarations=function_declarations)]
                _LOGGER.debug("Gemini: Sending %d function declarations", len(function_declarations))

        try:
            # Use native async support in google-genai
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                tools=tools if tools else None,
                system_instruction=system_instruction
            )

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            # ---- Debug logging of the full response structure ----
            _LOGGER.debug("Gemini raw response type: %s", type(response).__name__)

            # Check for prompt_feedback (safety blocks on the prompt itself)
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                _LOGGER.debug("Gemini prompt_feedback: %s", response.prompt_feedback)

            num_candidates = len(response.candidates) if response.candidates else 0
            _LOGGER.debug("Gemini: %d candidates in response", num_candidates)

            if response.candidates:
                for i, cand in enumerate(response.candidates):
                    finish_reason = getattr(cand, 'finish_reason', None)
                    _LOGGER.debug(
                        "Gemini candidate[%d]: finish_reason=%s, has_content=%s",
                        i,
                        finish_reason,
                        cand.content is not None if hasattr(cand, 'content') else 'N/A',
                    )
                    if hasattr(cand, 'content') and cand.content:
                        num_parts = len(cand.content.parts) if cand.content.parts else 0
                        _LOGGER.debug(
                            "Gemini candidate[%d]: role=%s, %d parts",
                            i,
                            getattr(cand.content, 'role', 'N/A'),
                            num_parts,
                        )
                        if cand.content.parts:
                            for j, part in enumerate(cand.content.parts):
                                part_attrs = []
                                if part.text is not None:
                                    part_attrs.append(f"text({len(part.text)} chars)")
                                if part.function_call:
                                    part_attrs.append(f"function_call({part.function_call.name})")
                                if hasattr(part, 'function_response') and part.function_response:
                                    part_attrs.append("function_response")
                                _LOGGER.debug(
                                    "Gemini candidate[%d].parts[%d]: %s",
                                    i, j,
                                    ", ".join(part_attrs) if part_attrs else "empty/unknown part type",
                                )
                    elif hasattr(cand, 'safety_ratings') and cand.safety_ratings:
                        _LOGGER.debug(
                            "Gemini candidate[%d] safety_ratings: %s",
                            i, cand.safety_ratings,
                        )

            # Also try response.text as a quick check
            try:
                resp_text_check = response.text
                _LOGGER.debug(
                    "Gemini response.text: %s",
                    repr(resp_text_check[:200]) if resp_text_check else "None",
                )
            except (AttributeError, ValueError) as e:
                _LOGGER.debug("Gemini response.text access error: %s", e)

            # ---- Process the response ----
            tool_calls = []
            text_content = ""

            if response.candidates:
                candidate = response.candidates[0]

                # Check finish_reason for blocked responses
                finish_reason = getattr(candidate, 'finish_reason', None)
                # finish_reason can be an enum or string depending on SDK version
                finish_reason_str = str(finish_reason).upper() if finish_reason else ""
                if "SAFETY" in finish_reason_str:
                    _LOGGER.warning("Gemini response blocked by safety filter: %s", finish_reason)
                    return json.dumps({
                        "request_type": "final_response",
                        "response": "I'm sorry, my response was blocked by the AI safety filter. Please try rephrasing your request."
                    })
                if "RECITATION" in finish_reason_str:
                    _LOGGER.warning("Gemini response blocked for recitation: %s", finish_reason)

                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Check function_call first
                        fn = part.function_call
                        if fn and fn.name:
                            args = {}
                            if fn.args:
                                for key, val in fn.args.items():
                                    args[key] = val

                            tool_calls.append({
                                "id": f"call_{int(time.time() * 1000)}",
                                "name": fn.name,
                                "arguments": args
                            })
                        # Then check text
                        elif part.text is not None and part.text != "":
                            text_content += part.text
                else:
                    _LOGGER.warning(
                        "Gemini candidate has no content or no parts. finish_reason=%s",
                        finish_reason,
                    )

            if tool_calls:
                return json.dumps({
                    "request_type": "_mcp_tool_calls",
                    "tool_calls": tool_calls,
                    "content": text_content
                })

            # Return text content if we got any
            if text_content.strip():
                return text_content

            # Fallback: try response.text property (aggregates all text parts)
            try:
                resp_text = response.text
                if resp_text is not None and resp_text.strip():
                    _LOGGER.debug("Using response.text fallback (%d chars)", len(resp_text))
                    return resp_text
            except (AttributeError, ValueError):
                pass

            # Nothing worked — return a structured error so the caller can handle it
            _LOGGER.warning(
                "Gemini returned no usable text or tool calls. candidates=%d, finish_reason=%s",
                num_candidates,
                finish_reason_str if response.candidates else "no_candidates",
            )
            return json.dumps({
                "request_type": "final_response",
                "response": "I'm sorry, I received an empty response from the AI model. This may be a temporary issue — please try again."
            })

        except Exception as e:
            _LOGGER.error("google-genai SDK error: %s", str(e), exc_info=True)
            raise
