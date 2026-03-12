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
        self.model_id = model
        self.client = None
        if self.token:
            self.client = genai.Client(api_key=self.token, http_options={'api_version': 'v1alpha'})

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to google-genai SDK with model: %s", self.model_id)

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
                    role="user",  # The new SDK often uses 'user' for function response parts or specialized roles
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=msg.get("name"),
                            response={"result": content}
                        )
                    )]
                ))

        # Tool definitions
        tools = []
        raw_tools = kwargs.get("tools")
        if raw_tools:
            function_declarations = []
            for t in raw_tools:
                if t.get("type") == "function":
                    func = t.get("function", {})
                    # Clean up parameters
                    params = func.get("parameters", {"type": "object", "properties": {}}).copy()
                    if "$schema" in params:
                        del params["$schema"]
                    
                    function_declarations.append(types.FunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=params
                    ))
            if function_declarations:
                tools = [types.Tool(function_declarations=function_declarations)]

        try:
            # Use native async support in google-genai
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                tools=tools,
                system_instruction=system_instruction
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config
            )

            # Process response
            tool_calls = []
            text_content = ""
            
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if fn := part.function_call:
                            # Convert to dict
                            args = {}
                            if fn.args:
                                for key, val in fn.args.items():
                                    args[key] = val
                                    
                            tool_calls.append({
                                "id": f"call_{int(time.time() * 1000)}", 
                                "name": fn.name,
                                "arguments": args
                            })
                        elif part.text is not None:
                            text_content += part.text

            if tool_calls:
                return json.dumps({
                    "request_type": "_mcp_tool_calls",
                    "tool_calls": tool_calls,
                    "content": text_content
                })

            # Return text content, falling back safely to avoid "None" string
            if text_content:
                return text_content

            # Try response.text but guard against None
            try:
                resp_text = response.text
                if resp_text is not None and resp_text.strip():
                    return resp_text
            except (AttributeError, ValueError):
                pass

            _LOGGER.warning(
                "Gemini returned empty/None response. Candidates: %s",
                response.candidates,
            )
            return "I'm sorry, I received an empty response from the AI. Please try again."

        except Exception as e:
            _LOGGER.error("google-genai SDK error: %s", str(e))
            raise e
