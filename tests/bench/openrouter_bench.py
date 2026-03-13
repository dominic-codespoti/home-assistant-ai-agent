import asyncio
import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from custom_components.ai_agent_ha.models.openrouter import OpenRouterClient

async def main():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Load .env
    env_path = root_path / ".env"
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    print(f"Testing OpenRouter with key: {api_key[:10]}...")
    
    client = OpenRouterClient(token=api_key, model="openai/gpt-4o")
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    print("\n--- Testing Plain Text Response ---")
    try:
        response = await client.get_response(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing Tool Call ---")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]
    
    messages = [
        {"role": "user", "content": "What is the weather in London?"}
    ]
    
    try:
        response = await client.get_response(messages, tools=tools)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
