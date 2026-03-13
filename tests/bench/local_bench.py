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

from custom_components.ai_agent_ha.models.local import LocalClient

async def main():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Load .env
    env_path = root_path / ".env"
    load_dotenv(dotenv_path=env_path)
    
    # Local usually doesn't need a token or uses a local endpoint
    api_key = os.getenv("LOCAL_API_KEY", "local-token")
    endpoint = os.getenv("LOCAL_ENDPOINT", "http://localhost:11434/v1")

    print(f"Testing Local AI at {endpoint}...")
    
    client = LocalClient(token=api_key, model="llama3")
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    print("\n--- Testing Plain Text Response ---")
    try:
        response = await client.get_response(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
