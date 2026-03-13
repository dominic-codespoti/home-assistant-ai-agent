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

from custom_components.ai_agent_ha.models.llama import LlamaClient

async def main():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Load .env
    env_path = root_path / ".env"
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        print("Error: LLAMA_API_KEY not found in .env")
        return

    print(f"Testing Llama with key: {api_key[:10]}...")
    
    client = LlamaClient(token=api_key, model="llama3-70b-8192")
    
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
