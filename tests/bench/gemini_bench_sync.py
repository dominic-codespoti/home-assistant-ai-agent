import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from google import genai
from google.genai import types

def main():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Load .env
    env_path = root_path / ".env"
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        return

    print(f"Testing Gemini (Sync) with key: {api_key[:10]}...")
    
    client = genai.Client(
        api_key=api_key,
    )

    print("\n--- Listing Available Models ---")
    try:
        for model in client.models.list():
            print(f"Model: {model}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Use a likely safe fallback for testing
    test_model = "gemini-2.5-flash"
    
    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text="What is the capital of France?")])
    ]
    
    print("\n--- Testing Plain Text Response ---")
    try:
        response = client.models.generate_content(
            model=test_model,
            contents=messages,
            config=types.GenerateContentConfig(temperature=0.7)
        )
        print(f"Response Text: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing Tool Call ---")
    tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get current weather in a location",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        )
    ])]
    
    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text="What is the weather in London?")])
    ]
    
    try:
        response = client.models.generate_content(
            model=test_model,
            contents=messages,
            config=types.GenerateContentConfig(
                temperature=0.7,
                tools=tools
            )
        )
        print(f"Response Function Calls: {response.function_calls}")
        if response.function_calls:
            for fc in response.function_calls:
                print(f"  Name: {fc.name}")
                print(f"  Args: {fc.args}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
