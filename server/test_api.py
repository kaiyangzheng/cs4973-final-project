import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

async def test_openai_api():
    # Load environment variables
    load_dotenv()
    
    # Get environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    model = os.environ.get("OPENAI_MODEL")
    
    print(f"API Key: {api_key}")
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    try:
        # Call the API
        print("Calling API...")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print("Response received!")
        print(f"Response: {response}")
        print(f"Content: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_openai_api()) 