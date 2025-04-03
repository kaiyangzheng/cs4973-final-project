import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables explicitly from .env file in the server directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Get environment variables with proper fallbacks
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "llama3p1-8b-instruct")

# Print the configuration for debugging
print(f"OpenAI Configuration:")
print(f"API KEY: {OPENAI_API_KEY}")
print(f"API BASE: {OPENAI_API_BASE}")
print(f"CHAT MODEL: {OPENAI_MODEL}")

# Initialize OpenAI client with appropriate base URL
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

async def call_llm(prompt: str) -> str:
    """Call the LLM with the given prompt and return the response"""
    try:
        # Try to call the LLM API
        print(f"Calling LLM API with model: {OPENAI_MODEL}")
        print(f"API Base URL: {OPENAI_API_BASE}")
        
        # Format messages for different API formats
        messages = [
            {"role": "system", "content": "You are a research paper analysis assistant that helps users understand complex academic papers. Provide clear explanations of technical concepts, methodologies, and findings. When appropriate, refer to related papers or research to provide broader context."},
            {"role": "user", "content": prompt}
        ]
        
        # Call the API
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        
        # Check response structure and extract content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
            else:
                print("API response missing message content structure")
                return "API response was missing expected content structure"
        else:
            print("API response missing choices")
            return "API response was missing expected choices"
            
    except Exception as e:
        error_msg = str(e)
        print(f"LLM API Error: {error_msg}")
        
        # Return a user-friendly error message
        if "API key" in error_msg.lower():
            return "API authentication error. Please check your API key configuration."
        elif "not found" in error_msg.lower() or "no such" in error_msg.lower():
            return f"Model '{OPENAI_MODEL}' not found or not available. Please check your model configuration."
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return "The API request timed out. The server might be busy, please try again later."
        else:
            return f"I couldn't process your request due to an API error: {error_msg}" 