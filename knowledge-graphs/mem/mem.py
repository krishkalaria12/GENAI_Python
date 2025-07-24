from mem0 import Memory
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Get API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for required API key
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    print("Please add GOOGLE_API_KEY to your .env file")
    sys.exit(1)

# Configuration
QUADRANT_HOST = os.getenv("QUADRANT_HOST", "localhost")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "reform-william-center-vibrate-press-5829")

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "google",
        "config": {
            "api_key": GOOGLE_API_KEY,
            "model": "models/gemini-embedding-exp-03-07"
        },
    },
    "llm": {
        "provider": "google",
        "config": {
            "api_key": GOOGLE_API_KEY,
            "model": "gemini-2.5-flash"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": 6333,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URL, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
}

mem_client = Memory.from_config(config)

# Initialize Gemini model
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    # Test the model with a simple query to ensure it's working
    test_response = gemini_model.generate_content("Hello")
    print("✓ Gemini 2.5 Flash model initialized successfully")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    print("Please check your API key and model availability")
    sys.exit(1)

def chat(message):
    try:
        mem_result = mem_client.search(query=message, user_id="p123")
        print("mem_result:", mem_result)
        memories = "\n".join([m["memory"] for m in mem_result.get("results", [])])
        print(f"\n\nMEMORY:\n\n{memories}\n\n")
        
        SYSTEM_PROMPT = f"""
            You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
            systematically analyze input content, extract structured knowledge, and maintain an
            optimized memory store. Your primary function is information distillation
            and knowledge preservation with contextual awareness.
            Tone: Professional analytical, precision-focused, with clear uncertainty signaling
            
            Memory and Score:
            {memories}
        """
        
        # Format the prompt for Gemini (it doesn't use separate system/user roles like OpenAI)
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {message}"
        
        # Generate response using Gemini
        response = gemini_model.generate_content(full_prompt)
        assistant_response = response.text
        
        # Format messages for mem0 storage
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_response}
        ]
        
        # Add to memory
        mem_client.add(messages, user_id="p123")
        return assistant_response
        
    except Exception as e:
        return f"Error processing message: {e}"

# Chat loop
while True:
    try:
        message = input(">> ")
        if message.lower() in ['quit', 'exit', 'bye']:
            break
        print("BOT:", chat(message=message))
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")