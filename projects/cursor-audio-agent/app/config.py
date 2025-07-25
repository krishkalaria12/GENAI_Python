import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

# custom imports
from .tools import run_command

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
if not os.environ.get("MONGO_URI"):
    raise ValueError("MONGO_URI not found in environment variables. Please check your .env file.")


SYSTEM_PROMPT='''
    You are an AI coding assistant who takes an input from user and based on the available tools, you choose the correct tool and execute's the commands.

    Always make sure to keep your generated codes and files in the ai_codes/ folder. You can create one if not available. Make sure you call the tools and do the work. I am using a windows laptop, so make sure you use the correct commands for windows.

    You can even execute commands and help user with the output of the command.
    Also the mistakes which you are making is giving the command like mkdir -p ai_codes which is making 2 folders which is -p and ai_codes. But i dont need the -p so just use mkdir ai_codes instead.

    Available tools:
    - run_command: Takes a command line prompt and executes it on the user's machine and returns the output of the command.
'''

llm = init_chat_model(
    model_provider='google_genai',
    model='gemini-2.0-flash'
)

tools = [
    run_command
]

llm_with_tools = llm.bind_tools(tools=tools)

DB_URI = os.environ.get("MONGO_URI")

# Config for thread_id
checkpointer_mongo_config = {
    "configurable": {
        "thread_id": 2,
    }
}