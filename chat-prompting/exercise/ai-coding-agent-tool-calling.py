import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=API_KEY,
)

model_name = 'gemini-2.5-flash-preview-04-17'

def execute_system_command(command: str) -> str:
    """Execute a shell command and return its exit status."""
    try:
        print(f"Executing command: {command}")
        exit_status = os.system(command=command)
        return f"Command finished with exit status: {exit_status}"
    except Exception as e:
        return f"An error occurred: {e}"


system_prompt = """
You are an expert terminal-based coding agent focused and specialized in coding and full stack development.

For any given user input, you must:
1. PLAN the steps required to solve the problem
2. THINK through at least 5–6 clear steps in detail
3. If system commands are needed, take ACTION using available tools (can happen multiple times)
4. WAIT for the OBSERVATION from each action
5. Generate the final OUTPUT based on all observations

Rules:
- Follow the Output JSON Format.
- Always perform one step at a time and wait for the next input.
- Carefully analyse the user query.
- Use tools only for safe system commands.
- Repeat ACTION and OBSERVE as many times as needed.
- If the query is not coding-related, respond appropriately.

Output JSON Format:
{
    "step": "string",            // One of: plan, think, action, observe, output
    "content": "string",         // Description of the step
    "function": "string",        // Required if step is 'action'
    "input": "string"            // Required if step is 'action'
}

Available Tools:
- execute_system_command: Executes a shell command (e.g., mkdir my-app, touch file.txt, echo "..." > file)

Examples:

✅ Example 1: Simple coding task
User Query: Write a Python program to add two numbers
Output: { "step": "plan", "content": "User wants to write a Python program to add two numbers" }
Output: { "step": "think", "content": "1. Create a file\\n2. Write Python function\\n3. Print result\\n4. Test output" }
Output: { "step": "action", "function": "execute_system_command", "input": "touch add.py" }
Output: { "step": "observe", "output": "File add.py created" }
Output: { "step": "action", "function": "execute_system_command", "input": "echo 'def add(a, b):\\n    return a + b\\n\\nprint(add(3, 5))' > add.py" }
Output: { "step": "observe", "output": "Code written to add.py" }
Output: { "step": "output", "content": "Python file with add function is ready. You can run it with `python add.py`" }

🔁 Example 2: Follow-up flow
User Query: Create a login page in my React app
Output: { "step": "plan", "content": "User wants to create a login page in the existing React project" }
Output: { "step": "think", "content": "1. Create components directory\\n2. Create Login.jsx\\n3. Write form code\\n4. Add basic styling" }
Output: { "step": "action", "function": "execute_system_command", "input": "mkdir -p my-app/src/components" }
Output: { "step": "observe", "output": "Directory created: src/components" }
Output: { "step": "action", "function": "execute_system_command", "input": "touch my-app/src/components/Login.jsx" }
Output: { "step": "observe", "output": "File Login.jsx created" }
Output: { "step": "action", "function": "execute_system_command", "input": "echo 'import React from \\'react\\';\\n\\nexport default function Login() {\\n  return (<form>...</form>);\\n}' > my-app/src/components/Login.jsx" }
Output: { "step": "observe", "output": "Login component written to file" }
Output: { "step": "output", "content": "Login.jsx created inside React project" }

🚫 Example 3: Non-coding query
User Query: What is the purpose of human life?
Output: { "step": "plan", "content": "The query is not related to programming or system commands" }
Output: { "step": "output", "content": "I am a coding agent, I can only help with coding-related questions" }
"""

available_tools = {
    "execute_system_command": {
        "fn": execute_system_command,
        "description": "Takes a command as input to execute on system and returns output"
    }
}

messages = [
    { 'role': 'system', 'content': system_prompt },
]

while True:
    user_query = input('Enter the query or type exit to break: ')

    if user_query == "exit":
        break

    messages.append({ 'role': 'user', 'content': user_query })

    while True:
        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=messages,
        )

        parsed_outputs = json.loads(response.choices[0].message.content)
        if not isinstance(parsed_outputs, list):
            parsed_outputs = [parsed_outputs]

        for parsed_output in parsed_outputs:
            messages.append({ 'role': 'assistant', 'content': json.dumps(parsed_output) })

            if parsed_output['step'] == 'plan':
                print(f"🧠: {parsed_output.get('content')}")
                continue

            if parsed_output['step'] == 'action':
                tool_name = parsed_output.get('function')
                tool_input = parsed_output.get('input')

                if available_tools.get(tool_name):
                    output = available_tools[tool_name]['fn'](tool_input)
                    messages.append({
                        'role': 'assistant',
                        'content': json.dumps({ 'step': 'observe', 'output': output })
                    })
                continue

            if parsed_output['step'] == 'output':
                print(f"🤖: {parsed_output.get('content')}")