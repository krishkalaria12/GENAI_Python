import os

from langchain_core.tools import tool

@tool()
def run_command(cmd: str) -> str:
    """
        Takes a command line prompt and executes it on the user's machine and returns the output of the command.
        Example: run_command("ls -l") where ls is the command to list the files
    """

    result = os.system(command=cmd)
    return result