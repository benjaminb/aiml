import uuid
from typing import Any, Optional
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event

import os
from pathlib import Path
from dotenv import load_dotenv

path = Path(__file__).resolve()
for parent in path.parents:
    env_file = parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment vars from: \033[1m{env_file}\033[0m")
assert 'GEMINI_API_KEY' in os.environ, "GEMINI_API_KEY not set in .env"

"""
SUB-AGENTS: need google-style docstrings?
"""


def booking_handler(request: str) -> str:
    """
    Args:
        request: the user's request for a booking
    Returns:
        a confirmation message that the booking was handled
    """
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'"


def info_handler(request: str) -> str:
    """
    Args:
        request: the user's request for information
    Returns:
        a message containing the requested information
    """
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'"


def unclear_handler(request: str) -> str:
    """
    Args:
        request: the user's unclear request
    Returns:
        a message indicating the request was unclear
    """
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Unclear Handler: Coordinator could not delegate request: '{request}'"

# Create tools from handlers
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

# Define agents
booking_agent = Agent(
    name="booker", 
    model="gemini-2.5-flash",
    tools=[booking_tool],
    description="Specialized agent that handles all flight and hotel requests by calling the booking tool."
)

info_agent = Agent(
    name="info", 
    model="gemini-2.5-flash",
    tools=[info_tool],
    description="Specialized agent that handles general information questions by calling the info tool."
)

coordinator_agent = Agent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction=(
        "You are the main coordinator. Your only task is to analyze"
        "incoming user requests "
        "and delegate them to the appropriate specialist agent."
        "Do not try to answer the user directly.\n"
        "- For any requests related to booking flights or hotels,"
        "delegate to the 'Booker' agent.\n"
        "- For all other general information questions, delegate"
        "to the 'Info' agent."
    ),
    description="Coordinator agent routes user requests to the correct specialist agent.",
    sub_agents=[booking_agent, info_agent],
)

async def run_coordinator(runner: InMemoryRunner, request: str):
    """Runs the coordinator agent with the given request, which delegates as it sees fit"""
    print(f"\n --- Running coordinator agent with request: '{request}' --- ")

    # Set up session
    final_result = ""
    try:
        user_id = 'some_id_123'
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id
        ) 
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            )
        ):
            if event.is_final_response():
                # Try to get text directly
                if hasattr(event.content, 'text') and event.content.text:
                    final_result = event.content.text
                # Fallback: iterate over parts to build response
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                break # is this necessary?
        print(f"Coordinator final response: {final_result}")
        return final_result
    except Exception as e:
        print(f"Exception occurred: {e}")
        return f"An exception occurred while processing your request: {e}"



    

async def main():
    print("--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
    print("Note: This requires Google ADK installed and authenticated.")

    # Verify Google ADK authenticates
    runner = InMemoryRunner(agent=coordinator_agent)

    result_a = await run_coordinator(runner, request="Book me a hotel in Paris.")
    print(f"Final Result A: {result_a}")

    result_b = await run_coordinator(runner, request="What is the the highest mountain in the world?")
    print(f"Final Result B: {result_b}")

    result_c = await run_coordinator(runner, request="Tell me a random fact.")
    print(f"Final Result C: {result_c}")

    result_d = await run_coordinator(runner, request="Find flights to Tokyo next month.")

if __name__ == "__main__":
    # import nest_asyncio
    # nest_asyncio.apply()
    # await main()
    import asyncio
    asyncio.run(main())
