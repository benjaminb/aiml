from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
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
SUB-AGENTS
"""


def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request."""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'"


def info_handler(request: str) -> str:
    """Simulates the Information Agent handling a request."""
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'"


def unclear_handler(request: str) -> str:
    """Handles unclear requests."""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Unclear Handler: Coordinator could not delegate request: '{request}'"


def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    print(f"Model initialized: {llm.model}")

    # Set up the coordinator chain
    coordinator_router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Analyze the user's request and determine which specialist handler should process it.
- If the request is related to booking flights or hotels, output 'booker'.
- For all other general information questions, output 'info'.
- If the request is unclear or doesn't fit either category, output 'unclear'.
ONLY output one word: 'booker', 'info', or 'unclear'."""),
            ("user", "{request}")
        ]
    )
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

    # Use RunnableBranch to delegate to the appropriate handler
    branches = {
        "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),
        "info": RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])),
        "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])),
    }
    delegation_branch = RunnableBranch(
        (lambda x: x['decision'].strip() == 'booker', branches['booker']),
        (lambda x: x['decision'].strip() == 'info', branches['info']),
        branches['unclear']
    )

    # Maps the output word of router chain to "decision", AND passes through the original request as 'request'
    coordinator_agent = {
        "decision": coordinator_router_chain,
        "request": RunnablePassthrough()
    } | delegation_branch | (lambda x: x['output'])

    print("--- Running with a booking request ---")

    request_a = "Book me a flight to London."
    result_a = coordinator_agent.invoke({"request": request_a})
    print(f"Final Result A: {result_a}")
    print("\n--- Running with an info request ---")
    request_b = "What is the capital of Italy?"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Final Result B: {result_b}")
    print("\n--- Running with an unclear request ---")
    request_c = "Tell me about quantum physics."
    result_c = coordinator_agent.invoke({"request": request_c})
    print(f"Final Result C: {result_c}")


if __name__ == "__main__":
    main()
