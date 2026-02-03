import os
from dotenv import load_dotenv
from src.agent import agent  # Import the agent you defined earlier

load_dotenv()


def chat():
    print("--- Project Titan Assistant (Type 'exit' to quit) ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # run_sync is perfect for a simple CLI loop
        result = agent.run_sync(user_input)

        print(f"\nAI: {result.data}")
        print("-" * 20)


if __name__ == "__main__":
    chat()
