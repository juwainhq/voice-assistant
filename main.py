"""Nova — a simple text chat assistant powered by Google's Gemini API.

Flow:
    1. Import the API key from config.py and connect to Gemini (google-generativeai).
    2. Introduce the assistant on startup.
    3. Loop: read a typed question, print Gemini's answer.
    4. Exit when the user types "quit".
"""

import sys

import google.generativeai as genai

from config import GEMINI_API_KEY

ASSISTANT_NAME = "Nova"
GEMINI_MODEL = "gemini-1.5-flash"


def main() -> None:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        sys.exit("Set your Gemini API key in config.py before running.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    print(f"Hello! I'm {ASSISTANT_NAME}, your Gemini-powered assistant.")
    print("Ask me anything. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            print(f"{ASSISTANT_NAME}: Goodbye!")
            break

        try:
            response = model.generate_content(question)
            print(f"{ASSISTANT_NAME}: {response.text.strip()}\n")
        except Exception as error:  # noqa: BLE001 - keep the loop alive
            print(f"{ASSISTANT_NAME}: Sorry, I ran into a problem: {error}\n")


if __name__ == "__main__":
    main()
