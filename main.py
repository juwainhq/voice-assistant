"""Voice assistant: listen with the microphone, talk to Gemini, speak the reply.

Flow:
    1. Capture speech from the microphone (SpeechRecognition + PyAudio).
    2. Send the transcript to the Gemini API (google-generativeai).
    3. Read the answer out loud (pyttsx3).

Before calling Gemini we use `requests` to confirm there is internet access,
since both speech recognition and Gemini need it.
"""

import sys

import google.generativeai as genai
import requests
import speech_recognition as sr
import pyttsx3

from config import GEMINI_API_KEY

WAKE_PHRASE = "hey assistant"  # optional filter; set to "" to accept any speech
LISTEN_TIMEOUT = 8  # seconds to wait for speech to start
LISTEN_PHRASE_LIMIT = 10  # max seconds of speech to capture per turn
GEMINI_MODEL = "gemini-1.5-flash"


def check_internet(url="https://www.google.com", timeout=3) -> bool:
    """Return True when we can reach the outside world."""
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.RequestException:
        return False


def init_tts() -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)  # speaking speed
    return engine


def init_gemini() -> genai.GenerativeModel:
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        sys.exit("Set your Gemini API key in config.py before running.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)


def listen(recognizer: sr.Recognizer, microphone: sr.Microphone) -> str | None:
    """Capture one utterance from the mic and return the transcript (or None)."""
    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(
                source,
                timeout=LISTEN_TIMEOUT,
                phrase_time_limit=LISTEN_PHRASE_LIMIT,
            )
        except sr.WaitTimeoutError:
            return None

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, I could not understand that.")
        return None
    except sr.RequestError:
        print("Speech recognition service is unavailable.")
        return None


def ask_gemini(model: genai.GenerativeModel, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()


def speak(engine: pyttsx3.Engine, text: str) -> None:
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()


def main() -> None:
    if not check_internet():
        sys.exit("No internet connection. The assistant needs online access.")

    tts = init_tts()
    model = init_gemini()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    speak(tts, "Hello! I am ready. How can I help you?")

    while True:
        text = listen(recognizer, microphone)
        if text is None:
            continue

        print(f"You said: {text}")

        if text in ("exit", "quit", "stop", "goodbye"):
            speak(tts, "Goodbye!")
            break

        if WAKE_PHRASE and not text.startswith(WAKE_PHRASE):
            continue

        prompt = text.removeprefix(WAKE_PHRASE).strip() if WAKE_PHRASE else text
        try:
            answer = ask_gemini(model, prompt)
        except Exception as error:  # noqa: BLE001 - keep the loop alive
            answer = f"Sorry, I ran into a problem: {error}"

        speak(tts, answer)


if __name__ == "__main__":
    main()
