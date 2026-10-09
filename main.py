"""Nova — a Gemini-powered assistant with voice input (sounddevice, no PyAudio).

Flow:
    1. Import the API key from config.py and connect to Gemini (google-generativeai).
    2. Introduce the assistant on startup.
    3. Loop: record a spoken question from the microphone via sounddevice,
       transcribe it to text, print Gemini's answer.
    4. Exit when the user says "quit".

Microphone capture uses `sounddevice` instead of PyAudio so the project
installs cleanly on newer Python versions (e.g. Python 3.14 in Codespaces).
SpeechRecognition is only used for speech-to-text, which does not need PyAudio.
"""

import math
import sys
import time

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import google.generativeai as genai

from config import GEMINI_API_KEY

ASSISTANT_NAME = "Nova"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Microphone recording settings
SAMPLE_RATE = 16000  # samples per second
CHUNK_SECONDS = 0.1  # how often the mic level is checked
SILENCE_THRESHOLD = 0.01  # RMS level treated as speech vs. silence
WAIT_FOR_SPEECH_SECONDS = 5  # give up if the user says nothing
SILENCE_SECONDS = 1.2  # stop after this much trailing silence
MAX_SECONDS = 10  # hard cap on one recording


def record_question() -> sr.AudioData | None:
    """Record one spoken question with sounddevice and return it as AudioData.

    Recording starts immediately, stops when the speaker goes quiet, and
    returns None when no speech was heard.
    """
    chunks: list[np.ndarray] = []
    state = {"heard_speech": False, "silent_chunks": 0}

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())
        rms = math.sqrt(sum(float(x) ** 2 for x in indata[:, 0]) / max(frames, 1))
        if rms >= SILENCE_THRESHOLD:
            state["heard_speech"] = True
            state["silent_chunks"] = 0
        else:
            state["silent_chunks"] += 1

    start = time.monotonic()
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
        callback=callback,
    ):
        while True:
            time.sleep(CHUNK_SECONDS)
            elapsed = time.monotonic() - start
            if not state["heard_speech"] and elapsed >= WAIT_FOR_SPEECH_SECONDS:
                return None
            if (
                state["heard_speech"]
                and state["silent_chunks"] * CHUNK_SECONDS >= SILENCE_SECONDS
            ):
                break
            if elapsed >= MAX_SECONDS:
                break

    if not state["heard_speech"] or not chunks:
        return None

    # Convert float32 samples to 16-bit PCM for SpeechRecognition.
    audio = np.concatenate(chunks, axis=0)
    pcm = (np.clip(audio[:, 0], -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return sr.AudioData(pcm, SAMPLE_RATE, 2)


def transcribe(audio: sr.AudioData) -> str | None:
    """Convert recorded audio to text with Google speech recognition."""
    recognizer = sr.Recognizer()
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as error:
        print(f"{ASSISTANT_NAME}: Speech recognition service is unavailable: {error}\n")
        return None


def main() -> None:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        sys.exit("Set your Gemini API key in config.py before running.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    print(f"Hello! I'm {ASSISTANT_NAME}, your Gemini-powered assistant.")
    print("Ask me anything. Say 'quit' to exit.\n")

    while True:
        print("Listening...")
        audio = record_question()
        if audio is None:
            print("I didn't hear anything. Try again.\n")
            continue

        question = transcribe(audio)
        if not question:
            print("Sorry, I couldn't understand that. Try again.\n")
            continue

        print(f"You: {question}")
        if question.lower().strip() == "quit":
            print(f"{ASSISTANT_NAME}: Goodbye!")
            break

        try:
            response = model.generate_content(question)
            print(f"{ASSISTANT_NAME}: {response.text.strip()}\n")
        except Exception as error:  # noqa: BLE001 - keep the loop alive
            print(f"{ASSISTANT_NAME}: Sorry, I ran into a problem: {error}\n")


if __name__ == "__main__":
    main()
