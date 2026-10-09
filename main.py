"""Nova — a desktop voice assistant built with tkinter.

Layout:
    - Chat window with the conversation history.
    - Mic button to ask by voice (sounddevice + SpeechRecognition).
    - Text input box + Send button as fallback.
    - Status line: "Listening...", "Thinking...", "Speaking...".
    - Settings panel: Gemini API key and assistant name (saved to config.py).

The AI brain is Gemini (gemini-3.5-flash-lite). Voice output uses gTTS
(temp mp3 played with playsound, then deleted). Saying "search for ..."
triggers a DuckDuckGo web search whose results are given to Gemini.
"""

import math
import os
import re
import tempfile
import threading
import time

import numpy as np
import requests
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai
import tkinter as tk

import config

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ASSISTANT_NAME = "Nova"
PLACEHOLDER_KEY = "YOUR_GEMINI_API_KEY_HERE"

# Microphone recording settings
SAMPLE_RATE = 16000  # samples per second
CHUNK_SECONDS = 0.1  # how often the mic level is checked
SILENCE_THRESHOLD = 0.01  # RMS level treated as speech vs. silence
WAIT_FOR_SPEECH_SECONDS = 5  # give up if the user says nothing
SILENCE_SECONDS = 0.8  # stop recording 0.8 s after you finish speaking
MAX_SECONDS = 10  # hard cap on one recording

# Dark theme colors
BG = "#0f1220"  # window background
PANEL = "#161a2c"  # settings panel
CARD = "#1b2036"  # chat surface
ENTRY_BG = "#242a42"
ENTRY_BORDER = "#333b5c"
TEXT = "#e9ecf7"
SOFT = "#b9c0d8"
MUTED = "#8189a6"
ACCENT = "#6c8cff"
ACCENT_HOVER = "#87a1ff"
WHITE = "#ffffff"
GREEN = "#63d68f"
AMBER = "#f0b35c"
RED = "#f07373"

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 8, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SECTION = ("Segoe UI", 12, "bold")


# ----------------------------------------------------------------------------
# Voice I/O, web search, and config helpers
# ----------------------------------------------------------------------------

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
    except sr.RequestError:
        return None


def speak(text: str) -> None:
    """Speak text out loud: gTTS -> temp mp3 -> playsound -> delete the temp file."""
    if not text:
        return
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        mp3_path = tmp_file.name
    try:
        gTTS(text).save(mp3_path)  # convert the text to speech (writes the mp3)
        playsound(mp3_path)  # play the mp3 out loud
    except Exception:  # noqa: BLE001 - speech output is optional
        pass
    finally:
        try:
            os.remove(mp3_path)  # delete the temp file after playing
        except OSError:
            pass


def extract_search_query(text: str) -> str | None:
    """Return the query when the message asks to "search for ..."."""
    parts = re.split(r"\bsearch for\b", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip(" .:!?")
    return None


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return a plain-text summary of the results.

    Uses the Instant Answer API first, then falls back to the HTML results
    page when the API has nothing for this query.
    """
    headers = {"User-Agent": "Mozilla/5.0 (voice-assistant)"}
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers=headers,
            timeout=10,
        )
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        return f"(web search failed: {error})"

    parts: list[str] = []
    if data.get("Answer"):
        parts.append(str(data["Answer"]))
    if data.get("AbstractText"):
        parts.append(data["AbstractText"])
    if data.get("AbstractURL"):
        parts.append(f"Source: {data['AbstractURL']}")
    for topic in data.get("RelatedTopics", []):
        if not isinstance(topic, dict):
            continue
        if topic.get("Text"):
            parts.append(f"- {topic['Text']}")
        for sub in topic.get("Topics", []):
            if isinstance(sub, dict) and sub.get("Text"):
                parts.append(f"- {sub['Text']}")
        if len(parts) >= max_results + 2:
            break
    if parts:
        return "\n".join(parts[: max_results + 2])

    # Fallback: parse the HTML results page.
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=10,
        )
        page = response.text
    except requests.RequestException as error:
        return f"(web search failed: {error})"

    results: list[str] = []
    pattern = (
        r'class="result__a"[^>]*>(?P<title>.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(?P<snippet>.*?)</(?:a|div)>'
    )
    for match in re.finditer(pattern, page, flags=re.DOTALL | re.IGNORECASE):
        title = re.sub(r"<[^>]+>", "", match.group("title")).strip()
        snippet = re.sub(r"<[^>]+>", "", match.group("snippet")).strip()
        if title:
            results.append(f"- {title}: {snippet}")
        if len(results) >= max_results:
            break
    return "\n".join(results) if results else "(no results found on DuckDuckGo)"


def save_config(api_key: str, assistant_name: str) -> None:
    """Persist the settings back to config.py."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    content = (
        '"""Configuration settings for the voice assistant."""\n\n'
        "# Get your Gemini API key from Google AI Studio: "
        "https://aistudio.google.com/apikey\n"
        "# NOTE: Do not commit a real API key to a public repository.\n"
        f"GEMINI_API_KEY = {api_key!r}\n\n"
        "# Name the assistant introduces itself with (editable in the app's Settings).\n"
        f"ASSISTANT_NAME = {assistant_name!r}\n"
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


# ----------------------------------------------------------------------------
# The desktop app
# ----------------------------------------------------------------------------

class VoiceAssistantApp:
    """tkinter desktop app for the voice assistant."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.api_key = getattr(config, "GEMINI_API_KEY", "")
        self.assistant_name = getattr(config, "ASSISTANT_NAME", DEFAULT_ASSISTANT_NAME)
        self.model = None
        self.chat = None
        self.speech_lock = threading.Lock()

        root.title(f"{self.assistant_name} — Voice Assistant")
        root.geometry("1000x660")
        root.minsize(780, 520)
        root.configure(bg=BG)

        self._build_ui()
        self._configure_gemini()
        self._greet()

    # ----- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(main, bg=BG)
        header.pack(fill=tk.X, padx=20, pady=(16, 6))
        self.title_label = tk.Label(
            header, text=f"● {self.assistant_name}",
            font=FONT_TITLE, bg=BG, fg=TEXT,
        )
        self.title_label.pack(side=tk.LEFT)
        tk.Label(
            header, text="  voice assistant", font=FONT, bg=BG, fg=MUTED,
        ).pack(side=tk.LEFT, padx=(2, 0), pady=(6, 0))

        # Chat window (conversation history)
        chat_frame = tk.Frame(main, bg=CARD, highlightthickness=1,
                              highlightbackground=ENTRY_BORDER)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        self.chat_view = tk.Text(
            chat_frame, bg=CARD, fg=TEXT, font=FONT, relief=tk.FLAT,
            wrap=tk.WORD, state=tk.DISABLED, padx=16, pady=14, spacing3=4,
            cursor="arrow", borderwidth=0,
        )
        scrollbar = tk.Scrollbar(chat_frame, command=self.chat_view.yview,
                                 width=10, relief=tk.FLAT, bg=ENTRY_BORDER,
                                 activebackground=MUTED, troughcolor=CARD)
        self.chat_view.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_view.tag_configure("user_name", foreground=ACCENT, font=FONT_BOLD)
        self.chat_view.tag_configure("assistant_name", foreground=GREEN, font=FONT_BOLD)
        self.chat_view.tag_configure("system_name", foreground=AMBER, font=FONT_BOLD)
        self.chat_view.tag_configure("user_text", foreground=TEXT, lmargin1=10, lmargin2=10)
        self.chat_view.tag_configure("assistant_text", foreground=SOFT,
                                     lmargin1=10, lmargin2=10)
        self.chat_view.tag_configure("system_text", foreground=MUTED,
                                     lmargin1=10, lmargin2=10)

        # Status line
        bottom = tk.Frame(main, bg=BG)
        bottom.pack(fill=tk.X, padx=16, pady=(2, 14))
        self.status_label = tk.Label(
            bottom, text="Ready", font=("Segoe UI", 9), bg=BG, fg=MUTED, anchor="w",
        )
        self.status_label.pack(fill=tk.X, pady=(0, 6))

        # Input row: mic button, text input box (fallback), send button
        input_row = tk.Frame(bottom, bg=BG)
        input_row.pack(fill=tk.X)
        self.mic_button = tk.Button(
            input_row, text="Mic", font=FONT_BOLD, command=self.on_mic,
            bg=ACCENT, fg=WHITE, activebackground=ACCENT_HOVER,
            activeforeground=WHITE, relief=tk.FLAT, padx=18, pady=8,
            cursor="hand2", borderwidth=0,
        )
        self.mic_button.pack(side=tk.LEFT, padx=(0, 10))
        self.entry = tk.Entry(
            input_row, bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT, font=FONT,
            relief=tk.FLAT, highlightthickness=1, highlightbackground=ENTRY_BORDER,
            highlightcolor=ACCENT,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.entry.bind("<Return>", lambda _event: self.on_send())
        self.send_button = tk.Button(
            input_row, text="Send", font=FONT_BOLD, command=self.on_send,
            bg=ACCENT, fg=WHITE, activebackground=ACCENT_HOVER,
            activeforeground=WHITE, relief=tk.FLAT, padx=18, pady=8,
            cursor="hand2", borderwidth=0,
        )
        self.send_button.pack(side=tk.LEFT, padx=(10, 0))

        # Settings panel
        settings = tk.Frame(self.root, bg=PANEL, width=260)
        settings.pack(side=tk.RIGHT, fill=tk.Y)
        settings.pack_propagate(False)
        tk.Label(settings, text="Settings", font=FONT_SECTION,
                 bg=PANEL, fg=TEXT).pack(anchor="w", padx=18, pady=(20, 14))
        tk.Label(settings, text="ASSISTANT NAME", font=FONT_SMALL,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=18)
        self.name_entry = tk.Entry(
            settings, bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT, font=FONT,
            relief=tk.FLAT, highlightthickness=1, highlightbackground=ENTRY_BORDER,
            highlightcolor=ACCENT,
        )
        self.name_entry.pack(fill=tk.X, padx=18, pady=(4, 12), ipady=6)
        self.name_entry.insert(0, self.assistant_name)
        tk.Label(settings, text="GEMINI API KEY", font=FONT_SMALL,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=18)
        self.key_entry = tk.Entry(
            settings, show="•", bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
            font=FONT, relief=tk.FLAT, highlightthickness=1,
            highlightbackground=ENTRY_BORDER, highlightcolor=ACCENT,
        )
        self.key_entry.pack(fill=tk.X, padx=18, pady=(4, 6), ipady=6)
        if self.api_key and self.api_key != PLACEHOLDER_KEY:
            self.key_entry.insert(0, self.api_key)
        tk.Label(settings, text="Stored in config.py", font=("Segoe UI", 8),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=18)
        save_button = tk.Button(
            settings, text="Save settings", font=FONT_BOLD, command=self.save_settings,
            bg=ACCENT, fg=WHITE, activebackground=ACCENT_HOVER,
            activeforeground=WHITE, relief=tk.FLAT, pady=8, cursor="hand2",
            borderwidth=0,
        )
        save_button.pack(fill=tk.X, padx=18, pady=18)

    # ----- Gemini -----------------------------------------------------------

    def _configure_gemini(self) -> None:
        if not self.api_key or self.api_key == PLACEHOLDER_KEY:
            self.model = None
            self.chat = None
            self._set_status("Set your Gemini API key in Settings to start chatting.", AMBER)
            return
        genai.configure(api_key=self.api_key)
        history = list(self.chat.history) if self.chat is not None else []
        self.model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=(
                f"You are {self.assistant_name}, a friendly voice assistant. "
                "Keep answers clear and conversational."
            ),
        )
        self.chat = self.model.start_chat(history=history)
        self._set_status("Ready")

    # ----- Thread-safe UI updates -------------------------------------------

    def _set_status(self, text: str, color: str = MUTED) -> None:
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    def _append_message(self, role: str, text: str) -> None:
        def insert() -> None:
            names = {"user": "You", "assistant": self.assistant_name, "system": "Notice"}
            self.chat_view.configure(state=tk.NORMAL)
            self.chat_view.insert(tk.END, f"{names[role]}\n", f"{role}_name")
            self.chat_view.insert(tk.END, f"{text}\n\n", f"{role}_text")
            self.chat_view.configure(state=tk.DISABLED)
            self.chat_view.see(tk.END)

        self.root.after(0, insert)

    def _set_busy(self, busy: bool) -> None:
        def apply() -> None:
            state = tk.DISABLED if busy else tk.NORMAL
            self.mic_button.config(state=state)
            self.send_button.config(state=state)

        self.root.after(0, apply)

    # ----- User actions -----------------------------------------------------

    def on_send(self) -> None:
        question = self.entry.get().strip()
        if not question:
            return
        self.entry.delete(0, tk.END)
        self._append_message("user", question)
        threading.Thread(target=self._ask_worker, args=(question,), daemon=True).start()

    def on_mic(self) -> None:
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def save_settings(self) -> None:
        self.assistant_name = self.name_entry.get().strip() or DEFAULT_ASSISTANT_NAME
        self.api_key = self.key_entry.get().strip()
        save_config(self.api_key, self.assistant_name)
        self._configure_gemini()
        self.title_label.config(text=f"● {self.assistant_name}")
        self.root.title(f"{self.assistant_name} — Voice Assistant")
        self._set_status("Settings saved.", GREEN)

    # ----- Background workers -----------------------------------------------

    def _greet(self) -> None:
        greeting = (
            f"Hello! I'm {self.assistant_name}, your voice assistant. "
            "Ask me anything, or say 'search for' to look something up."
        )
        self._append_message("assistant", greeting)

        def run() -> None:
            self._set_status("Speaking...", GREEN)
            with self.speech_lock:
                speak(greeting)
            self._set_status("Ready")

        threading.Thread(target=run, daemon=True).start()

    def _listen_worker(self) -> None:
        self._set_busy(True)
        self._set_status("Listening...", AMBER)
        try:
            audio = record_question()
        except Exception as error:  # noqa: BLE001 - mic problems fall back to typing
            self._append_message("system", f"Microphone failed: {error}. "
                                           "Type your message instead.")
            self._set_status("Mic failed — type instead", RED)
            self._set_busy(False)
            return
        if audio is None:
            self._append_message("system", "I didn't hear anything. Try again.")
            self._set_status("Ready")
            self._set_busy(False)
            return
        question = transcribe(audio)
        if not question:
            self._append_message("system", "Sorry, I couldn't understand that.")
            self._set_status("Ready")
            self._set_busy(False)
            return
        self._append_message("user", question)
        self._ask_worker(question)

    def _ask_worker(self, question: str) -> None:
        self._set_busy(True)
        prompt = question
        query = extract_search_query(question)
        if query:
            self._set_status("Searching...", AMBER)
            results = duckduckgo_search(query)
            prompt = (
                f"The user asked me to search the web for: {query}\n"
                f"DuckDuckGo results:\n{results}\n\n"
                "Based on those results, reply to the user's message below. "
                "Be concise and mention sources when useful.\n\n"
                f"User message: {question}"
            )
        self._set_status("Thinking...", ACCENT)
        if self.chat is None:
            answer = "Set your Gemini API key in the Settings panel to start chatting."
        else:
            try:
                response = self.chat.send_message(prompt)
                answer = (response.text or "").strip() or "(no response)"
            except Exception as error:  # noqa: BLE001 - keep the app alive
                answer = f"Sorry, I ran into a problem: {error}"
        self._append_message("assistant", answer)
        self._set_status("Speaking...", GREEN)
        with self.speech_lock:
            speak(answer)
        self._set_status("Ready")
        self._set_busy(False)


def main() -> None:
    root = tk.Tk()
    VoiceAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
