"""Nova — a desktop voice assistant built with tkinter.

Layout:
    - Chat window with the conversation history and a [Copy] link per message.
    - Mic button to ask by voice (sounddevice + SpeechRecognition).
    - Passive wake-word listener: say the assistant's name ("Nova") to activate.
    - Text input box + Send button as fallback, Stop button to cut speech.
    - Status line: "Listening...", "Thinking...", "Speaking...".
    - Typing indicator dots while Gemini is thinking.
    - Settings panel: Gemini API key, assistant name, voice speed, Allowed Apps.
    - Closing the window minimizes to the system tray (pystray).

The AI brain is Gemini (gemini-3.5-flash-lite). Voice output uses gTTS.
Speech is played through sounddevice (so speed control and Stop work) with
playsound as a fallback. Saying "search for ..." triggers a DuckDuckGo web
search; saying "open [app]" launches an app from the Allowed Apps list.
"""

import math
import os
import re
import subprocess
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

try:
    import miniaudio  # mp3 -> PCM decoding (speed control + interruptible playback)
except ImportError:  # pragma: no cover - optional dependency
    miniaudio = None

try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TRAY_AVAILABLE = False

import config

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ASSISTANT_NAME = "Nova"
PLACEHOLDER_KEY = "YOUR_GEMINI_API_KEY_HERE"
NO_APP_REPLY = "That app isn't on my allowed list."
SPEEDS = {0: "slow", 1: "normal", 2: "fast"}
FAST_RATE = 1.35  # playback rate for the "fast" voice speed

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
# Voice I/O, web search, commands, and config helpers
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


def _play_mp3(path: str, rate: float, cancel: threading.Event | None) -> None:
    """Play an mp3 at the given rate, interruptible via `cancel`/sd.stop()."""
    if miniaudio is not None and abs(rate - 1.0) > 0.01:
        try:
            decoded = miniaudio.decode_file(path)
            samples = np.array(decoded.samples, dtype=np.int16)
            audio = samples.astype(np.float32) / 32768.0
            if decoded.nchannels > 1:
                audio = audio.reshape(-1, decoded.nchannels)
            sd.play(audio, samplerate=int(decoded.sample_rate * rate))
            sd.wait()  # returns early when sd.stop() is called
            return
        except Exception:  # noqa: BLE001 - fall back to plain playback
            pass
    if miniaudio is not None:
        try:
            decoded = miniaudio.decode_file(path)
            samples = np.array(decoded.samples, dtype=np.int16)
            audio = samples.astype(np.float32) / 32768.0
            if decoded.nchannels > 1:
                audio = audio.reshape(-1, decoded.nchannels)
            sd.play(audio, samplerate=decoded.sample_rate)
            sd.wait()
            return
        except Exception:  # noqa: BLE001 - fall back to playsound
            pass
    playsound(path)  # fallback: cannot change speed or stop mid-sentence


def speak(text: str, speed: str = "normal",
          cancel: threading.Event | None = None) -> None:
    """Speak text out loud: gTTS -> temp mp3 -> playback -> delete the temp file.

    "slow" uses gTTS slow mode; "fast" plays the audio at a faster rate.
    """
    if not text or (cancel is not None and cancel.is_set()):
        return
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        mp3_path = tmp_file.name
    try:
        gTTS(text, slow=(speed == "slow")).save(mp3_path)
        rate = FAST_RATE if speed == "fast" else 1.0
        _play_mp3(mp3_path, rate, cancel)
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


def extract_open_app(text: str) -> str | None:
    """Return the app name when the message asks to "open [app name]"."""
    match = re.match(
        r"^\s*(?:please\s+)?(?:can\s+you\s+)?open\s+(.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" .:!?")
    return None


def split_wake_word(text: str, wake_name: str) -> str | None:
    """Split an utterance on the wake word.

    Returns None when the wake word is absent, "" when the utterance is only
    the wake word, or the question that follows the wake word.
    """
    if not wake_name:
        return None
    match = re.search(rf"\b{re.escape(wake_name)}\b(.*)$", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip(" .:!?")


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


def save_config(api_key: str, assistant_name: str,
                allowed_apps: dict | None = None) -> None:
    """Persist the settings back to config.py."""
    apps = allowed_apps if allowed_apps is not None else {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    content = (
        '"""Configuration settings for the voice assistant."""\n\n'
        "# Get your Gemini API key from Google AI Studio: "
        "https://aistudio.google.com/apikey\n"
        "# NOTE: Do not commit a real API key to a public repository.\n"
        f"GEMINI_API_KEY = {api_key!r}\n\n"
        "# Name the assistant introduces itself with (editable in the app's Settings).\n"
        f"ASSISTANT_NAME = {assistant_name!r}\n\n"
        "# Apps the assistant may open with 'open [app name]'.\n"
        f"ALLOWED_APPS = {apps!r}\n"
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
        self.allowed_apps: dict[str, str] = dict(
            getattr(config, "ALLOWED_APPS", {}) or {}
        )
        self.model = None
        self.chat = None
        self.tray_icon = None

        self.speech_lock = threading.Lock()
        self._audio_lock = threading.Lock()  # one mic stream at a time
        self._busy_event = threading.Event()  # set while listening/thinking
        self._speaking_event = threading.Event()  # set while TTS plays
        self._speech_cancel = threading.Event()  # set to stop playback
        self._running = True
        self._speed = "normal"
        self._msg_counter = 0
        self._copy_texts: dict[str, str] = {}
        self._typing_active = False
        self._typing_start = "1.0"
        self._typing_ticks = 0
        self._typing_job = None

        root.title(f"{self.assistant_name} — Voice Assistant")
        root.geometry("1080x740")
        root.minsize(860, 560)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self._hide_to_tray)

        self._build_ui()
        self._configure_gemini()
        self._greet()
        self._setup_tray()
        threading.Thread(target=self._wake_word_worker, daemon=True).start()

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
        self.clear_button = tk.Button(
            header, text="Clear chat", font=FONT_BOLD, command=self._clear_chat,
            bg=PANEL, fg=SOFT, activebackground=ENTRY_BG, activeforeground=TEXT,
            relief=tk.FLAT, padx=12, pady=6, cursor="hand2", borderwidth=0,
        )
        self.clear_button.pack(side=tk.RIGHT, pady=(4, 0))

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
        self.chat_view.tag_configure("typing_line", foreground=MUTED,
                                     font=("Segoe UI", 10, "italic"))
        self.chat_view.tag_configure("copy_link", foreground=ACCENT,
                                     font=("Segoe UI", 8, "underline"))

        # Status line
        bottom = tk.Frame(main, bg=BG)
        bottom.pack(fill=tk.X, padx=16, pady=(2, 14))
        self.status_label = tk.Label(
            bottom, text="Ready", font=("Segoe UI", 9), bg=BG, fg=MUTED, anchor="w",
        )
        self.status_label.pack(fill=tk.X, pady=(0, 6))

        # Input row: mic, stop, text input box (fallback), send
        input_row = tk.Frame(bottom, bg=BG)
        input_row.pack(fill=tk.X)
        self.mic_button = tk.Button(
            input_row, text="Mic", font=FONT_BOLD, command=self.on_mic,
            bg=ACCENT, fg=WHITE, activebackground=ACCENT_HOVER,
            activeforeground=WHITE, relief=tk.FLAT, padx=18, pady=8,
            cursor="hand2", borderwidth=0,
        )
        self.mic_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_button = tk.Button(
            input_row, text="Stop", font=FONT_BOLD, command=self._stop_speaking,
            bg=PANEL, fg=RED, activebackground=ENTRY_BG, activeforeground=RED,
            relief=tk.FLAT, padx=14, pady=8, cursor="hand2", borderwidth=0,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
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

        # Settings panel (scrollable)
        settings = tk.Frame(self.root, bg=PANEL, width=300)
        settings.pack(side=tk.RIGHT, fill=tk.Y)
        settings.pack_propagate(False)
        canvas = tk.Canvas(settings, bg=PANEL, highlightthickness=0, borderwidth=0)
        settings_scroll = tk.Scrollbar(settings, command=canvas.yview, width=10,
                                       relief=tk.FLAT, bg=ENTRY_BORDER,
                                       activebackground=MUTED, troughcolor=PANEL)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=settings_scroll.set)
        settings_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window_id, width=e.width),
        )

        def _label(text: str) -> None:
            tk.Label(inner, text=text, font=FONT_SMALL,
                     bg=PANEL, fg=MUTED).pack(anchor="w", padx=18, pady=(10, 0))

        def _entry(show: str = "") -> tk.Entry:
            widget = tk.Entry(
                inner, show=show, bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                font=FONT, relief=tk.FLAT, highlightthickness=1,
                highlightbackground=ENTRY_BORDER, highlightcolor=ACCENT,
            )
            widget.pack(fill=tk.X, padx=18, pady=(4, 0), ipady=6)
            return widget

        tk.Label(inner, text="Settings", font=FONT_SECTION,
                 bg=PANEL, fg=TEXT).pack(anchor="w", padx=18, pady=(20, 4))
        _label("ASSISTANT NAME")
        self.name_entry = _entry()
        self.name_entry.insert(0, self.assistant_name)
        _label("GEMINI API KEY")
        self.key_entry = _entry(show="•")
        if self.api_key and self.api_key != PLACEHOLDER_KEY:
            self.key_entry.insert(0, self.api_key)
        tk.Label(inner, text="Stored in config.py", font=("Segoe UI", 8),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=18)

        # Voice speed slider (slow / normal / fast)
        _label("VOICE SPEED")
        self.speed_scale = tk.Scale(
            inner, from_=0, to=2, resolution=1, orient=tk.HORIZONTAL,
            showvalue=False, bg=PANEL, fg=TEXT, highlightthickness=0,
            troughcolor=ENTRY_BG, activebackground=ACCENT, sliderrelief=tk.FLAT,
            command=self._on_speed_change,
        )
        self.speed_scale.set(1)
        self.speed_scale.pack(fill=tk.X, padx=14, pady=(2, 0))
        speed_labels = tk.Frame(inner, bg=PANEL)
        speed_labels.pack(fill=tk.X, padx=18)
        for text, side in (("Slow", tk.LEFT), ("Normal", tk.CENTER), ("Fast", tk.RIGHT)):
            tk.Label(speed_labels, text=text, font=("Segoe UI", 8),
                     bg=PANEL, fg=MUTED).pack(side=side, expand=True)

        # Allowed Apps
        _label("ALLOWED APPS")
        self.apps_list = tk.Listbox(
            inner, bg=ENTRY_BG, fg=TEXT, font=("Segoe UI", 9), relief=tk.FLAT,
            highlightthickness=1, highlightbackground=ENTRY_BORDER,
            selectbackground=ACCENT, selectforeground=WHITE, height=5,
            activestyle="none", borderwidth=0,
        )
        self.apps_list.pack(fill=tk.X, padx=18, pady=(4, 0))
        self.apps_list.bind("<<ListboxSelect>>", self._on_app_select)
        self._refresh_apps_list()
        self.app_name_entry = _entry()
        self.app_name_entry.insert(0, "app name")
        self.app_path_entry = _entry()
        self.app_path_entry.insert(0, "file path")
        apps_buttons = tk.Frame(inner, bg=PANEL)
        apps_buttons.pack(fill=tk.X, padx=18, pady=(8, 0))
        self.add_app_button = tk.Button(
            apps_buttons, text="Add", font=FONT_BOLD, command=self._add_app,
            bg=ACCENT, fg=WHITE, activebackground=ACCENT_HOVER,
            activeforeground=WHITE, relief=tk.FLAT, pady=6, cursor="hand2",
            borderwidth=0,
        )
        self.add_app_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.remove_app_button = tk.Button(
            apps_buttons, text="Remove", font=FONT_BOLD, command=self._remove_app,
            bg=PANEL, fg=RED, activebackground=ENTRY_BG, activeforeground=RED,
            relief=tk.FLAT, pady=6, cursor="hand2", borderwidth=0,
        )
        self.remove_app_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        save_button = tk.Button(
            inner, text="Save settings", font=FONT_BOLD, command=self.save_settings,
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
        self._set_status(f'Ready — say "{self.assistant_name}" to talk')

    # ----- Thread-safe UI updates -------------------------------------------

    def _set_status(self, text: str, color: str = MUTED) -> None:
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    def _append_message(self, role: str, text: str) -> None:
        def insert() -> None:
            self._hide_typing_now()
            names = {"user": "You", "assistant": self.assistant_name, "system": "Notice"}
            self._msg_counter += 1
            copy_tag = f"copy_{self._msg_counter}"
            self._copy_texts[copy_tag] = text
            self.chat_view.configure(state=tk.NORMAL)
            self.chat_view.insert(tk.END, f"{names[role]}", f"{role}_name")
            self.chat_view.insert(tk.END, "   [Copy]\n", ("copy_link", copy_tag))
            self.chat_view.insert(tk.END, f"{text}\n\n", f"{role}_text")
            self.chat_view.configure(state=tk.DISABLED)
            self.chat_view.tag_bind(
                copy_tag, "<Button-1>",
                lambda _event, t=copy_tag: self._copy_message(t),
            )
            self.chat_view.tag_bind(
                copy_tag, "<Enter>",
                lambda _event: self.chat_view.config(cursor="hand2"),
            )
            self.chat_view.tag_bind(
                copy_tag, "<Leave>",
                lambda _event: self.chat_view.config(cursor="arrow"),
            )
            self.chat_view.see(tk.END)

        self.root.after(0, insert)

    def _copy_message(self, tag: str) -> None:
        text = self._copy_texts.get(tag, "")
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Copied to clipboard.", GREEN)

    def _set_busy(self, busy: bool) -> None:
        def apply() -> None:
            state = tk.DISABLED if busy else tk.NORMAL
            self.mic_button.config(state=state)
            self.send_button.config(state=state)

        self.root.after(0, apply)

    # ----- Typing indicator ---------------------------------------------------

    def _show_typing(self) -> None:
        def start() -> None:
            self._hide_typing_now()
            self.chat_view.configure(state=tk.NORMAL)
            self._typing_start = self.chat_view.index(tk.END)
            self.chat_view.insert(tk.END, f"{self.assistant_name} is typing",
                                  "typing_line")
            self.chat_view.configure(state=tk.DISABLED)
            self.chat_view.see(tk.END)
            self._typing_active = True
            self._typing_ticks = 0
            self._typing_job = self.root.after(350, self._animate_typing)

        self.root.after(0, start)

    def _animate_typing(self) -> None:
        if not self._typing_active:
            return
        self._typing_ticks = (self._typing_ticks + 1) % 4
        dots = "." * self._typing_ticks
        self.chat_view.configure(state=tk.NORMAL)
        self.chat_view.delete(self._typing_start, tk.END)
        self.chat_view.insert(self._typing_start,
                              f"{self.assistant_name} is typing{dots}", "typing_line")
        self.chat_view.configure(state=tk.DISABLED)
        self.chat_view.see(tk.END)
        self._typing_job = self.root.after(350, self._animate_typing)

    def _hide_typing(self) -> None:
        self.root.after(0, self._hide_typing_now)

    def _hide_typing_now(self) -> None:
        self._typing_active = False
        if self._typing_job is not None:
            try:
                self.root.after_cancel(self._typing_job)
            except Exception:  # noqa: BLE001 - job may already have run
                pass
            self._typing_job = None
        if self._typing_start != "1.0":
            self.chat_view.configure(state=tk.NORMAL)
            self.chat_view.delete(self._typing_start, tk.END)
            self.chat_view.configure(state=tk.DISABLED)

    # ----- User actions -------------------------------------------------------

    def on_send(self) -> None:
        question = self.entry.get().strip()
        if not question:
            return
        self.entry.delete(0, tk.END)
        self._append_message("user", question)
        threading.Thread(target=self._ask_worker, args=(question,), daemon=True).start()

    def on_mic(self) -> None:
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _on_speed_change(self, value: str) -> None:
        self._speed = SPEEDS.get(int(float(value)), "normal")

    def _stop_speaking(self) -> None:
        self._speech_cancel.set()
        try:
            sd.stop()  # interrupts sounddevice playback mid-sentence
        except Exception:  # noqa: BLE001 - no active stream
            pass
        self._speaking_event.clear()
        self._set_status("Ready")

    def _clear_chat(self) -> None:
        self.chat_view.configure(state=tk.NORMAL)
        self.chat_view.delete("1.0", tk.END)
        self.chat_view.configure(state=tk.DISABLED)
        self._copy_texts.clear()
        self._typing_start = "1.0"
        self.chat = None
        self._configure_gemini()  # fresh chat with empty history
        self._set_status("Chat cleared — memory reset.", GREEN)

    def save_settings(self) -> None:
        self.assistant_name = self.name_entry.get().strip() or DEFAULT_ASSISTANT_NAME
        self.api_key = self.key_entry.get().strip()
        save_config(self.api_key, self.assistant_name, self.allowed_apps)
        self._configure_gemini()
        self.title_label.config(text=f"● {self.assistant_name}")
        self.root.title(f"{self.assistant_name} — Voice Assistant")
        self._set_status("Settings saved.", GREEN)

    # ----- Allowed Apps -------------------------------------------------------

    def _refresh_apps_list(self) -> None:
        self.apps_list.delete(0, tk.END)
        for name in sorted(self.allowed_apps):
            self.apps_list.insert(tk.END, name)

    def _on_app_select(self, _event=None) -> None:
        selection = self.apps_list.curselection()
        if not selection:
            return
        name = self.apps_list.get(selection[0])
        self.app_name_entry.delete(0, tk.END)
        self.app_name_entry.insert(0, name)
        self.app_path_entry.delete(0, tk.END)
        self.app_path_entry.insert(0, self.allowed_apps.get(name, ""))

    def _add_app(self) -> None:
        name = self.app_name_entry.get().strip()
        path = self.app_path_entry.get().strip()
        if not name or not path or name == "app name" or path == "file path":
            self._set_status("Enter both an app name and a file path.", AMBER)
            return
        self.allowed_apps[name] = path
        save_config(self.api_key, self.assistant_name, self.allowed_apps)
        self._refresh_apps_list()
        self._set_status(f"Added '{name}' to Allowed Apps.", GREEN)

    def _remove_app(self) -> None:
        selection = self.apps_list.curselection()
        if not selection:
            self._set_status("Select an app to remove.", AMBER)
            return
        name = self.apps_list.get(selection[0])
        self.allowed_apps.pop(name, None)
        save_config(self.api_key, self.assistant_name, self.allowed_apps)
        self._refresh_apps_list()
        self._set_status(f"Removed '{name}' from Allowed Apps.", GREEN)

    def _open_app(self, app_name: str) -> str:
        for name, path in self.allowed_apps.items():
            if name.lower() == app_name.lower():
                try:
                    subprocess.Popen([path])
                    return f"Opening {name}."
                except Exception as error:  # noqa: BLE001 - report and continue
                    return f"I couldn't open {name}: {error}"
        return NO_APP_REPLY

    # ----- System tray ---------------------------------------------------------

    def _setup_tray(self) -> None:
        if not TRAY_AVAILABLE:
            return
        image = Image.new("RGBA", (64, 64), (15, 18, 32, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((6, 6, 58, 58), fill=(108, 140, 255, 255))
        draw.ellipse((20, 20, 44, 44), fill=(22, 26, 44, 255))
        draw.ellipse((28, 28, 36, 36), fill=(99, 214, 143, 255))
        menu = pystray.Menu(
            pystray.MenuItem("Open", self._show_window, default=True),
            pystray.MenuItem("Quit", self._quit_app),
        )
        self.tray_icon = pystray.Icon(
            "voice_assistant", image, f"{self.assistant_name} — Voice Assistant", menu,
        )
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def _hide_to_tray(self) -> None:
        self.root.withdraw()
        self._set_status("Minimized to tray.", MUTED)

    def _show_window(self, *_args) -> None:
        self.root.after(0, self._restore_window)

    def _restore_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _quit_app(self, *_args) -> None:
        self._running = False
        self._speech_cancel.set()
        try:
            sd.stop()
        except Exception:  # noqa: BLE001 - no active stream
            pass
        if self.tray_icon is not None:
            self.tray_icon.stop()
        self.root.after(0, self.root.destroy)

    # ----- Background workers ---------------------------------------------------

    def _greet(self) -> None:
        greeting = (
            f"Hello! I'm {self.assistant_name}, your voice assistant. "
            f"Say '{self.assistant_name}' to talk, ask me anything, "
            "or say 'search for' to look something up."
        )
        self._append_message("assistant", greeting)
        threading.Thread(target=self._speak, args=(greeting,), daemon=True).start()

    def _speak(self, text: str) -> None:
        """Speak a line at the selected speed; cancellable via the Stop button."""
        cancel = threading.Event()
        self._speech_cancel = cancel
        self._speaking_event.set()
        self._set_status("Speaking...", GREEN)
        with self.speech_lock:
            speak(text, self._speed, cancel)
        self._speaking_event.clear()
        self._set_status("Ready")

    def _wake_word_worker(self) -> None:
        """Passively listen for the assistant's name and activate on a match."""
        while self._running:
            if self._busy_event.is_set() or self._speaking_event.is_set():
                time.sleep(0.2)
                continue
            with self._audio_lock:
                try:
                    audio = record_question()
                except Exception:  # noqa: BLE001 - mic trouble; retry
                    time.sleep(1.0)
                    continue
            if audio is None or self._busy_event.is_set() or self._speaking_event.is_set():
                continue
            text = transcribe(audio)
            if not text:
                continue
            question = split_wake_word(text, self.assistant_name)
            if question is None:
                continue
            self.root.after(0, self._restore_window)
            if question:
                self._append_message("user", question)
                self._ask_worker(question)
            else:
                self._listen_worker()  # just the wake word -> hear the question

    def _listen_worker(self) -> None:
        self._busy_event.set()
        self._set_busy(True)
        self._set_status("Listening...", AMBER)
        with self._audio_lock:
            try:
                audio = record_question()
            except Exception as error:  # noqa: BLE001 - mic problems fall back to typing
                self._append_message("system", f"Microphone failed: {error}. "
                                               "Type your message instead.")
                self._set_status("Mic failed — type instead", RED)
                self._busy_event.clear()
                self._set_busy(False)
                return
        if audio is None:
            self._append_message("system", "I didn't hear anything. Try again.")
            self._set_status("Ready")
            self._busy_event.clear()
            self._set_busy(False)
            return
        question = transcribe(audio)
        if not question:
            self._append_message("system", "Sorry, I couldn't understand that.")
            self._set_status("Ready")
            self._busy_event.clear()
            self._set_busy(False)
            return
        self._append_message("user", question)
        self._ask_worker(question)

    def _ask_worker(self, question: str) -> None:
        self._busy_event.set()
        self._set_busy(True)

        # "open [app name]" -> launch from the Allowed Apps list.
        app_name = extract_open_app(question)
        if app_name is not None:
            answer = self._open_app(app_name)
            self._append_message("assistant", answer)
            self._speak(answer)
            self._busy_event.clear()
            self._set_busy(False)
            return

        # "search for ..." -> DuckDuckGo results as Gemini context.
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

        self._show_typing()  # animated dots while Gemini thinks
        self._set_status("Thinking...", ACCENT)
        if self.chat is None:
            answer = "Set your Gemini API key in the Settings panel to start chatting."
        else:
            try:
                response = self.chat.send_message(prompt)
                answer = (response.text or "").strip() or "(no response)"
            except Exception as error:  # noqa: BLE001 - keep the app alive
                answer = f"Sorry, I ran into a problem: {error}"
        self._hide_typing()
        self._append_message("assistant", answer)
        self._speak(answer)
        self._busy_event.clear()
        self._set_busy(False)


def main() -> None:
    root = tk.Tk()
    VoiceAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
