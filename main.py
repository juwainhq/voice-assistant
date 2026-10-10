"""Nova — a desktop voice assistant built with tkinter.

Layout:
    - Full-width dark window (900x620): header bar, chat, input row.
    - Header: avatar + assistant name on the left; a status dot and a gear
      button (settings popup) on the right.
    - Chat window with the conversation history and a [Copy] link per message.
    - Input row: text entry with Mic + Send buttons on the right. While the
      assistant speaks the Mic button turns red — click it to stop.
    - Status line below the input: "Listening...", "Thinking...", "Speaking...".
    - Typing indicator dots while the selected AI is thinking.
    - Settings popup (gear button): Gemini API or Local AI (Ollama), assistant
      name, voice speed, Allowed Apps.
    - Closing the window minimizes to the system tray (pystray).

The AI brain can be Gemini (gemini-3.5-flash-lite) or local Ollama. Voice
output uses gTTS with a pyttsx3 offline fallback when the network is unavailable.
Speech is played
through sounddevice (so speed control and Stop work) with playsound as a
fallback. Saying "search for ..." triggers a DuckDuckGo web search; saying
"open [app]" launches an app from the Allowed Apps list.
"""

import collections
import json
import math
import os
import queue
import random
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
from google import genai
from google.genai import types
import tkinter as tk

try:
    import ollama
except ImportError:  # pragma: no cover - Local AI is an optional mode
    ollama = None

try:
    import miniaudio  # mp3 -> PCM decoding (speed control + interruptible playback)
except ImportError:  # pragma: no cover - optional dependency
    miniaudio = None

try:
    import pyttsx3  # offline TTS fallback when gTTS cannot reach the network
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageTk = None

try:
    import pystray
    TRAY_AVAILABLE = Image is not None
except ImportError:  # pragma: no cover - optional dependency
    TRAY_AVAILABLE = False

import config

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ASSISTANT_NAME = "Nova"
DEFAULT_OLLAMA_MODEL = "phi3:mini"
OLLAMA_HOST = "http://localhost:11434"
AI_PROVIDER_GEMINI = "gemini"
AI_PROVIDER_OLLAMA = "ollama"
PLACEHOLDER_KEY = "YOUR_GEMINI_API_KEY_HERE"
NO_APP_REPLY = "That app isn't on my allowed list."
SPEEDS = {0: "slow", 1: "normal", 2: "fast"}
FAST_RATE = 1.35  # playback rate for the "fast" voice speed
STOP_RED = "#FF4444"  # mic button background while speaking (click to stop)

# Avatar choices are selectable in Settings and saved as config.AVATAR_MODE.
# These legacy values define the default for older config.py files without it.
AVATAR_MODE_PORTRAIT = "portrait"
AVATAR_MODE_GROKBOT = "grokbot"
AVATAR_MODE_IMAGE = "image"
AVATAR_MODES = (AVATAR_MODE_PORTRAIT, AVATAR_MODE_GROKBOT, AVATAR_MODE_IMAGE)
USE_IMAGE_AVATAR = False  # legacy default selector; Settings is the normal control
AVATAR_STYLE = "portrait"  # legacy default selector: "portrait" or "grokbot"
DEFAULT_AVATAR_MODE = (AVATAR_MODE_IMAGE if USE_IMAGE_AVATAR else AVATAR_STYLE)
AVATAR_FILE = "avatar.png"  # reference artwork and optional still avatar

# Wake-word listener (shares the single mic stream with question recording)
WAKE_CHECK_SECONDS = 2.0  # how often the rolling wake buffer is transcribed
WAKE_WINDOW_SECONDS = 3.0  # rolling 3-second wake buffer sent to recognition

# Microphone recording settings
SAMPLE_RATE = 16000  # samples per second
CHUNK_SECONDS = 0.1  # how often the mic level is checked
SILENCE_THRESHOLD = 0.01  # RMS level treated as speech vs. silence
WAIT_FOR_SPEECH_SECONDS = 5  # give up if the user says nothing
SILENCE_SECONDS = 0.8  # stop recording 0.8 s after you finish speaking
MAX_SECONDS = 10  # hard cap on one recording

# Monochrome design system
BG = "#000000"  # window background (pure black)
PANEL = "#0A0A0A"  # settings panel background
CARD = "#0A0A0A"  # chat window background
GRAY = "#141414"  # cards, panels, idle buttons
ENTRY_BG = "#141414"  # input fields
ENTRY_BORDER = "#242424"  # borders
TEXT = "#FAFAFA"  # white text
SOFT = "#FAFAFA"  # message text
MUTED = "#8C8C8C"  # muted text (status, hints)
WHITE = "#FAFAFA"  # highlight: active buttons, focused fields, mic button
BLACK = "#000000"  # text on white buttons

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 8, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SECTION = ("Segoe UI", 12, "bold")


# ----------------------------------------------------------------------------
# Voice I/O, web search, commands, and config helpers
# ----------------------------------------------------------------------------





def transcribe(audio: sr.AudioData) -> str | None:
    """Convert recorded audio to text with Google speech recognition.

    Any failure (including missing audioop on new Python versions) is
    swallowed so the listening threads never die silently.
    """
    recognizer = sr.Recognizer()
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except Exception as error:  # noqa: BLE001 - keep the listener alive
        print(f"[transcribe] speech recognition failed: {error}")
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


_OFFLINE_ENGINE = None  # pyttsx3 fallback engine, created once at startup
_OFFLINE_BASE_RATE = 200  # engine's default speaking rate
_OFFLINE_LOCK = threading.Lock()


def init_offline_speech() -> None:
    """Create the pyttsx3 fallback engine once at startup."""
    global _OFFLINE_ENGINE, _OFFLINE_BASE_RATE
    if pyttsx3 is None or _OFFLINE_ENGINE is not None:
        return
    try:
        _OFFLINE_ENGINE = pyttsx3.init()
        _OFFLINE_BASE_RATE = _OFFLINE_ENGINE.getProperty("rate")
    except Exception:  # noqa: BLE001 - offline speech stays optional
        _OFFLINE_ENGINE = None


def _speak_offline(text: str, speed: str,
                  cancel: threading.Event | None) -> None:
    """Fallback speech with the startup pyttsx3 engine.

    runAndWait() blocks, so it is called in its own thread; this call waits
    for the utterance to finish (or for cancel, which stops the engine
    mid-sentence) so "Speaking..." stays accurate.
    """
    if not text or (cancel is not None and cancel.is_set()):
        return
    if _OFFLINE_ENGINE is None:
        init_offline_speech()  # safety net if startup init was skipped
    engine = _OFFLINE_ENGINE
    if engine is None:
        return

    def _run() -> None:
        with _OFFLINE_LOCK:
            try:
                if speed == "fast":
                    engine.setProperty("rate", _OFFLINE_BASE_RATE * FAST_RATE)
                elif speed == "slow":
                    engine.setProperty("rate", _OFFLINE_BASE_RATE * 0.75)
                else:
                    engine.setProperty("rate", _OFFLINE_BASE_RATE)
                engine.say(text)
                engine.runAndWait()  # blocks - always called in this thread
            except Exception:  # noqa: BLE001 - speech output is optional
                pass

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    while worker.is_alive():
        if cancel is not None and cancel.is_set():
            try:
                engine.stop()  # cut the speech mid-sentence
            except Exception:  # noqa: BLE001 - engine may be busy
                pass
            break
        worker.join(0.1)


def speak(text: str, speed: str = "normal",
          cancel: threading.Event | None = None) -> None:
    """Speak text out loud: gTTS -> temp mp3 -> playback -> delete the temp file.

    "slow" uses gTTS slow mode; "fast" plays the audio at a faster rate.
    gTTS needs an internet request that can hang, so every gTTS call runs in
    a worker thread capped by a 3-second threading.Timer. On failure or
    timeout the text is spoken immediately with the local pyttsx3 engine.
    """
    if not text or (cancel is not None and cancel.is_set()):
        return

    result: dict = {}
    done = threading.Event()
    deliver_lock = threading.Lock()

    def _fetch() -> None:
        """Run every gTTS call (build + save) behind the 3-second cap."""
        mp3_path = None
        delivered = False
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3",
                                             delete=False) as tmp_file:
                mp3_path = tmp_file.name
            try:
                tts = gTTS(text, slow=(speed == "slow"), timeout=3)
            except TypeError:  # older gTTS without a timeout parameter
                tts = gTTS(text, slow=(speed == "slow"))
            tts.save(mp3_path)
            with deliver_lock:
                if not done.is_set():
                    result["mp3"] = mp3_path
                    delivered = True
        except Exception as error:  # noqa: BLE001 - handled by the fallback
            result["error"] = error
        finally:
            if not delivered and mp3_path:
                try:
                    os.remove(mp3_path)  # failed or late fetch: drop the file
                except OSError:
                    pass
            done.set()

    worker = threading.Thread(target=_fetch, daemon=True)
    worker.start()
    timer = threading.Timer(3.0, done.set)  # hard 3-second cap on gTTS
    timer.start()
    while not done.is_set():
        if cancel is not None and cancel.is_set():
            break
        done.wait(0.1)
    timer.cancel()

    if cancel is not None and cancel.is_set():
        with deliver_lock:
            done.set()  # abandon the fetch; _fetch cleans up undelivered files
            mp3_path = result.pop("mp3", None)
        if mp3_path:
            try:
                os.remove(mp3_path)
            except OSError:
                pass
        return

    mp3_path = result.get("mp3")
    if mp3_path:
        try:
            rate = FAST_RATE if speed == "fast" else 1.0
            _play_mp3(mp3_path, rate, cancel)
        finally:
            try:
                os.remove(mp3_path)  # delete the temp file after playing
            except OSError:
                pass
    else:
        # gTTS failed or timed out -> speak locally right away
        _speak_offline(text, speed, cancel)


def play_tone(frequency: int = 880, duration: float = 0.12,
              volume: float = 0.25) -> None:
    """Play a short confirmation tone (the audible "I'm listening" cue)."""
    try:
        rate = 22050
        t = np.linspace(0, duration, int(rate * duration), endpoint=False)
        tone = (np.sin(2 * np.pi * frequency * t) * volume).astype(np.float32)
        sd.play(tone, samplerate=rate)
        sd.wait()
    except Exception:  # noqa: BLE001 - audio feedback is optional
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


# Utterances that are just "wake me up" chatter with no real question yet.
WAKE_ONLY_TAILS = {
    "", "i have a question", "i've got a question", "ive got a question",
    "question", "yes", "yeah", "hello", "hi", "hey", "are you there",
    "can you hear me", "wake up", "you there", "it's me", "its me",
}


def extract_wake_question(text: str, wake_name: str) -> str | None:
    """Detect the wake phrase and split off the question that follows.

    Matches "Nova", "hey Nova", "ok Nova", "Nova I have a question", etc.
    Returns None when the wake phrase is absent, "" when the utterance is
    only the wake phrase (so the caller should listen for the question),
    or the question that follows it.
    """
    if not wake_name:
        return None
    match = re.match(
        rf"^\s*(?:please\s+)?(?:(?:hey(?:\s+there)?|ok(?:ay)?|hi|hello|yo)\s+)?"
        rf"{re.escape(wake_name)}\b(?P<tail>.*)$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    tail = match.group("tail").strip(" .,:!?")
    return "" if tail.lower() in WAKE_ONLY_TAILS else tail


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
                allowed_apps: dict | None = None,
                ai_provider: str | None = None,
                ollama_model: str | None = None,
                avatar_mode: str | None = None) -> None:
    """Persist the settings back to config.py."""
    apps = allowed_apps if allowed_apps is not None else {}
    provider = ai_provider or getattr(config, "AI_PROVIDER", AI_PROVIDER_GEMINI)
    if provider not in (AI_PROVIDER_GEMINI, AI_PROVIDER_OLLAMA):
        provider = AI_PROVIDER_GEMINI
    model = (ollama_model or getattr(config, "OLLAMA_MODEL",
                                     DEFAULT_OLLAMA_MODEL)).strip()
    if not model:
        model = DEFAULT_OLLAMA_MODEL
    mode = avatar_mode or getattr(config, "AVATAR_MODE", DEFAULT_AVATAR_MODE)
    if mode not in AVATAR_MODES:
        mode = DEFAULT_AVATAR_MODE
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    content = (
        '"""Configuration settings for the voice assistant."""\n\n'
        "# Get your Gemini API key from Google AI Studio: "
        "https://aistudio.google.com/apikey\n"
        "# NOTE: Do not commit a real API key to a public repository.\n"
        f"GEMINI_API_KEY = {api_key!r}\n\n"
        "# AI provider used by the desktop app: 'gemini' or 'ollama'.\n"
        f"AI_PROVIDER = {provider!r}\n\n"
        "# Ollama model name (download with: ollama pull <model>).\n"
        f"OLLAMA_MODEL = {model!r}\n\n"
        "# Desktop avatar: 'portrait', 'grokbot', or 'image'.\n"
        f"AVATAR_MODE = {mode!r}\n\n"
        "# Name the assistant introduces itself with (editable in the app's Settings).\n"
        f"ASSISTANT_NAME = {assistant_name!r}\n\n"
        "# Apps the assistant may open with 'open [app name]'.\n"
        f"ALLOWED_APPS = {apps!r}\n"
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


# ----------------------------------------------------------------------------
# Long-term memory (user name + facts the user asked to remember)
# ----------------------------------------------------------------------------

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "memory.json")


def load_memory() -> dict:
    """Load the user's memory (name + remembered facts) from memory.json."""
    try:
        with open(MEMORY_FILE, encoding="utf-8") as file:
            data = json.load(file)
        return {
            "user_name": str(data.get("user_name", "")),
            "facts": [str(f) for f in data.get("facts", [])],
        }
    except (OSError, ValueError):
        return {"user_name": "", "facts": []}


def save_memory(memory: dict) -> None:
    """Persist the user's memory to memory.json."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump({"user_name": memory.get("user_name", ""),
                       "facts": list(memory.get("facts", []))},
                      file, indent=2)
    except OSError:
        pass


def parse_memory_command(text: str) -> tuple[str, str] | None:
    """Parse memory commands exactly as spoken.

    Returns (action, payload) with action one of:
    set_name, remember, forget, forget_name, forget_all, recall_name, recall_all.
    Returns None when the utterance is not a memory command.
    """
    s = text.strip()

    # "my name is Ali" / "remember my name is Ali" / "call me Ali"
    m = re.match(
        r"^(?:please\s+)?(?:remember\s+(?:that\s+)?)?my name is (?P<name>.+)$",
        s, flags=re.IGNORECASE)
    if m:
        return "set_name", m.group("name").strip(" .!?\"'")
    m = re.match(
        r"^(?:please\s+)?(?:(?:you\s+)?can\s+)?call me (?P<name>.+)$",
        s, flags=re.IGNORECASE)
    if m:
        return "set_name", m.group("name").strip(" .!?\"'")

    # "forget my name" / "forget that my name is ..."
    if re.match(r"^forget\s+(?:that\s+)?my name\b.*$", s, flags=re.IGNORECASE):
        return "forget_name", ""

    # "forget everything" is handled by the generic forget below.
    # "forget i like pizza" / "forget that i like pizza"
    m = re.match(r"^forget\s+(?:that\s+)?(?P<fact>.+)$", s, flags=re.IGNORECASE)
    if m:
        fact = m.group("fact").strip(" .!?\"'")
        if fact.lower() in ("everything", "all", "all my memories",
                            "what you know", "what you know about me",
                            "your memories", "memories"):
            return "forget_all", ""
        return "forget", fact

    # "don't remember i like pizza" is a forget too
    m = re.match(
        r"^(?:do not|don't)\s+remember\s+(?:that\s+)?(?P<fact>.+)$",
        s, flags=re.IGNORECASE)
    if m:
        return "forget", m.group("fact").strip(" .!?\"'")

    # "what's my name" / "who am i"
    if re.match(r"^(?:what(?:'s|s| is) my name|who am i|do you know my name)\??$",
                s, flags=re.IGNORECASE):
        return "recall_name", ""

    # "what do you remember" / "what do you know about me" / "what are my memories"
    if re.match(r"^(?:what do you (?:remember|know about me)"
                r"|what are (?:my )?(?:your )?memories|what do you know)\??$",
                s, flags=re.IGNORECASE):
        return "recall_all", ""

    # "remember (that) i like pizza" — keep the fact verbatim as spoken
    m = re.match(
        r"^(?:please\s+)?remember\s+(?:that\s+)?(?P<fact>.+)$",
        s, flags=re.IGNORECASE)
    if m:
        return "remember", m.group("fact").strip(" .!?\"'")

    return None


def infer_emotion(status: str) -> str:
    """Map a status line to the face emotion."""
    s = status.lower()
    if "listening" in s:
        return "listening"
    if "thinking" in s or "searching" in s:
        return "thinking"
    if "speaking" in s:
        return "talking"
    if any(bad in s for bad in ("mic failed", "can't understand", "can't open",
                               "can't speak", "sorry", "failed", "enter both",
                               "select an app", "don't have a memory")):
        return "sad"
    if any(good in s for good in ("copied", "saved", "cleared", "added",
                                  "removed", "forgot", "got it",
                                  "i'll remember", "opening")):
        return "happy"
    return "idle"


class BubblyFace(tk.Canvas):
    """A 1:1 Grok Bot face: smooth pearl ball, twin capsule eyes, blush.

    The classic Grok-bot look is a clean round head with two solid black
    vertical capsule eyes and soft oval cheeks - no mouth, brows, glints or
    shine line. Expressions live in the capsules themselves: they grow,
    drift, tilt, squash to blink and chatter while talking. At rest the ball
    only breathes - the whole ball does not bounce around.
    """

    FRAME_MS = 80  # animation frame period

    def __init__(self, parent, size: int = 96, **kwargs) -> None:
        super().__init__(parent, width=size, height=size, bg=BG,
                         highlightthickness=0, borderwidth=0, **kwargs)
        self.size = size
        self.emotion = "idle"
        self._t = 0.0
        self._blink_in = 3.0
        self._blink_left = 0.0

        self.shadow = self.create_oval(0, 0, 0, 0, fill=MUTED, outline="")
        self.head = self.create_oval(0, 0, 0, 0, fill=WHITE,
                                     outline=ENTRY_BORDER, width=1)
        self.blush_l = self.create_oval(0, 0, 0, 0, fill=MUTED, outline="")
        self.blush_r = self.create_oval(0, 0, 0, 0, fill=MUTED, outline="")
        self.eye_l = self.create_line(0, 0, 0, 0, fill=BLACK, width=3,
                                      capstyle=tk.ROUND)
        self.eye_r = self.create_line(0, 0, 0, 0, fill=BLACK, width=3,
                                      capstyle=tk.ROUND)

        self.after(self.FRAME_MS, self._tick)

    def set_emotion(self, emotion: str) -> None:
        """Switch expression: idle, happy, listening, thinking, talking, sad."""
        if emotion in ("idle", "happy", "listening",
                       "thinking", "talking", "sad"):
            self.emotion = emotion

    def _tick(self) -> None:
        dt = self.FRAME_MS / 1000.0
        self._t += dt
        self._blink_in -= dt
        if self._blink_left > 0:
            self._blink_left -= dt
        elif self._blink_in <= 0:
            self._blink_left = 0.12
            self._blink_in = 2.6 + random.random() * 3.0
        self._layout()
        self.after(self.FRAME_MS, self._tick)

    def _layout(self) -> None:
        s = self.size
        t = self._t
        emo = self.emotion

        # Calm pearl head: a very slow, tiny breath; never bounces.
        cx = s / 2
        cy = s / 2
        rx = s * 0.42
        ry = s * 0.42
        if emo != "talking":
            cy += math.sin(t * 0.8) * 0.6
            rx *= 1.0 + math.sin(t * 0.8) * 0.004
            ry *= 1.0 - math.sin(t * 0.8) * 0.004

        # Gray layer behind, slightly offset -> crescent shadow = 3D ball
        off = s * 0.03
        self.coords(self.shadow, cx + off - rx, cy + off * 0.6 - ry,
                    cx + off + rx, cy + off * 0.6 + ry)
        self.coords(self.head, cx - rx, cy - ry, cx + rx, cy + ry)

        # Soft oval blush (part of the Grok-bot look)
        boost = 1.15 if emo == "happy" else 1.0
        brx, bry = s * 0.052 * boost, s * 0.036 * boost
        for item, side in ((self.blush_l, -1), (self.blush_r, 1)):
            bx = cx + side * rx * 0.58
            by = cy + ry * 0.42
            self.coords(item, bx - brx, by - bry, bx + brx, by + bry)

        # Twin capsule eyes - the whole expression lives in these.
        blinking = self._blink_left > 0
        eye_dx = s * 0.155
        eye_cy = cy + s * 0.02
        w = s * 0.086  # capsule stroke width (a capsule is 3w tall in total)
        h = w          # half-length of the straight part
        shear = 0.0    # horizontal lean of the eye tips
        dx = dy = 0.0

        if emo == "listening":
            w *= 1.18
            h *= 1.24
            dy = -s * 0.018
        elif emo == "thinking":
            dx = math.sin(t * 0.9) * s * 0.018 - s * 0.012
            dy = -s * 0.016
        elif emo == "sad":
            w *= 0.86
            h *= 0.9
            dy = s * 0.018
            shear = -s * 0.014  # tips lean inward = worried
        elif emo == "happy":
            w *= 1.32
            h *= 0.78
            dy = -s * 0.008
            shear = s * 0.02  # tips lean outward = the happy look
        elif emo == "talking":
            # No mouth in this style: the capsules chatter with the speech.
            h *= 0.85 + 0.4 * abs(math.sin(t * 9.0))
        else:  # idle: the eyes drift very slowly
            dx = math.sin(t * 0.5) * s * 0.006

        if blinking:
            h = w * 0.12

        for side, eye in ((-1, self.eye_l), (1, self.eye_r)):
            ex = cx + side * eye_dx + dx
            ey = eye_cy + dy
            tip = -side * shear  # positive tip leans the top of the eye outward
            self.coords(eye, ex - tip, ey - h, ex + tip, ey + h)
            self.itemconfig(eye, width=w)


def _load_avatar_photo(size: int):
    """Load avatar.png from the app folder, scaled to size x size.

    Uses PIL's ImageTk when available (exact scaling for any source size),
    falling back to plain tk.PhotoImage with integer subsample/zoom.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        AVATAR_FILE)
    if not os.path.isfile(path):
        return None
    try:
        if Image is not None and ImageTk is not None:
            img = Image.open(path).convert("RGBA").resize(
                (size, size), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
    except Exception:  # noqa: BLE001 - fall back to plain PhotoImage
        pass
    try:
        photo = tk.PhotoImage(file=path)
        w = photo.width()
        if w >= size:
            photo = photo.subsample(max(1, int(round(w / size))))
        else:
            photo = photo.zoom(max(1, int(round(size / w))))
        return photo
    except Exception:  # noqa: BLE001 - no usable image
        return None


class AvatarView(tk.Frame):
    """Optional still-image version with animated state indicators.

    The default is AnimatedReferenceAvatar, a mouthless redraw of the supplied
    design with animated eyes, brows, body movement, and state indicators.
    Set USE_IMAGE_AVATAR=True to use this original still-image view instead;
    set AVATAR_STYLE="grokbot" to use the earlier animated Grok Bot.
    """

    FRAME_MS = 120  # overlay animation tick
    WAVE_FRAMES = (  # bar-height patterns for the listening sound wave
        (0.35, 0.85, 0.50, 1.00, 0.45),
        (0.85, 0.45, 1.00, 0.40, 0.80),
        (0.50, 1.00, 0.40, 0.85, 0.60),
        (1.00, 0.55, 0.85, 0.55, 0.95),
    )

    def __init__(self, parent, size: int = 64, **kwargs) -> None:
        super().__init__(parent, width=size, height=size, bg=BG, **kwargs)
        self.size = size
        self.emotion = "idle"
        self.ok = False
        self._frame = 0

        self._photo = _load_avatar_photo(size)
        if self._photo is None:
            return  # no avatar image - the caller falls back to BubblyFace
        self.ok = True
        self.image_label = tk.Label(self, image=self._photo, bg=BG,
                                    borderwidth=0, highlightthickness=0)
        self.image_label.place(x=0, y=0, width=size, height=size)

        # Overlay chip in the bottom-right corner of the avatar.
        self.dots = tk.Label(self, text="", width=3, bg=PANEL, fg=WHITE,
                             font=("Segoe UI", 8, "bold"),
                             padx=2, pady=0, borderwidth=0)
        self.wave = tk.Canvas(self, width=26, height=16, bg=PANEL,
                              highlightthickness=0, borderwidth=0)
        self.after(self.FRAME_MS, self._tick)

    def set_emotion(self, emotion: str) -> None:
        """Switch expression: idle, happy, listening, thinking, talking, sad."""
        if emotion in ("idle", "happy", "listening",
                       "thinking", "talking", "sad"):
            self.emotion = emotion

    def _show_overlay(self, widget) -> None:
        for item in (self.dots, self.wave):
            if item is widget:
                item.place(relx=1.0, rely=1.0, anchor="se", x=-2, y=-2)
            else:
                item.place_forget()

    def _tick(self) -> None:
        emo = self.emotion
        if emo == "listening":
            self._show_overlay(self.wave)
            pattern = self.WAVE_FRAMES[self._frame % len(self.WAVE_FRAMES)]
            self._frame += 1
            self.wave.delete("all")
            x, bar_w, gap, mid, half = 2, 3, 2, 8, 6
            for height in pattern:
                self.wave.create_rectangle(
                    x, mid - height * half, x + bar_w, mid + height * half,
                    fill=WHITE, outline="")
                x += bar_w + gap
        elif emo == "thinking":
            self._show_overlay(self.dots)
            self._frame += 1
            self.dots.config(text="." * (self._frame % 3 + 1))
        elif emo == "talking":
            self._show_overlay(self.dots)
            self._frame += 1
            self.dots.config(text=("○", "●")[self._frame % 2])
        else:
            self._show_overlay(None)
        self.after(self.FRAME_MS, self._tick)


class AnimatedReferenceAvatar(tk.Label):
    """A fully drawn, mouthless animated portrait matching the supplied art.

    The bitmap is only a design reference. This renderer draws the character
    itself (swept dark hair, tilted peach face, thick brows, capsule eyes,
    catchlights, blush, ear, neck, and shoulders) and animates its gaze,
    blinks, expression, and subtle upper-body breathing.
    """

    FRAME_MS = 80
    SUPERSAMPLE = 4
    BG_COLOR = "#201E21"
    HAIR = "#402E2A"
    HAIR_DARK = "#302321"
    HAIR_MID = "#432F2B"
    SKIN = "#FDEBE1"
    SKIN_SHADE = "#F5D2C7"
    BLUSH = "#F7C8C6"
    EYE_INK = "#241A1A"
    EYE_GLINT = "#FFF9F3"
    BROW = "#3A2826"
    WAVE_FRAMES = (
        (0.35, 0.85, 0.50, 1.00, 0.45),
        (0.85, 0.45, 1.00, 0.40, 0.80),
        (0.50, 1.00, 0.40, 0.85, 0.60),
        (1.00, 0.55, 0.85, 0.55, 0.95),
    )

    def __init__(self, parent, size: int = 64, **kwargs) -> None:
        super().__init__(parent, width=size, height=size, bg=self.BG_COLOR,
                         borderwidth=0, highlightthickness=0, **kwargs)
        self.size = size
        self.emotion = "idle"
        self.ok = Image is not None and ImageDraw is not None and ImageTk is not None
        self._time = 0.0
        self._frame = 0
        self._blink_left = 0.0
        self._blink_in = 2.7 + random.random() * 2.2
        self._photo = None
        if not self.ok:
            return
        self._render()
        self.after(self.FRAME_MS, self._tick)

    def set_emotion(self, emotion: str) -> None:
        """Change expression using eyes, brows, and body pose only."""
        if emotion in ("idle", "happy", "listening", "thinking", "talking", "sad"):
            self.emotion = emotion

    @staticmethod
    def _bezier(start, segments, steps=18):
        """Sample cubic curves into a smooth path in 100-unit design space."""
        points = [start]
        x0, y0 = start
        for c1, c2, end in segments:
            x1, y1 = c1
            x2, y2 = c2
            x3, y3 = end
            for index in range(1, steps + 1):
                t = index / steps
                u = 1.0 - t
                x = (u ** 3 * x0 + 3 * u * u * t * x1
                     + 3 * u * t * t * x2 + t ** 3 * x3)
                y = (u ** 3 * y0 + 3 * u * u * t * y1
                     + 3 * u * t * t * y2 + t ** 3 * y3)
                points.append((x, y))
            x0, y0 = x3, y3
        return points

    @staticmethod
    def _oval_points(cx, cy, rx, ry, angle=0.0, count=36):
        """Return a rotated oval polygon in the same design coordinate space."""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        points = []
        for index in range(count):
            theta = 2 * math.pi * index / count
            x, y = rx * math.cos(theta), ry * math.sin(theta)
            points.append((cx + x * cos_a - y * sin_a,
                           cy + x * sin_a + y * cos_a))
        return points

    def _render_frame(self):
        """Draw the reference-inspired character from shapes, not the bitmap."""
        if not self.ok:
            return None
        ss = self.SUPERSAMPLE
        scale = self.size * ss / 100.0
        frame = Image.new("RGB", (self.size * ss, self.size * ss), self.BG_COLOR)
        draw = ImageDraw.Draw(frame)
        emotion = self.emotion
        breath = math.sin(self._time * 1.7) * 2.0
        sway = math.sin(self._time * 1.05) * 0.7
        head_y = -breath * 0.24
        body_y = breath * 1.1
        if emotion == "listening":
            head_y -= 0.55
        elif emotion == "thinking":
            head_y += 0.22
        head_x = sway + (-0.32 if emotion == "listening" else 0.0)

        def scaled(points, dx=0.0, dy=0.0):
            return [((x + dx) * scale, (y + dy) * scale) for x, y in points]

        def fill_path(start, segments, color, dx=0.0, dy=0.0, steps=18):
            points = self._bezier(start, segments, steps)
            draw.polygon(scaled(points, dx, dy), fill=color)

        def fill_oval(box, color):
            x0, y0, x1, y1 = box
            draw.ellipse((x0 * scale, y0 * scale, x1 * scale, y1 * scale),
                         fill=color)

        def stroke_path(start, segments, color, width, dx=0.0, dy=0.0):
            points = scaled(self._bezier(start, segments), dx, dy)
            px_width = max(1, round(width * scale))
            draw.line(points, fill=color, width=px_width, joint="curve")
            radius = px_width / 2
            for x, y in (points[0], points[-1]):
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

        def paint_oval(cx, cy, rx, ry, color, angle=0.0, dx=0.0, dy=0.0):
            points = self._oval_points(cx + dx, cy + dy, rx, ry, angle)
            draw.polygon(scaled(points), fill=color)

        # Back hair fills the silhouette and continues around the shoulders.
        fill_oval((-12 + head_x, -20 + head_y, 111 + head_x, 111 + head_y),
                  self.HAIR)
        fill_path(
            (66, 14),
            [((83, 17), (91, 31), (90, 47)),
             ((89, 64), (79, 81), (75, 101)),
             ((67, 103), (60, 98), (56, 93)),
             ((65, 77), (72, 61), (75, 48)),
             ((77, 34), (72, 22), (66, 14))],
            self.HAIR_DARK, head_x * 0.35, head_y,
        )

        # Shoulders and upper torso move gently as the character breathes.
        fill_path(
            (12, 103),
            [((15, 93), (23, 88), (31, 88)),
             ((38, 88), (43, 94), (49, 91)),
             ((60, 86), (73, 91), (82, 103)),
             ((61, 108), (32, 108), (12, 103))],
            "#2A282C", sway * 0.45, body_y,
        )
        fill_path(
            (31, 74),
            [((37, 78), (49, 80), (57, 75)),
             ((55, 86), (57, 96), (63, 103)),
             ((48, 105), (32, 104), (23, 103)),
             ((30, 93), (31, 83), (31, 74))],
            self.SKIN_SHADE, head_x * 0.2, body_y * 0.75,
        )
        # Right ear, then the softly tilted face.
        paint_oval(72, 65, 9.0, 10.5, self.SKIN_SHADE,
                   angle=-0.12, dx=head_x, dy=head_y)
        paint_oval(74, 65, 3.1, 5.0, "#EAB8AE",
                   angle=-0.12, dx=head_x, dy=head_y)
        fill_path(
            (39, 26),
            [((53, 23), (66, 30), (71, 42)),
             ((76, 54), (72, 68), (64, 78)),
             ((57, 87), (48, 91), (38, 90)),
             ((25, 89), (15, 81), (10, 71)),
             ((5, 60), (7, 47), (13, 38)),
             ((20, 29), (30, 25), (39, 26))],
            self.SKIN, head_x, head_y,
        )

        # Long side locks and the sweeping fringe reproduce the reference hair.
        fill_path(
            (-4, 38),
            [((7, 40), (14, 48), (15, 59)),
             ((17, 74), (11, 88), (15, 103)),
             ((8, 104), (1, 102), (-4, 99)),
             ((-2, 78), (-2, 55), (-4, 38))],
            self.HAIR_DARK, head_x * 0.75, head_y,
        )
        fill_path(
            (-4, 43),
            [((0, 26), (6, 12), (19, 4)),
             ((32, -4), (49, -3), (61, 4)),
             ((73, 10), (80, 22), (80, 34)),
             ((81, 41), (78, 47), (74, 50)),
             ((68, 42), (62, 37), (55, 34)),
             ((47, 30), (39, 31), (32, 35)),
             ((24, 40), (20, 47), (16, 55)),
             ((11, 63), (4, 68), (-4, 68))],
            self.HAIR_MID, head_x, head_y,
        )
        # Swept shadow in the bangs; deliberately no shine streak.
        fill_path(
            (-4, 40),
            [((7, 32), (16, 25), (27, 23)),
             ((40, 19), (53, 21), (63, 28)),
             ((51, 25), (40, 27), (31, 33)),
             ((23, 38), (18, 45), (14, 53)),
             ((10, 59), (4, 63), (-4, 63))],
            self.HAIR_DARK, head_x, head_y,
        )

        # Expression is carried only by brows, eyes, catchlights, and blush.
        brow_raise = -1.6 if emotion == "listening" else 0.0
        if emotion == "sad":
            left_brow = ((15, 44), [((21, 41), (27, 37), (34, 38))])
            right_brow = ((52, 47), [((59, 49), (66, 54), (71, 58))])
        elif emotion == "thinking":
            left_brow = ((15, 42), [((21, 39), (28, 39), (34, 42))])
            right_brow = ((52, 48), [((59, 48), (66, 51), (71, 55))])
        else:
            left_brow = ((15, 42), [((21, 39), (28, 40), (34, 44))])
            right_brow = ((52, 49), [((59, 50), (66, 53), (71, 57))])
        stroke_path(*left_brow, self.BROW, 3.3, head_x, head_y + brow_raise)
        stroke_path(*right_brow, self.BROW, 3.3, head_x, head_y + brow_raise)

        blush_color = ("#F5B9B9" if emotion == "happy" else
                       "#F3D1CE" if emotion == "sad" else self.BLUSH)
        paint_oval(15.5, 63.0, 6.0, 3.7, blush_color, angle=0.24,
                   dx=head_x, dy=head_y)
        paint_oval(62.5, 72.0, 6.5, 3.7, blush_color, angle=0.20,
                   dx=head_x, dy=head_y)

        # The reference has large dark, slightly tilted capsule eyes with slim glints.
        gaze_x = 0.0
        gaze_y = 0.0
        if emotion == "thinking":
            gaze_x, gaze_y = 1.3, -1.35
        elif emotion == "sad":
            gaze_y = 1.0
        elif emotion == "listening":
            gaze_x = math.sin(self._time * 2.0) * 0.28
        elif emotion == "talking":
            gaze_x = math.sin(self._time * 3.2) * 0.35
        else:
            gaze_x = math.sin(self._time * 0.75) * 0.30
            gaze_y = math.sin(self._time * 0.55) * 0.20

        eyes = ((24.0, 54.5, -0.16), (58.0, 64.0, 0.16))
        closed = self._blink_left > 0 or emotion == "happy"
        for index, (cx, cy, angle) in enumerate(eyes):
            ex, ey = cx + gaze_x + head_x, cy + gaze_y + head_y
            if closed:
                lid_y = ey + (0.4 if emotion == "happy" else 0.0)
                stroke_path((ex - 6.0, lid_y),
                            [((ex - 2.5, lid_y + 1.2),
                              (ex + 2.5, lid_y + 1.2),
                              (ex + 6.0, lid_y))],
                            self.EYE_INK, 2.4 if emotion == "happy" else 2.0)
                continue
            eye_height = 9.5
            eye_width = 5.35
            if emotion == "listening":
                eye_height *= 1.14
            elif emotion == "sad":
                eye_height *= 0.78
            eye_points = self._oval_points(ex, ey, eye_width, eye_height,
                                           angle=angle, count=40)
            draw.polygon(scaled(eye_points), fill=self.EYE_INK)
            glint_dx = -1.35 + (0.18 if emotion == "thinking" else 0.0)
            glint_dy = -3.1 + (0.20 * math.sin(self._time * 2.3))
            paint_oval(ex + glint_dx, ey + glint_dy, 1.05, 2.65,
                       self.EYE_GLINT, angle=angle)

        # Speech remains mouthless: a compact sound wave animates while talking.
        if emotion in ("listening", "thinking", "talking"):
            if emotion in ("listening", "talking"):
                pattern = self.WAVE_FRAMES[self._frame % len(self.WAVE_FRAMES)]
                mid, half = 91 * scale, 5.0 * scale
                bar_w = 2.1 * scale
                for index, height in enumerate(pattern):
                    x = (80.0 + index * 3.7) * scale
                    top = mid - height * half
                    bottom = mid + height * half
                    draw.rounded_rectangle(
                        (x, top, x + bar_w, bottom),
                        radius=bar_w / 2, fill="#FAFAFA",
                    )
            else:
                active = self._frame % 3
                for index in range(3):
                    x = (83 + index * 5) * scale
                    r = (1.15 if active == index else 0.75) * scale
                    cy = 91 * scale
                    fill_oval((x-r, cy-r, x+r, cy+r),
                              "#FAFAFA" if active == index else "#8C8C8C")

        resampling = (Image.Resampling.LANCZOS if hasattr(Image, "Resampling")
                      else Image.LANCZOS)
        return frame.resize((self.size, self.size), resampling)

    def _render(self) -> None:
        frame = self._render_frame()
        if frame is None:
            return
        photo = ImageTk.PhotoImage(frame, master=self)
        self.configure(image=photo)
        self._photo = photo

    def _tick(self) -> None:
        dt = self.FRAME_MS / 1000.0
        self._time += dt
        self._frame += 1
        if self._blink_left > 0:
            self._blink_left = max(0.0, self._blink_left - dt)
        else:
            self._blink_in -= dt
            if self._blink_in <= 0:
                self._blink_left = 0.16
                self._blink_in = 2.7 + random.random() * 2.5
        self._render()
        self.after(self.FRAME_MS, self._tick)




# ----------------------------------------------------------------------------
# The desktop app
# ----------------------------------------------------------------------------

class VoiceAssistantApp:
    """tkinter desktop app for the voice assistant."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.api_key = getattr(config, "GEMINI_API_KEY", "")
        self.ai_provider = str(getattr(config, "AI_PROVIDER", AI_PROVIDER_GEMINI)).lower()
        if self.ai_provider not in (AI_PROVIDER_GEMINI, AI_PROVIDER_OLLAMA):
            self.ai_provider = AI_PROVIDER_GEMINI
        self.ollama_model = str(
            getattr(config, "OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            or DEFAULT_OLLAMA_MODEL
        ).strip()
        if not self.ollama_model:
            self.ollama_model = DEFAULT_OLLAMA_MODEL
        self.avatar_mode = str(
            getattr(config, "AVATAR_MODE", DEFAULT_AVATAR_MODE)
        ).lower()
        if self.avatar_mode not in AVATAR_MODES:
            self.avatar_mode = DEFAULT_AVATAR_MODE
        self.assistant_name = getattr(config, "ASSISTANT_NAME", DEFAULT_ASSISTANT_NAME)
        self.allowed_apps: dict[str, str] = dict(
            getattr(config, "ALLOWED_APPS", {}) or {}
        )
        self.memory = load_memory()  # user name + remembered facts
        self.client = None
        self.chat = None
        self._gemini_error = ""
        self._ollama_client = None
        self._ollama_history: list[dict[str, str]] = []
        self.tray_icon = None

        self.speech_lock = threading.Lock()
        self._audio_lock = threading.Lock()  # one question recording at a time
        self._busy_event = threading.Event()  # set while listening/thinking
        self._speaking_event = threading.Event()  # set while TTS plays
        self._speech_cancel = threading.Event()  # set to stop playback
        # _listening_mode switches the ONE shared mic stream between its two
        # consumers: 'wake' (chunks fill the rolling wake buffer) or
        # 'question' (chunks are captured into _question_chunks).
        self._listening_mode = "wake"
        self._mic_queue = queue.Queue()  # every chunk from the shared stream
        self._wake_buffer = collections.deque(  # rolling 3-second buffer
            maxlen=max(int(WAKE_WINDOW_SECONDS / CHUNK_SECONDS), 2))
        self._question_chunks: list = []  # chunks while capturing a question
        self._mic_lock = threading.Lock()  # guards the rolling wake buffer
        self._chunks_lock = threading.Lock()  # guards _question_chunks
        self._mic_rate = SAMPLE_RATE
        self._wake_last_check = 0.0
        self._wake_unclear = 0
        self._running = True
        self._speed = "normal"
        self._last_answer = ""
        self._pending_quit = False
        self._msg_counter = 0
        self._copy_texts: dict[str, str] = {}
        self._typing_active = False
        self._typing_start = "1.0"
        self._typing_ticks = 0
        self._typing_job = None
        self._settings_win = None
        self._dot_blink_job = None
        self._dot_color = MUTED

        root.title(f"{self.assistant_name} — Voice Assistant")
        root.geometry("900x620")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self._hide_to_tray)

        self._build_ui()
        self._configure_ai()
        init_offline_speech()  # pyttsx3 fallback engine, created once
        self._greet()
        self._setup_tray()
        threading.Thread(target=self._mic_thread, name="_mic_thread",
                         daemon=True).start()
        threading.Thread(target=self._audio_router_thread,
                         name="_audio_router_thread", daemon=True).start()

    # ----- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        # Header: avatar + name on the left; status dot + gear on the right.
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill=tk.X, padx=18, pady=(12, 8))
        self._avatar_header = header
        self.face = self._make_avatar_widget(header, self.avatar_mode)
        self.face.pack(side=tk.LEFT, padx=(0, 12))
        title_box = tk.Frame(header, bg=BG)
        title_box.pack(side=tk.LEFT)
        self.title_label = tk.Label(
            title_box, text=self.assistant_name,
            font=FONT_TITLE, bg=BG, fg=TEXT,
        )
        self.title_label.pack(anchor="w")
        tk.Label(
            title_box, text="voice assistant", font=FONT, bg=BG, fg=MUTED,
        ).pack(anchor="w")

        self.gear_button = tk.Button(
            header, text="\u2699 Settings", font=FONT_BOLD,
            command=self._open_settings,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, padx=14, pady=8, cursor="hand2", borderwidth=0,
        )
        self.gear_button.pack(side=tk.RIGHT)
        self.status_dot = tk.Canvas(header, width=12, height=12, bg=BG,
                                    highlightthickness=0, borderwidth=0)
        self.status_dot.pack(side=tk.RIGHT, padx=(0, 10))
        self._dot = self.status_dot.create_oval(1, 1, 11, 11,
                                                fill=MUTED, outline="")

        # Thin #242424 separator line between the header and the chat window.
        tk.Frame(self.root, bg=ENTRY_BORDER, height=1).pack(fill=tk.X)

        # Chat window (conversation history) - full width, no border.
        chat_frame = tk.Frame(self.root, bg=CARD)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 6))
        self.chat_view = tk.Text(
            chat_frame, bg=CARD, fg=TEXT, font=FONT, relief=tk.FLAT,
            wrap=tk.WORD, state=tk.DISABLED, padx=18, pady=14, spacing3=6,
            cursor="arrow", borderwidth=0, highlightthickness=0,
        )
        scrollbar = tk.Scrollbar(chat_frame, command=self.chat_view.yview,
                                 width=10, relief=tk.FLAT, bg=ENTRY_BORDER,
                                 activebackground=MUTED, troughcolor=CARD)
        self.chat_view.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_view.tag_configure("user_name", foreground=TEXT,
                                     font=FONT_BOLD, justify=tk.RIGHT)
        self.chat_view.tag_configure("assistant_name", foreground=MUTED,
                                     font=FONT_BOLD, justify=tk.LEFT)
        self.chat_view.tag_configure("system_name", foreground=MUTED,
                                     font=FONT_BOLD)
        self.chat_view.tag_configure("user_text", foreground=TEXT,
                                     justify=tk.RIGHT,
                                     background=GRAY, spacing1=6, spacing3=6,
                                     lmargin1=60, lmargin2=60)
        self.chat_view.tag_configure("assistant_text", foreground=MUTED,
                                     justify=tk.LEFT)
        self.chat_view.tag_configure("system_text", foreground=MUTED,
                                     lmargin1=60, lmargin2=60)
        self.chat_view.tag_configure("typing_line", foreground=MUTED,
                                     font=("Segoe UI", 10, "italic"))
        self.chat_view.tag_configure("copy_link", foreground=WHITE,
                                     font=("Segoe UI", 8, "underline"))

        # Input row: entry full width, Mic + Send buttons on the right.
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill=tk.X, padx=14, pady=(2, 12))
        input_row = tk.Frame(bottom, bg=BG)
        input_row.pack(fill=tk.X)
        self.send_button = tk.Button(
            input_row, text="Send", font=FONT_BOLD, command=self.on_send,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, padx=18, pady=8, cursor="hand2", borderwidth=0,
        )
        self.send_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.mic_button = tk.Button(
            input_row, text="Mic", font=FONT_BOLD, command=self.on_mic,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, padx=18, pady=8, cursor="hand2", borderwidth=0,
        )
        self.mic_button.pack(side=tk.RIGHT)
        self.entry = tk.Entry(
            input_row, bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT, font=FONT,
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=ENTRY_BORDER, highlightcolor=WHITE,
            borderwidth=0,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8,
                        padx=(0, 8))
        self.entry.bind("<Return>", lambda _event: self.on_send())

        # Tiny status line below the input.
        self.status_label = tk.Label(
            bottom, text=f"Say {self.assistant_name} to start",
            font=("Segoe UI", 8), bg=BG, fg=MUTED,
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, pady=(6, 0))

    def _make_avatar_widget(self, parent, mode: str):
        """Build one of the three selectable header avatars."""
        if mode == AVATAR_MODE_GROKBOT:
            return BubblyFace(parent, size=48)
        if mode == AVATAR_MODE_IMAGE:
            candidate = AvatarView(parent, size=64)
        else:
            candidate = AnimatedReferenceAvatar(parent, size=72)
        if candidate.ok:
            return candidate
        candidate.destroy()
        return BubblyFace(parent, size=48)

    def _switch_avatar(self, mode: str) -> None:
        """Replace the header avatar while retaining its current expression."""
        if mode not in AVATAR_MODES:
            mode = DEFAULT_AVATAR_MODE
        emotion = getattr(self.face, "emotion", "idle")
        self.face.destroy()
        self.avatar_mode = mode
        self.face = self._make_avatar_widget(self._avatar_header, mode)
        self.face.set_emotion(emotion)
        self.face.pack(side=tk.LEFT, padx=(0, 12))

    def _open_settings(self) -> None:
        """Open the settings popup (the gear button in the header)."""
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.lift()
            self._settings_win.focus_force()
            return
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.configure(bg=BG)
        win.geometry("440x680")
        win.transient(self.root)
        self._settings_win = win

        body = tk.Frame(win, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        tk.Label(body, text="Settings", font=FONT_SECTION,
                 bg=BG, fg=TEXT).pack(anchor="w")

        def _label(text: str) -> None:
            tk.Label(body, text=text, font=FONT_SMALL,
                     bg=BG, fg=MUTED).pack(anchor="w", pady=(12, 0))

        def _entry(show: str = "", parent=None) -> tk.Entry:
            parent = parent or body
            widget = tk.Entry(
                parent, show=show, bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                font=FONT, relief=tk.FLAT, highlightthickness=1,
                highlightbackground=ENTRY_BORDER, highlightcolor=WHITE,
                borderwidth=0,
            )
            widget.pack(fill=tk.X, pady=(4, 0), ipady=6)
            return widget

        _label("ASSISTANT NAME")
        self.name_entry = _entry()
        self.name_entry.insert(0, self.assistant_name)

        _label("AVATAR")
        self._avatar_mode_var = tk.StringVar(value=self.avatar_mode)
        avatar_row = tk.Frame(body, bg=BG)
        avatar_row.pack(fill=tk.X, pady=(4, 0))
        for text, value in (("Animated portrait", AVATAR_MODE_PORTRAIT),
                            ("Classic Grok Bot", AVATAR_MODE_GROKBOT),
                            ("Static image", AVATAR_MODE_IMAGE)):
            tk.Radiobutton(
                avatar_row, text=text, value=value,
                variable=self._avatar_mode_var, bg=BG, fg=TEXT,
                selectcolor=GRAY, activebackground=BG,
                activeforeground=WHITE, highlightthickness=0, borderwidth=0,
                font=("Segoe UI", 8),
            ).pack(side=tk.LEFT, expand=True, anchor="w")

        _label("AI PROVIDER")
        self._ai_provider_var = tk.StringVar(value=self.ai_provider)
        provider_row = tk.Frame(body, bg=BG)
        provider_row.pack(fill=tk.X, pady=(4, 0))
        for text, value in (("Gemini API", AI_PROVIDER_GEMINI),
                            ("Local AI (Ollama)", AI_PROVIDER_OLLAMA)):
            tk.Radiobutton(
                provider_row, text=text, value=value,
                variable=self._ai_provider_var,
                command=self._update_provider_fields, bg=BG, fg=TEXT,
                selectcolor=GRAY, activebackground=BG,
                activeforeground=WHITE, highlightthickness=0, borderwidth=0,
                font=FONT,
            ).pack(side=tk.LEFT, expand=True, anchor="w")

        self._provider_fields = tk.Frame(body, bg=BG)
        self._provider_fields.pack(fill=tk.X)
        self._gemini_fields = tk.Frame(self._provider_fields, bg=BG)
        tk.Label(self._gemini_fields, text="GEMINI API KEY", font=FONT_SMALL,
                 bg=BG, fg=MUTED).pack(anchor="w", pady=(8, 0))
        self.key_entry = _entry(show="\u2022", parent=self._gemini_fields)
        if self.api_key and self.api_key != PLACEHOLDER_KEY:
            self.key_entry.insert(0, self.api_key)
        tk.Label(self._gemini_fields, text="Stored in config.py",
                 font=("Segoe UI", 8), bg=BG, fg=MUTED).pack(anchor="w")

        self._ollama_fields = tk.Frame(self._provider_fields, bg=BG)
        tk.Label(self._ollama_fields, text="OLLAMA MODEL", font=FONT_SMALL,
                 bg=BG, fg=MUTED).pack(anchor="w", pady=(8, 0))
        self.ollama_model_entry = _entry(parent=self._ollama_fields)
        self.ollama_model_entry.insert(0, self.ollama_model)
        tk.Label(
            self._ollama_fields,
            text=f"Default: {DEFAULT_OLLAMA_MODEL} · Server: {OLLAMA_HOST}",
            font=("Segoe UI", 8), bg=BG, fg=MUTED,
        ).pack(anchor="w")
        self._update_provider_fields()

        _label("VOICE SPEED")
        self._speed_var = tk.StringVar(value=self._speed)
        speeds = tk.Frame(body, bg=BG)
        speeds.pack(fill=tk.X, pady=(4, 0))
        for text, value in (("Slow", "slow"), ("Normal", "normal"),
                            ("Fast", "fast")):
            tk.Radiobutton(
                speeds, text=text, value=value, variable=self._speed_var,
                command=self._on_speed_change, bg=BG, fg=TEXT,
                selectcolor=GRAY, activebackground=BG,
                activeforeground=WHITE, highlightthickness=0, borderwidth=0,
                font=FONT,
            ).pack(side=tk.LEFT, expand=True, anchor="w")

        _label("ALLOWED APPS")
        self.apps_list = tk.Listbox(
            body, bg=ENTRY_BG, fg=TEXT, font=("Segoe UI", 9), relief=tk.FLAT,
            highlightthickness=1, highlightbackground=ENTRY_BORDER,
            selectbackground=WHITE, selectforeground=BLACK, height=5,
            activestyle="none", borderwidth=0,
        )
        self.apps_list.pack(fill=tk.X, pady=(4, 0))
        self.apps_list.bind("<<ListboxSelect>>", self._on_app_select)
        self._refresh_apps_list()
        self.app_name_entry = _entry()
        self.app_name_entry.insert(0, "app name")
        self.app_path_entry = _entry()
        self.app_path_entry.insert(0, "file path")
        apps_buttons = tk.Frame(body, bg=BG)
        apps_buttons.pack(fill=tk.X, pady=(8, 0))
        self.add_app_button = tk.Button(
            apps_buttons, text="Add", font=FONT_BOLD, command=self._add_app,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, pady=6, cursor="hand2", borderwidth=0,
        )
        self.add_app_button.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=(0, 6))
        self.remove_app_button = tk.Button(
            apps_buttons, text="Remove", font=FONT_BOLD,
            command=self._remove_app,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, pady=6, cursor="hand2", borderwidth=0,
        )
        self.remove_app_button.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                    padx=(6, 0))

        save_button = tk.Button(
            body, text="Save settings", font=FONT_BOLD,
            command=self.save_settings,
            bg=WHITE, fg=BLACK, activebackground=WHITE,
            activeforeground=BLACK, relief=tk.FLAT, pady=8, cursor="hand2",
            borderwidth=0,
        )
        save_button.pack(fill=tk.X, pady=(16, 0))
        clear_button = tk.Button(
            body, text="Clear chat", font=FONT_BOLD, command=self._clear_chat,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, pady=8, cursor="hand2", borderwidth=0,
        )
        clear_button.pack(fill=tk.X, pady=(8, 0))

    def _update_provider_fields(self) -> None:
        """Show only the credential/model field for the selected provider."""
        if not hasattr(self, "_gemini_fields"):
            return
        self._gemini_fields.pack_forget()
        self._ollama_fields.pack_forget()
        if self._ai_provider_var.get() == AI_PROVIDER_OLLAMA:
            self._ollama_fields.pack(fill=tk.X)
        else:
            self._gemini_fields.pack(fill=tk.X)

    def _set_mic_speaking(self, speaking: bool) -> None:
        """Mic button doubles as Stop: red while the assistant speaks."""

        def apply() -> None:
            if speaking:
                self.mic_button.config(
                    text="Stop", bg=STOP_RED, fg=WHITE, state=tk.NORMAL,
                    activebackground=STOP_RED, activeforeground=WHITE,
                )
            else:
                self.mic_button.config(
                    text="Mic", bg=GRAY, fg=TEXT,
                    activebackground=WHITE, activeforeground=BLACK,
                )

        self.root.after(0, apply)

    def _set_dot(self, emotion: str) -> None:
        """Header status dot: gray=idle, white=listening, blinking=speaking."""
        if self._dot_blink_job is not None:
            try:
                self.root.after_cancel(self._dot_blink_job)
            except Exception:  # noqa: BLE001 - job may already have run
                pass
            self._dot_blink_job = None
        if emotion == "listening":
            self._dot_color = WHITE
            self.status_dot.itemconfig(self._dot, fill=WHITE)
        elif emotion == "talking":
            self._dot_color = WHITE
            self.status_dot.itemconfig(self._dot, fill=WHITE)
            self._dot_blink_job = self.root.after(400, self._blink_dot)
        elif emotion == "thinking":
            self._dot_color = MUTED
            self.status_dot.itemconfig(self._dot, fill=MUTED)
            self._dot_blink_job = self.root.after(500, self._blink_dot)
        else:
            self._dot_color = MUTED
            self.status_dot.itemconfig(self._dot, fill=MUTED)

    def _blink_dot(self) -> None:
        current = self.status_dot.itemcget(self._dot, "fill")
        self.status_dot.itemconfig(
            self._dot, fill=BG if current != BG else self._dot_color,
        )
        self._dot_blink_job = self.root.after(400, self._blink_dot)

    # ----- Gemini -----------------------------------------------------------

    def _system_instruction(self) -> str:
        """Build the shared assistant prompt, including saved user memories."""
        memory_notes = []
        if self.memory.get("user_name"):
            memory_notes.append(f"- The user's name is {self.memory['user_name']}.")
        for fact in self.memory.get("facts", []):
            memory_notes.append(f"- Remembered about the user: {fact}.")
        memory_block = ""
        if memory_notes:
            memory_block = (
                "\n\nThings you remember about the user "
                "(use them naturally in conversation; never recite this list):\n"
                + "\n".join(memory_notes)
            )
        return (
            f"You are {self.assistant_name}, a friendly voice assistant. "
            "Keep answers clear and conversational." + memory_block
        )

    def _configure_gemini(self) -> None:
        """Prepare Gemini for normal use or as the Local AI fallback."""
        self._gemini_error = ""
        if not self.api_key or self.api_key == PLACEHOLDER_KEY:
            self.client = None
            self.chat = None
            self._gemini_error = "Gemini API key is not configured."
            return
        try:
            history = self.chat.get_history() if self.chat is not None else []
            self.client = genai.Client(api_key=self.api_key)
            self.chat = self.client.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction(),
                ),
                history=list(history),
            )
        except Exception as error:  # noqa: BLE001 - Local AI can still work
            self.client = None
            self.chat = None
            self._gemini_error = str(error) or type(error).__name__

    def _configure_ai(self) -> None:
        """Configure the selected provider and keep Gemini ready for fallback."""
        self._configure_gemini()
        if self.ai_provider == AI_PROVIDER_OLLAMA:
            self._set_idle_status()
        elif self.chat is None:
            if self._gemini_error == "Gemini API key is not configured.":
                self._set_status(
                    "Set your Gemini API key in Settings to start chatting.")
            else:
                self._set_status(f"Gemini setup problem: {self._gemini_error}")
        else:
            self._set_idle_status()

    def _gemini_reply(self, prompt: str) -> str:
        """Send one prompt to the configured Gemini conversation."""
        if self.chat is None:
            raise RuntimeError(self._gemini_error or
                               "Gemini API is not configured.")
        response = self.chat.send_message(prompt)
        return (response.text or "").strip() or "(no response)"

    def _remember_ollama_turn(self, prompt: str, answer: str) -> None:
        self._ollama_history.extend((
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ))

    def _ollama_reply(self, prompt: str) -> str:
        """Send a conversation to the user's local Ollama server."""
        if ollama is None:
            raise RuntimeError(
                "The ollama Python package is missing; install requirements.txt.")
        messages = [
            {"role": "system", "content": self._system_instruction()},
            *self._ollama_history,
            {"role": "user", "content": prompt},
        ]
        if self._ollama_client is None:
            self._ollama_client = ollama.Client(host=OLLAMA_HOST)
        response = self._ollama_client.chat(
            model=self.ollama_model,
            messages=messages,
        )
        message = (response.get("message") if isinstance(response, dict)
                   else getattr(response, "message", None))
        content = (message.get("content", "") if isinstance(message, dict)
                   else getattr(message, "content", ""))
        answer = str(content or "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        self._remember_ollama_turn(prompt, answer)
        return answer

    # ----- Thread-safe UI updates -------------------------------------------

    def _set_status(self, text: str) -> None:
        def apply() -> None:
            self.status_label.config(text=text, fg=MUTED)
            emotion = infer_emotion(text)
            self.face.set_emotion(emotion)
            self._set_dot(emotion)

        self.root.after(0, apply)

    def _set_idle_status(self) -> None:
        """Status shown when idle - always visible, never blank."""
        self._set_status(f"Say {self.assistant_name} to start")

    def _append_message(self, role: str, text: str) -> None:
        def insert() -> None:
            self._hide_typing_now()
            names = {"user": "You", "assistant": self.assistant_name, "system": "Notice"}
            self._msg_counter += 1
            copy_tag = f"copy_{self._msg_counter}"
            self._copy_texts[copy_tag] = text
            self.chat_view.configure(state=tk.NORMAL)
            if role == "user":
                # User: name row, then a #141414 bubble with 6px padding.
                self.chat_view.insert(tk.END, f"{names[role]}", f"{role}_name")
                self.chat_view.insert(tk.END, "   [Copy]\n",
                                      ("copy_link", copy_tag))
                self.chat_view.insert(tk.END, "   ", f"{role}_text")
                self.chat_view.insert(tk.END, f"{text}   ", f"{role}_text")
                self.chat_view.insert(tk.END, "\n\n")
            elif role == "assistant":
                # Assistant: the name in #8C8C8C before the message text.
                self.chat_view.insert(tk.END, f"{names[role]}: ", f"{role}_name")
                self.chat_view.insert(tk.END, f"{text}   ", f"{role}_text")
                self.chat_view.insert(tk.END, "[Copy]\n\n",
                                      ("copy_link", copy_tag))
            else:
                self.chat_view.insert(tk.END, f"{names[role]}", f"{role}_name")
                self.chat_view.insert(tk.END, "   [Copy]\n",
                                      ("copy_link", copy_tag))
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
        self._set_status("Copied to clipboard.")

    def _set_busy(self, busy: bool) -> None:
        def apply() -> None:
            self.send_button.config(state=tk.DISABLED if busy else tk.NORMAL)
            if not self._speaking_event.is_set():
                self.mic_button.config(
                    state=tk.DISABLED if busy else tk.NORMAL)

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
        """Mic button: start listening, or stop speech while speaking."""
        if self._speaking_event.is_set():
            self._stop_speaking()
            return
        self._switch_listening("question")  # switch the shared stream directly
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _on_speed_change(self) -> None:
        self._speed = self._speed_var.get()

    def _stop_speaking(self) -> None:
        self._speech_cancel.set()
        try:
            sd.stop()  # interrupts sounddevice playback mid-sentence
        except Exception:  # noqa: BLE001 - no active stream
            pass
        self._speaking_event.clear()
        self._set_mic_speaking(False)
        self._set_idle_status()

    def _clear_chat(self) -> None:
        self.chat_view.configure(state=tk.NORMAL)
        self.chat_view.delete("1.0", tk.END)
        self.chat_view.configure(state=tk.DISABLED)
        self._copy_texts.clear()
        self._typing_start = "1.0"
        self.chat = None
        self._ollama_history.clear()
        self._configure_ai()  # fresh chat with empty history
        self._set_status("Chat cleared — memory reset.")

    def save_settings(self) -> None:
        self.assistant_name = self.name_entry.get().strip() or DEFAULT_ASSISTANT_NAME
        self.api_key = self.key_entry.get().strip()
        self.ai_provider = self._ai_provider_var.get()
        if self.ai_provider not in (AI_PROVIDER_GEMINI, AI_PROVIDER_OLLAMA):
            self.ai_provider = AI_PROVIDER_GEMINI
        self.ollama_model = (
            self.ollama_model_entry.get().strip() or DEFAULT_OLLAMA_MODEL
        )
        selected_avatar = self._avatar_mode_var.get()
        self.avatar_mode = (selected_avatar if selected_avatar in AVATAR_MODES
                            else DEFAULT_AVATAR_MODE)
        save_config(self.api_key, self.assistant_name, self.allowed_apps,
                    self.ai_provider, self.ollama_model, self.avatar_mode)
        self._configure_ai()
        self._switch_avatar(self.avatar_mode)
        self.title_label.config(text=self.assistant_name)
        self.root.title(f"{self.assistant_name} — Voice Assistant")
        self._set_status("Settings saved.")
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.destroy()
        self._settings_win = None

    # ----- Allowed Apps -------------------------------------------------------

    def _refresh_apps_list(self) -> None:
        if not hasattr(self, "apps_list") or not self.apps_list.winfo_exists():
            return
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
            self._set_status("Enter both an app name and a file path.")
            return
        self.allowed_apps[name] = path
        save_config(self.api_key, self.assistant_name, self.allowed_apps,
                    self.ai_provider, self.ollama_model, self.avatar_mode)
        self._refresh_apps_list()
        self._set_status(f"Added '{name}' to Allowed Apps.")

    def _remove_app(self) -> None:
        selection = self.apps_list.curselection()
        if not selection:
            self._set_status("Select an app to remove.")
            return
        name = self.apps_list.get(selection[0])
        self.allowed_apps.pop(name, None)
        save_config(self.api_key, self.assistant_name, self.allowed_apps,
                    self.ai_provider, self.ollama_model, self.avatar_mode)
        self._refresh_apps_list()
        self._set_status(f"Removed '{name}' from Allowed Apps.")

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
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((6, 6, 58, 58), fill=(250, 250, 250, 255))
        draw.ellipse((20, 20, 44, 44), fill=(10, 10, 10, 255))
        draw.ellipse((28, 28, 36, 36), fill=(250, 250, 250, 255))
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
        self._set_status("Minimized to tray.")

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
        user = self.memory.get("user_name")
        hello = f"Hello, {user}!" if user else "Hello!"
        greeting = (
            f"{hello} I'm {self.assistant_name}, your voice assistant. "
            f"Say 'Hey {self.assistant_name}' or '{self.assistant_name}' to talk, "
            "ask me anything, or say 'search for' to look something up. "
            "Say 'help' to hear what I can do."
        )
        self._append_message("assistant", greeting)
        threading.Thread(target=self._speak, args=(greeting,), daemon=True).start()

    def _speak(self, text: str) -> None:
        """Speak a line at the selected speed; cancel via the red Mic button."""
        cancel = threading.Event()
        self._speech_cancel = cancel
        self._speaking_event.set()
        self._set_status("Speaking...")
        self._set_mic_speaking(True)
        with self.speech_lock:
            speak(text, self._speed, cancel)
        self._speaking_event.clear()
        self._set_mic_speaking(False)
        self._set_idle_status()

    def _mic_thread(self) -> None:
        """Open the ONE shared mic stream and feed _mic_queue for the app's life.

        Windows drivers allow only a single sd.InputStream, so this is the
        only place in the whole program where a stream is opened. It tries
        common device configs and reopens the stream if it ever stops.
        """
        while self._running:
            try:
                try:
                    device_rate = int(
                        sd.query_devices(kind="input")["default_samplerate"])
                except Exception:  # noqa: BLE001 - no device info available
                    device_rate = SAMPLE_RATE
                stream = None
                last_error = None
                for rate, channels in ((SAMPLE_RATE, 1), (SAMPLE_RATE, 2),
                                       (device_rate, 1), (device_rate, 2)):
                    try:
                        stream = sd.InputStream(
                            samplerate=rate,
                            channels=channels,
                            dtype="float32",
                            blocksize=max(int(rate * CHUNK_SECONDS), 128),
                            callback=self._mic_callback,
                        )
                        self._mic_rate = rate
                        break
                    except Exception as error:  # noqa: BLE001 - try next
                        last_error = error
                if stream is None:
                    raise RuntimeError(
                        f"Could not open the microphone: {last_error}")
                stream.start()
                try:
                    while self._running and stream.active:
                        time.sleep(0.1)
                finally:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:  # noqa: BLE001 - already closed
                        pass
                if self._running:
                    print("[mic] stream stopped - reopening")
                    time.sleep(0.5)
            except Exception as error:  # noqa: BLE001 - must never die
                print(f"[mic] {error}")
                time.sleep(1.0)

    def _mic_callback(self, indata, frames, time_info, status) -> None:
        """The shared stream's callback: every chunk goes into _mic_queue."""
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        self._mic_queue.put(mono.copy())

    def _audio_router_thread(self) -> None:
        """Route chunks from _mic_queue to the wake buffer or question list.

        While _listening_mode == 'wake' chunks fill the rolling wake buffer;
        while it is 'question' they are captured for the question collector.
        This thread also runs the wake-word check every 2 seconds.
        """
        while self._running:
            try:
                try:
                    chunk = self._mic_queue.get(timeout=CHUNK_SECONDS)
                except queue.Empty:
                    chunk = None
                if chunk is not None:
                    if self._listening_mode == "wake":
                        with self._mic_lock:
                            self._wake_buffer.append(chunk)
                    else:
                        with self._chunks_lock:
                            self._question_chunks.append(chunk)
                now = time.monotonic()
                if (self._listening_mode == "wake"
                        and now - self._wake_last_check >= WAKE_CHECK_SECONDS):
                    self._wake_last_check = now
                    if not (self._busy_event.is_set()
                            or self._speaking_event.is_set()):
                        self._check_wake_buffer()
            except Exception as error:  # noqa: BLE001 - must never die
                print(f"[audio router] {error}")
                time.sleep(0.5)

    def _switch_listening(self, mode: str) -> None:
        """Switch the shared stream's routing between 'wake' and 'question'."""
        if mode == self._listening_mode:
            return
        if mode == "question":
            with self._chunks_lock:
                self._question_chunks.clear()  # capture starts fresh
        self._listening_mode = mode

    def _check_wake_buffer(self) -> None:
        """Transcribe the rolling wake buffer and activate on the name."""
        with self._mic_lock:
            chunks = list(self._wake_buffer)
        if not chunks:
            return
        audio_np = np.concatenate(chunks, axis=0)
        # Lightweight gate: only spend a transcription on sound.
        rms = (float(np.sqrt(np.mean(audio_np ** 2)))
               if audio_np.size else 0.0)
        if rms < SILENCE_THRESHOLD:
            return
        pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        text = transcribe(sr.AudioData(pcm, self._mic_rate, 2))
        if not text:
            self._wake_unclear += 1
            if self._wake_unclear >= 5:
                self._wake_unclear = 0
                self._set_status(
                    "Wake listener is running but can't understand "
                    "audio — check the mic and internet connection."
                )
            return
        self._wake_unclear = 0
        if not re.search(rf"\b{re.escape(self.assistant_name)}\b",
                         text, re.IGNORECASE):
            return
        # Wake phrase confirmed: switch to question mode, beep, then collect.
        self.root.after(0, self._restore_window)
        question = extract_wake_question(text, self.assistant_name) or ""
        self._switch_listening("question")
        play_tone(880, 0.12)
        threading.Thread(target=self._listen_worker,
                         args=(question,), daemon=True).start()

    def _collect_question(self) -> sr.AudioData | None:
        """Collect one question from _question_chunks with silence detection.

        The router fills the list while _listening_mode == 'question'; this
        waits for speech and stops 0.8 s after the speaker goes quiet.
        Returns None when no speech was heard.
        """
        chunks: list[np.ndarray] = []
        heard_speech = False
        silent_chunks = 0
        taken = 0
        start = time.monotonic()
        while True:
            time.sleep(CHUNK_SECONDS)
            with self._chunks_lock:
                new = self._question_chunks[taken:]
                taken = len(self._question_chunks)
            for chunk in new:
                chunks.append(chunk)
                rms = (float(np.sqrt(np.mean(chunk ** 2)))
                       if chunk.size else 0.0)
                if rms >= SILENCE_THRESHOLD:
                    heard_speech = True
                    silent_chunks = 0
                else:
                    silent_chunks += 1
            elapsed = time.monotonic() - start
            if not heard_speech and elapsed >= WAIT_FOR_SPEECH_SECONDS:
                return None
            if (heard_speech and
                    silent_chunks * CHUNK_SECONDS >= SILENCE_SECONDS):
                break
            if elapsed >= MAX_SECONDS:
                break
        if not heard_speech or not chunks:
            return None
        audio = np.concatenate(chunks, axis=0)
        mono = audio.mean(axis=1) if audio.ndim > 1 else audio
        pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return sr.AudioData(pcm, self._mic_rate, 2)

    def _listen_worker(self, question: str | None = None) -> None:
        """Collect one question from the shared stream and answer it."""
        self._busy_event.set()
        self._set_busy(True)
        if not question:
            self._set_status("Listening...")
            self._switch_listening("question")
            try:
                with self._audio_lock:
                    audio = self._collect_question()
            except Exception as error:  # noqa: BLE001 - fall back to typing
                self._switch_listening("wake")
                self._append_message("system", f"Microphone failed: {error}. "
                                               "Type your message instead.")
                self._set_status("Mic failed — type instead")
                self._busy_event.clear()
                self._set_busy(False)
                return
            self._switch_listening("wake")  # capture done - back to wake
            if audio is None:
                self._append_message("system",
                                     "I didn't hear anything. Try again.")
                self._set_idle_status()
                self._busy_event.clear()
                self._set_busy(False)
                return
            question = transcribe(audio)
            if not question:
                self._append_message("system", "Sorry, I couldn't understand that.")
                self._set_idle_status()
                self._busy_event.clear()
                self._set_busy(False)
                return
        else:
            self._switch_listening("wake")  # inline question - nothing to collect
        self._append_message("user", question)
        self._ask_worker(question)

    def _apply_memory_command(self, action: str, payload: str) -> str:
        """Apply a parsed memory command and reply about what changed."""
        if action == "set_name":
            name = payload.strip()
            if not name:
                return "I didn't catch a name. Say 'my name is' followed by your name."
            self.memory["user_name"] = name
            save_memory(self.memory)
            self._configure_ai()
            return f"Got it — I'll call you {name}."
        if action == "remember":
            fact = payload.strip()
            if not fact:
                return "What should I remember? Say 'remember that' and then the fact."
            if fact.lower() in {f.lower() for f in self.memory["facts"]}:
                return "I already remember that."
            self.memory["facts"].append(fact)
            save_memory(self.memory)
            self._configure_ai()
            return f"Okay, I'll remember that {fact}."
        if action == "forget":
            before = len(self.memory["facts"])
            self.memory["facts"] = [
                f for f in self.memory["facts"]
                if payload.strip().lower() not in f.lower()
            ]
            save_memory(self.memory)
            self._configure_ai()
            if len(self.memory["facts"]) < before:
                return "Done — I forgot that."
            return f"I don't have a memory matching '{payload}'."
        if action == "forget_name":
            self.memory["user_name"] = ""
            save_memory(self.memory)
            self._configure_ai()
            return "Okay, I forgot your name."
        if action == "forget_all":
            self.memory = {"user_name": "", "facts": []}
            save_memory(self.memory)
            self._configure_ai()
            return "Okay, I forgot everything."
        if action == "recall_name":
            if self.memory.get("user_name"):
                return f"Your name is {self.memory['user_name']}."
            return ("You haven't told me your name yet. "
                    "Say 'my name is' followed by your name and I'll remember it.")
        if action == "recall_all":
            parts = []
            if self.memory.get("user_name"):
                parts.append(f"your name is {self.memory['user_name']}")
            if self.memory.get("facts"):
                parts.append("I also remember: " + "; ".join(self.memory["facts"]))
            if not parts:
                return ("I don't have any memories yet. "
                        "Say 'remember that' followed by anything and I'll keep it.")
            return "I remember that " + " and ".join(parts) + "."
        return "I couldn't apply that memory command."

    def _local_command(self, question: str) -> str | None:
        """Handle built-in commands locally; return None to defer to the AI."""
        memory_cmd = parse_memory_command(question)
        if memory_cmd is not None:
            return self._apply_memory_command(*memory_cmd)
        q = question.lower().strip().strip(" .!?")
        if q in ("stop", "stop speaking", "be quiet", "quiet"):
            self._stop_speaking()
            return "Okay, stopping."
        if q in ("clear chat", "clear the chat", "reset conversation",
                 "start over", "new chat"):
            self.root.after(0, self._clear_chat)
            return "Chat cleared. What would you like to talk about?"
        if q in ("what time is it", "tell me the time", "the time please"):
            return f"It's {time.strftime('%I:%M %p').lstrip('0')}."
        if q in ("what's the date", "what is the date", "what day is it",
                 "what's today", "what is today", "today's date"):
            return f"Today is {time.strftime('%A, %B %d, %Y')}."
        if q in ("help", "what can you do", "your features", "features"):
            return (
                "You can ask me anything, say 'search for' to look something up, "
                "'open' plus an app name to launch a program, 'what time is it', "
                "'clear chat', 'repeat that', or 'stop' to silence me. "
                "Tell me 'my name is' or 'remember that' and I'll keep it in "
                "memory — ask 'what do you remember' to hear it back. "
                f"Just say 'Hey {self.assistant_name}' to start."
            )
        if q in ("repeat", "repeat that", "say that again", "come again"):
            return self._last_answer or "I haven't said anything yet."
        if q in ("goodbye", "exit app", "quit app", "bye"):
            self._pending_quit = True
            return "Goodbye!"
        return None

    def _ask_worker(self, question: str) -> None:
        self._busy_event.set()
        self._set_busy(True)

        # Built-in commands: instant, no AI round-trip needed.
        local_answer = self._local_command(question)
        if local_answer is not None:
            self._last_answer = local_answer
            self._append_message("assistant", local_answer)
            self._speak(local_answer)
            if self._pending_quit:
                self._pending_quit = False
                self._quit_app()
                return
            self._busy_event.clear()
            self._set_busy(False)
            return

        # "open [app name]" -> launch from the Allowed Apps list.
        app_name = extract_open_app(question)
        if app_name is not None:
            answer = self._open_app(app_name)
            self._append_message("assistant", answer)
            self._speak(answer)
            self._busy_event.clear()
            self._set_busy(False)
            return

        # "search for ..." -> DuckDuckGo results as AI context.
        prompt = question
        query = extract_search_query(question)
        if query:
            self._set_status("Searching...")
            results = duckduckgo_search(query)
            prompt = (
                f"The user asked me to search the web for: {query}\n"
                f"DuckDuckGo results:\n{results}\n\n"
                "Based on those results, reply to the user's message below. "
                "Be concise and mention sources when useful.\n\n"
                f"User message: {question}"
            )

        self._show_typing()  # animated dots while the AI thinks
        self._set_status("Thinking...")
        if self.ai_provider == AI_PROVIDER_OLLAMA:
            try:
                answer = self._ollama_reply(prompt)
            except Exception as error:  # noqa: BLE001 - fall back to Gemini
                detail = " ".join(str(error).split()) or type(error).__name__
                if len(detail) > 180:
                    detail = detail[:177] + "..."
                if self.chat is None:
                    gemini_problem = (self._gemini_error or
                                      "Gemini API is unavailable.")
                    if len(gemini_problem) > 160:
                        gemini_problem = gemini_problem[:157] + "..."
                    notice = (
                        f"Local AI at {OLLAMA_HOST} is unavailable ({detail}); "
                        f"Gemini fallback is unavailable ({gemini_problem})."
                    )
                    self._append_message("system", notice)
                    if self._gemini_error == "Gemini API key is not configured.":
                        answer = (
                            "I couldn't reach Local AI, and Gemini fallback isn't "
                            "configured. Start Ollama or add a Gemini API key in Settings."
                        )
                    else:
                        answer = (
                            "I couldn't reach Local AI or initialize Gemini. "
                            "Check that Ollama is running and verify the Gemini settings."
                        )
                else:
                    notice = (
                        f"Local AI at {OLLAMA_HOST} is unavailable ({detail}); "
                        "falling back to Gemini API."
                    )
                    self._append_message("system", notice)
                    try:
                        answer = self._gemini_reply(prompt)
                        self._remember_ollama_turn(prompt, answer)
                    except Exception as fallback_error:  # noqa: BLE001
                        answer = (
                            "Sorry, Local AI failed and Gemini fallback also "
                            f"failed: {fallback_error}"
                        )
        elif self.chat is None:
            answer = "Set your Gemini API key in the Settings panel to start chatting."
        else:
            try:
                answer = self._gemini_reply(prompt)
            except Exception as error:  # noqa: BLE001 - keep the app alive
                answer = f"Sorry, I ran into a problem: {error}"
        self._hide_typing()
        self._last_answer = answer
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
