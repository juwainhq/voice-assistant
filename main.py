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

import json
import math
import os
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

def _open_input_stream(callback):
    """Open a mic stream, trying common configs until one is accepted.

    Some devices reject 16 kHz or mono. Failing silently on those machines
    made it look like the assistant "wasn't listening", so try a few configs.
    """
    try:
        device_rate = int(sd.query_devices(kind="input")["default_samplerate"])
    except Exception:  # noqa: BLE001 - no device info available
        device_rate = SAMPLE_RATE
    last_error = None
    for rate, channels in ((SAMPLE_RATE, 1), (SAMPLE_RATE, 2),
                           (device_rate, 1), (device_rate, 2)):
        try:
            stream = sd.InputStream(
                samplerate=rate,
                channels=channels,
                dtype="float32",
                blocksize=max(int(rate * CHUNK_SECONDS), 128),
                callback=callback,
            )
            return stream, rate
        except Exception as error:  # noqa: BLE001 - try the next config
            last_error = error
    raise RuntimeError(f"Could not open the microphone: {last_error}")


def record_question() -> sr.AudioData | None:
    """Record one spoken question with sounddevice and return it as AudioData.

    Recording starts immediately, stops when the speaker goes quiet, and
    returns None when no speech was heard.
    """
    chunks: list[np.ndarray] = []
    state = {"heard_speech": False, "silent_chunks": 0}

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
        if rms >= SILENCE_THRESHOLD:
            state["heard_speech"] = True
            state["silent_chunks"] = 0
        else:
            state["silent_chunks"] += 1

    stream, rate = _open_input_stream(callback)
    start = time.monotonic()
    with stream:
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

    # Convert float32 samples to 16-bit mono PCM for SpeechRecognition.
    audio = np.concatenate(chunks, axis=0)
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return sr.AudioData(pcm, rate, 2)


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
        try:
            tts = gTTS(text, slow=(speed == "slow"), timeout=10)
        except TypeError:  # older gTTS without a timeout parameter
            tts = gTTS(text, slow=(speed == "slow"))
        tts.save(mp3_path)
        rate = FAST_RATE if speed == "fast" else 1.0
        _play_mp3(mp3_path, rate, cancel)
    except Exception:  # noqa: BLE001 - speech output is optional
        pass
    finally:
        try:
            os.remove(mp3_path)  # delete the temp file after playing
        except OSError:
            pass


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
    if re.match(r"^(?:what(?:'s| is| s) my name|who am i|do you know my name)\??$",
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
    """An animated bubbly face that shows what the assistant is feeling.

    Emotions: idle, happy, listening, thinking, talking, sad.
    Every frame it gently bobs, blinks at random intervals, glances around
    while thinking, and moves its mouth while talking. Drawn with the same
    monochrome palette as the rest of the UI.
    """

    FRAME_MS = 90  # animation frame period

    def __init__(self, parent, size: int = 96, **kwargs):
        super().__init__(parent, width=size, height=size, bg=BG,
                         highlightthickness=0, borderwidth=0, **kwargs)
        self.size = size
        self.emotion = "idle"
        self._t = 0.0
        self._blink_in = 2.4
        self._blink_left = 0.0

        self.head = self.create_oval(0, 0, 0, 0, fill=WHITE,
                                     outline=ENTRY_BORDER, width=2)
        self.blush_l = self.create_oval(0, 0, 0, 0, outline=MUTED)
        self.blush_r = self.create_oval(0, 0, 0, 0, outline=MUTED)
        self.eye_l = self.create_oval(0, 0, 0, 0, fill=BLACK, outline="")
        self.eye_r = self.create_oval(0, 0, 0, 0, fill=BLACK, outline="")
        self.eye_l_happy = self.create_arc(0, 0, 0, 0, style=tk.ARC,
                                           outline=BLACK, width=3)
        self.eye_r_happy = self.create_arc(0, 0, 0, 0, style=tk.ARC,
                                           outline=BLACK, width=3)
        self.shine_l = self.create_oval(0, 0, 0, 0, fill=WHITE, outline="")
        self.shine_r = self.create_oval(0, 0, 0, 0, fill=WHITE, outline="")
        self.brow_l = self.create_line(0, 0, 0, 0, fill=BLACK, width=3)
        self.brow_r = self.create_line(0, 0, 0, 0, fill=BLACK, width=3)
        self.mouth_arc = self.create_arc(0, 0, 0, 0, style=tk.ARC,
                                         outline=BLACK, width=3)
        self.mouth_open = self.create_oval(0, 0, 0, 0, fill=BLACK, outline="")

        self.after(self.FRAME_MS, self._tick)

    def set_emotion(self, emotion: str) -> None:
        """Switch expression: idle, happy, listening, thinking, talking, sad."""
        if emotion in ("idle", "happy", "listening", "thinking", "talking", "sad"):
            self.emotion = emotion

    def _tick(self) -> None:
        dt = self.FRAME_MS / 1000.0
        self._t += dt
        self._blink_in -= dt
        if self._blink_left > 0:
            self._blink_left -= dt
        elif self._blink_in <= 0:
            self._blink_left = 0.13
            self._blink_in = 2.2 + random.random() * 2.8
        self._layout()
        self.after(self.FRAME_MS, self._tick)

    def _layout(self) -> None:
        s = self.size
        t = self._t
        emo = self.emotion

        # Head, gently bobbing (faster and bouncier while talking)
        speed = 3.2 if emo == "talking" else 1.7
        amp = 2.6 if emo in ("talking", "happy", "listening") else 1.1
        cx, cy = s / 2, s / 2 + math.sin(t * speed) * amp
        rx, ry = s * 0.42, s * 0.40
        self.coords(self.head, cx - rx, cy - ry, cx + rx, cy + ry)

        # Blush
        br = s * 0.065
        for item, side in ((self.blush_l, -1), (self.blush_r, 1)):
            bx = cx + side * rx * 0.62
            by = cy + ry * 0.20
            self.coords(item, bx - br, by - br, bx + br, by + br)

        # Eyes (happy/talking use closed smile arcs; blinking squashes them)
        blinking = self._blink_left > 0 and emo not in ("happy", "talking")
        happy_eyes = emo in ("happy", "talking")
        eye_dx = rx * 0.40
        eye_cy = cy - ry * 0.12
        ew = s * (0.105 if emo == "listening" else 0.078)
        eh = s * (0.115 if emo == "listening" else 0.105)
        if emo == "thinking":
            pupil_dx, pupil_dy = math.sin(t * 1.3) * s * 0.022, -s * 0.022
        elif emo == "sad":
            pupil_dx, pupil_dy = 0.0, s * 0.02
        elif emo == "listening":
            pupil_dx, pupil_dy = 0.0, 0.0
        else:
            pupil_dx, pupil_dy = math.sin(t * 0.7) * s * 0.008, 0.0
        if blinking:
            eh = 2.0

        for side, eye, happy, shine in (
            (-1, self.eye_l, self.eye_l_happy, self.shine_l),
            (1, self.eye_r, self.eye_r_happy, self.shine_r),
        ):
            ex = cx + side * eye_dx
            if happy_eyes:
                self.itemconfig(eye, state="hidden")
                self.itemconfig(shine, state="hidden")
                self.itemconfig(happy, state="normal", start=180, extent=180)
                self.coords(happy, ex - ew * 1.4, eye_cy - eh * 1.1,
                            ex + ew * 1.4, eye_cy + eh * 1.1)
            else:
                self.itemconfig(happy, state="hidden")
                self.itemconfig(eye, state="normal")
                self.itemconfig(shine, state="normal")
                ecx, ecy = ex + pupil_dx, eye_cy + pupil_dy
                self.coords(eye, ecx - ew, ecy - eh, ecx + ew, ecy + eh)
                sw = ew * 0.30
                self.coords(shine, ecx - ew * 0.35 - sw, ecy - eh * 0.45 - sw,
                            ecx - ew * 0.35 + sw, ecy - eh * 0.45 + sw)

        # Eyebrows carry most of the emotion
        brow_w = ew * 2.4
        brow_y = eye_cy - eh * 2.15
        lift = s * 0.030
        for side, brow in ((-1, self.brow_l), (1, self.brow_r)):
            bx = cx + side * eye_dx
            x1, x2 = bx - brow_w / 2, bx + brow_w / 2
            y1 = y2 = brow_y
            if emo == "listening":
                y1 = y2 = brow_y - lift
            elif emo == "thinking":
                if side == -1:
                    y1, y2 = brow_y - lift, brow_y - lift * 0.15
                else:
                    y1, y2 = brow_y - lift * 0.15, brow_y - lift
            elif emo == "sad":
                if side == -1:  # worried: inner ends raised
                    y1, y2 = brow_y + lift * 0.45, brow_y - lift * 0.55
                else:
                    y1, y2 = brow_y - lift * 0.55, brow_y + lift * 0.45
            elif emo == "happy":
                y1 = y2 = brow_y - lift * 0.5
            self.coords(brow, x1, y1, x2, y2)

        # Mouth: an oval that opens/closes while talking, an arc otherwise
        mouth_cy = cy + ry * 0.32
        if emo == "talking":
            open_amt = 0.35 + 0.65 * abs(math.sin(t * 11.0))
            self.itemconfig(self.mouth_arc, state="hidden")
            self.itemconfig(self.mouth_open, state="normal")
            mw = rx * 0.26
            mh = s * 0.028 + open_amt * s * 0.085
            self.coords(self.mouth_open, cx - mw, mouth_cy - mh,
                        cx + mw, mouth_cy + mh)
        else:
            self.itemconfig(self.mouth_open, state="hidden")
            self.itemconfig(self.mouth_arc, state="normal")
            if emo in ("happy", "listening"):
                mw, mh = rx * 0.62, ry * 0.42
                self.itemconfig(self.mouth_arc, start=0, extent=180)  # big smile
            elif emo == "sad":
                mw, mh = rx * 0.55, ry * 0.38
                self.itemconfig(self.mouth_arc, start=180, extent=180)  # frown
            else:  # idle / thinking: small gentle smile
                mw, mh = rx * 0.50, ry * 0.32
                self.itemconfig(self.mouth_arc, start=20, extent=140)
            self.coords(self.mouth_arc, cx - mw, mouth_cy - mh,
                        cx + mw, mouth_cy + mh)


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
        self.memory = load_memory()  # user name + remembered facts
        self.client = None
        self.chat = None
        self.tray_icon = None

        self.speech_lock = threading.Lock()
        self._audio_lock = threading.Lock()  # one mic stream at a time
        self._busy_event = threading.Event()  # set while listening/thinking
        self._speaking_event = threading.Event()  # set while TTS plays
        self._speech_cancel = threading.Event()  # set to stop playback
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
        header.pack(fill=tk.X, padx=20, pady=(10, 6))
        self.face = BubblyFace(header, size=92)
        self.face.pack(side=tk.LEFT, padx=(0, 14))
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
        self.clear_button = tk.Button(
            header, text="Clear chat", font=FONT_BOLD, command=self._clear_chat,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
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
        self.chat_view.tag_configure("user_name", foreground=TEXT, font=FONT_BOLD)
        self.chat_view.tag_configure("assistant_name", foreground=TEXT, font=FONT_BOLD)
        self.chat_view.tag_configure("system_name", foreground=MUTED, font=FONT_BOLD)
        self.chat_view.tag_configure("user_text", foreground=TEXT, lmargin1=10, lmargin2=10)
        self.chat_view.tag_configure("assistant_text", foreground=SOFT,
                                     lmargin1=10, lmargin2=10)
        self.chat_view.tag_configure("system_text", foreground=MUTED,
                                     lmargin1=10, lmargin2=10)
        self.chat_view.tag_configure("typing_line", foreground=MUTED,
                                     font=("Segoe UI", 10, "italic"))
        self.chat_view.tag_configure("copy_link", foreground=WHITE,
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
            bg=WHITE, fg=BLACK, activebackground=WHITE,
            activeforeground=BLACK, relief=tk.FLAT, padx=18, pady=8,
            cursor="hand2", borderwidth=0,
        )
        self.mic_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_button = tk.Button(
            input_row, text="Stop", font=FONT_BOLD, command=self._stop_speaking,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, padx=14, pady=8, cursor="hand2", borderwidth=0,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        self.entry = tk.Entry(
            input_row, bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT, font=FONT,
            relief=tk.FLAT, highlightthickness=1, highlightbackground=ENTRY_BORDER,
            highlightcolor=WHITE,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.entry.bind("<Return>", lambda _event: self.on_send())
        self.send_button = tk.Button(
            input_row, text="Send", font=FONT_BOLD, command=self.on_send,
            bg=GRAY, fg=TEXT, activebackground=WHITE,
            activeforeground=BLACK, relief=tk.FLAT, padx=18, pady=8,
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
                highlightbackground=ENTRY_BORDER, highlightcolor=WHITE,
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
            troughcolor=ENTRY_BG, activebackground=WHITE, sliderrelief=tk.FLAT,
            command=self._on_speed_change,
        )
        self.speed_scale.set(1)
        self.speed_scale.pack(fill=tk.X, padx=14, pady=(2, 0))
        speed_labels = tk.Frame(inner, bg=PANEL)
        speed_labels.pack(fill=tk.X, padx=18)
        for text, side in (("Slow", tk.LEFT), ("Normal", tk.LEFT), ("Fast", tk.RIGHT)):
            tk.Label(speed_labels, text=text, font=("Segoe UI", 8),
                     bg=PANEL, fg=MUTED).pack(side=side, expand=True)

        # Allowed Apps
        _label("ALLOWED APPS")
        self.apps_list = tk.Listbox(
            inner, bg=ENTRY_BG, fg=TEXT, font=("Segoe UI", 9), relief=tk.FLAT,
            highlightthickness=1, highlightbackground=ENTRY_BORDER,
            selectbackground=WHITE, selectforeground=BLACK, height=5,
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
            bg=GRAY, fg=TEXT, activebackground=WHITE,
            activeforeground=BLACK, relief=tk.FLAT, pady=6, cursor="hand2",
            borderwidth=0,
        )
        self.add_app_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.remove_app_button = tk.Button(
            apps_buttons, text="Remove", font=FONT_BOLD, command=self._remove_app,
            bg=GRAY, fg=TEXT, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, pady=6, cursor="hand2", borderwidth=0,
        )
        self.remove_app_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        save_button = tk.Button(
            inner, text="Save settings", font=FONT_BOLD, command=self.save_settings,
            bg=GRAY, fg=TEXT, activebackground=WHITE,
            activeforeground=BLACK, relief=tk.FLAT, pady=8, cursor="hand2",
            borderwidth=0,
        )
        save_button.pack(fill=tk.X, padx=18, pady=18)

    # ----- Gemini -----------------------------------------------------------

    def _configure_gemini(self) -> None:
        if not self.api_key or self.api_key == PLACEHOLDER_KEY:
            self.client = None
            self.chat = None
            self._set_status("Set your Gemini API key in Settings to start chatting.")
            return
        self.client = genai.Client(api_key=self.api_key)
        history = self.chat.get_history() if self.chat is not None else []
        memory_notes = []
        if self.memory.get("user_name"):
            memory_notes.append(
                f"- The user's name is {self.memory['user_name']}.")
        for fact in self.memory.get("facts", []):
            memory_notes.append(f"- Remembered about the user: {fact}.")
        memory_block = ""
        if memory_notes:
            memory_block = (
                "\n\nThings you remember about the user "
                "(use them naturally in conversation; never recite this list):\n"
                + "\n".join(memory_notes)
            )
        self.chat = self.client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=(
                    f"You are {self.assistant_name}, a friendly voice assistant. "
                    "Keep answers clear and conversational." + memory_block
                ),
            ),
            history=list(history),
        )
        self._set_idle_status()

    # ----- Thread-safe UI updates -------------------------------------------

    def _set_status(self, text: str) -> None:
        def apply() -> None:
            self.status_label.config(text=text, fg=MUTED)
            self.face.set_emotion(infer_emotion(text))

        self.root.after(0, apply)

    def _set_idle_status(self) -> None:
        """Status shown whenever the passive wake listener is armed."""
        self._set_status(f'Ready — say "Hey {self.assistant_name}" to talk')

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
        self._set_status("Copied to clipboard.")

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
        self._set_idle_status()

    def _clear_chat(self) -> None:
        self.chat_view.configure(state=tk.NORMAL)
        self.chat_view.delete("1.0", tk.END)
        self.chat_view.configure(state=tk.DISABLED)
        self._copy_texts.clear()
        self._typing_start = "1.0"
        self.chat = None
        self._configure_gemini()  # fresh chat with empty history
        self._set_status("Chat cleared — memory reset.")

    def save_settings(self) -> None:
        self.assistant_name = self.name_entry.get().strip() or DEFAULT_ASSISTANT_NAME
        self.api_key = self.key_entry.get().strip()
        save_config(self.api_key, self.assistant_name, self.allowed_apps)
        self._configure_gemini()
        self.title_label.config(text=self.assistant_name)
        self.root.title(f"{self.assistant_name} — Voice Assistant")
        self._set_status("Settings saved.")

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
            self._set_status("Enter both an app name and a file path.")
            return
        self.allowed_apps[name] = path
        save_config(self.api_key, self.assistant_name, self.allowed_apps)
        self._refresh_apps_list()
        self._set_status(f"Added '{name}' to Allowed Apps.")

    def _remove_app(self) -> None:
        selection = self.apps_list.curselection()
        if not selection:
            self._set_status("Select an app to remove.")
            return
        name = self.apps_list.get(selection[0])
        self.allowed_apps.pop(name, None)
        save_config(self.api_key, self.assistant_name, self.allowed_apps)
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
        """Speak a line at the selected speed; cancellable via the Stop button."""
        cancel = threading.Event()
        self._speech_cancel = cancel
        self._speaking_event.set()
        self._set_status("Speaking...")
        with self.speech_lock:
            speak(text, self._speed, cancel)
        self._speaking_event.clear()
        self._set_idle_status()

    def _wake_word_worker(self) -> None:
        """Passively listen for the wake phrase and activate on a match.

        Wake phrases: "Hey Nova", "Nova", "Nova I have a question", ...
        The loop must never die — every error is caught and retried.
        """
        unclear = 0
        while self._running:
            try:
                if self._busy_event.is_set() or self._speaking_event.is_set():
                    time.sleep(0.2)
                    continue
                with self._audio_lock:
                    audio = record_question()
                if audio is None or self._busy_event.is_set() or self._speaking_event.is_set():
                    continue
                text = transcribe(audio)
                if not text:
                    unclear += 1
                    if unclear >= 5:
                        unclear = 0
                        self._set_status(
                            "Wake listener is running but can't understand audio — "
                            "check the mic and internet connection."
                        )
                    continue
                unclear = 0
                question = extract_wake_question(text, self.assistant_name)
                if question is None:
                    continue
                # Wake phrase confirmed: show the window, beep, then act.
                self.root.after(0, self._restore_window)
                play_tone(880, 0.12)
                if question:
                    self._append_message("user", question)
                    self._ask_worker(question)
                else:
                    self._listen_worker()  # just the wake phrase -> hear the question
            except Exception as error:  # noqa: BLE001 - the listener must never die
                print(f"[wake listener] {error}")
                time.sleep(1.0)

    def _listen_worker(self) -> None:
        self._busy_event.set()
        self._set_busy(True)
        self._set_status("Listening...")
        with self._audio_lock:
            try:
                audio = record_question()
            except Exception as error:  # noqa: BLE001 - mic problems fall back to typing
                self._append_message("system", f"Microphone failed: {error}. "
                                               "Type your message instead.")
                self._set_status("Mic failed — type instead")
                self._busy_event.clear()
                self._set_busy(False)
                return
        if audio is None:
            self._append_message("system", "I didn't hear anything. Try again.")
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
            self._configure_gemini()
            return f"Got it — I'll call you {name}."
        if action == "remember":
            fact = payload.strip()
            if not fact:
                return "What should I remember? Say 'remember that' and then the fact."
            if fact.lower() in {f.lower() for f in self.memory["facts"]}:
                return "I already remember that."
            self.memory["facts"].append(fact)
            save_memory(self.memory)
            self._configure_gemini()
            return f"Okay, I'll remember that {fact}."
        if action == "forget":
            before = len(self.memory["facts"])
            self.memory["facts"] = [
                f for f in self.memory["facts"]
                if payload.strip().lower() not in f.lower()
            ]
            save_memory(self.memory)
            self._configure_gemini()
            if len(self.memory["facts"]) < before:
                return "Done — I forgot that."
            return f"I don't have a memory matching '{payload}'."
        if action == "forget_name":
            self.memory["user_name"] = ""
            save_memory(self.memory)
            self._configure_gemini()
            return "Okay, I forgot your name."
        if action == "forget_all":
            self.memory = {"user_name": "", "facts": []}
            save_memory(self.memory)
            self._configure_gemini()
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
        """Handle built-in commands locally; return None to defer to Gemini."""
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

        # Built-in commands: instant, no Gemini round-trip needed.
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

        # "search for ..." -> DuckDuckGo results as Gemini context.
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

        self._show_typing()  # animated dots while Gemini thinks
        self._set_status("Thinking...")
        if self.chat is None:
            answer = "Set your Gemini API key in the Settings panel to start chatting."
        else:
            try:
                response = self.chat.send_message(prompt)
                answer = (response.text or "").strip() or "(no response)"
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
