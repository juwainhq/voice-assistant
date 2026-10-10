"""Nova — a voice-first chat app for desktop.

The desktop chat follows the compact conversation/model-picker workflow of
Coucou by Louis Raillé (MIT); it keeps Nova's own mouthless animated portrait,
shared-microphone wake listener, and voice I/O. Chat providers are independent
adapters in chat_providers.py, so adding another compatible backend is simple.
"""

import collections
import importlib
import math
import os
import queue
import random
import re
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import tkinter as tk
from tkinter import ttk

import chat_providers
import config

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

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DEFAULT_ASSISTANT_NAME = "Nova"
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
WAKE_CHECK_SECONDS = 0.5  # how often the rolling wake buffer is transcribed
WAKE_WINDOW_SECONDS = 2.0  # rolling 2-second wake buffer sent to recognition
WAKE_ONLY_TAILS = frozenset({
    "", "i have a question", "i've got a question", "ive got a question",
    "question", "yes", "yeah", "hello", "hi", "hey", "are you there",
    "can you hear me", "wake up", "you there", "it's me", "its me",
})

# Microphone recording settings
SAMPLE_RATE = 16000  # samples per second
CHUNK_SECONDS = 0.1  # how often the mic level is checked
SILENCE_THRESHOLD = 0.005  # RMS level treated as speech vs. silence
WAKE_RMS_THRESHOLD = 0.008  # minimum sound level to bother transcribing
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
# Voice I/O and config helpers
# ----------------------------------------------------------------------------





_RECOGNIZER = sr.Recognizer()


def transcribe(audio: sr.AudioData) -> str | None:
    """Convert recorded audio to text with Google speech recognition.

    Any failure (including missing audioop on new Python versions) is
    swallowed so the listening threads never die silently.
    """
    try:
        return _RECOGNIZER.recognize_google(audio)
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


def infer_emotion(status: str) -> str:
    """Map a status line to the face emotion."""
    s = status.lower()
    if "listening" in s:
        return "listening"
    if "thinking" in s or "searching" in s:
        return "thinking"
    if "speaking" in s:
        return "talking"
    if any(bad in s for bad in ("mic failed", "can't understand", "sorry", "failed")):
        return "sad"
    if any(good in s for good in ("copied", "saved", "cleared", "connected")):
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
        self.ai_provider = chat_providers.normalize_provider(
            getattr(config, "AI_PROVIDER", "google"))
        self.model = chat_providers.model_for(config, self.ai_provider)
        self.chat_history: list[dict[str, str]] = []
        self.avatar_mode = (AVATAR_MODE_IMAGE if USE_IMAGE_AVATAR else str(
            getattr(config, "AVATAR_MODE", DEFAULT_AVATAR_MODE)
        ).lower())
        if self.avatar_mode not in AVATAR_MODES:
            self.avatar_mode = DEFAULT_AVATAR_MODE
        self.assistant_name = str(
            getattr(config, "ASSISTANT_NAME", DEFAULT_ASSISTANT_NAME)
            or DEFAULT_ASSISTANT_NAME
        )
        saved_speed = str(getattr(config, "VOICE_SPEED", "normal")).lower()
        self._speed = saved_speed if saved_speed in ("slow", "normal", "fast") else "normal"
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
        self._wake_buffer = collections.deque(  # rolling 2-second buffer
            maxlen=max(int(WAKE_WINDOW_SECONDS / CHUNK_SECONDS), 2))
        self._question_chunks: list = []  # chunks while capturing a question
        self._mic_lock = threading.Lock()  # guards the rolling wake buffer
        self._chunks_lock = threading.Lock()  # guards _question_chunks
        self._mic_rate = SAMPLE_RATE
        self._wake_last_check = 0.0
        self._wake_unclear = 0
        self._running = True
        self._msg_counter = 0
        self._copy_texts: dict[str, str] = {}
        self._typing_active = False
        self._typing_start = "1.0"
        self._typing_ticks = 0
        self._typing_job = None
        self._settings_win = None
        self._model_picker_win = None
        self._dot_blink_job = None
        self._dot_color = MUTED

        root.title(f"{self.assistant_name} — Chat")
        root.geometry("940x700")
        root.minsize(640, 500)
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
        """Build the compact, provider-first voice chat screen."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "TCombobox", fieldbackground=ENTRY_BG, background=GRAY,
            foreground=TEXT, arrowcolor=TEXT, bordercolor=ENTRY_BORDER,
            lightcolor=ENTRY_BORDER, darkcolor=ENTRY_BORDER,
        )
        style.map(
            "TCombobox", fieldbackground=[("readonly", ENTRY_BG)],
            foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", GRAY)],
            selectforeground=[("readonly", TEXT)],
        )
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill=tk.X, padx=18, pady=(10, 8))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=2)
        header.grid_columnconfigure(2, weight=1)

        self.gear_button = tk.Button(
            header, text="Settings", font=FONT_BOLD, command=self._open_settings,
            bg=BG, fg=MUTED, activebackground=GRAY, activeforeground=TEXT,
            relief=tk.FLAT, padx=10, pady=6, cursor="hand2", borderwidth=0,
        )
        self.gear_button.grid(row=0, column=0, sticky="w")

        self._avatar_header = tk.Frame(header, bg=BG)
        self._avatar_header.grid(row=0, column=1, sticky="n")
        self.face = self._make_avatar_widget(self._avatar_header, self.avatar_mode)
        self.face.pack(anchor="center")
        self.title_label = tk.Label(
            self._avatar_header, text=self.assistant_name,
            font=FONT_TITLE, bg=BG, fg=TEXT,
        )
        self.title_label.pack(anchor="center", pady=(2, 0))
        tk.Label(
            self._avatar_header, text="voice chat", font=("Segoe UI", 8),
            bg=BG, fg=MUTED,
        ).pack(anchor="center")

        status_box = tk.Frame(header, bg=BG)
        status_box.grid(row=0, column=2, sticky="e")
        self.status_dot = tk.Canvas(status_box, width=12, height=12, bg=BG,
                                    highlightthickness=0, borderwidth=0)
        self.status_dot.pack(side=tk.RIGHT, padx=(8, 0))
        self._dot = self.status_dot.create_oval(1, 1, 11, 11,
                                                fill=MUTED, outline="")

        tk.Frame(self.root, bg=ENTRY_BORDER, height=1).pack(fill=tk.X)

        chat_frame = tk.Frame(self.root, bg=CARD)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 4))
        self.chat_view = tk.Text(
            chat_frame, bg=CARD, fg=TEXT, font=FONT, relief=tk.FLAT,
            wrap=tk.WORD, state=tk.DISABLED, padx=22, pady=18, spacing3=8,
            cursor="arrow", borderwidth=0, highlightthickness=0,
        )
        scrollbar = tk.Scrollbar(
            chat_frame, command=self.chat_view.yview, width=8, relief=tk.FLAT,
            bg=ENTRY_BORDER, activebackground=MUTED, troughcolor=CARD,
        )
        self.chat_view.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_view.tag_configure(
            "user_bubble", foreground=TEXT, background=GRAY, justify=tk.RIGHT,
            lmargin1=100, lmargin2=100, rmargin=4, spacing1=8, spacing3=10,
        )
        self.chat_view.tag_configure(
            "assistant_name", foreground=TEXT, font=FONT_BOLD,
            spacing1=8, spacing3=2,
        )
        self.chat_view.tag_configure(
            "assistant_text", foreground="#C8C8C8", justify=tk.LEFT,
            lmargin1=3, lmargin2=3, spacing3=2,
        )
        self.chat_view.tag_configure(
            "system_text", foreground=MUTED, justify=tk.CENTER,
            font=("Segoe UI", 8), spacing1=4, spacing3=8,
        )
        self.chat_view.tag_configure(
            "typing_line", foreground=MUTED, font=("Segoe UI", 9, "italic"),
        )
        self.chat_view.tag_configure(
            "copy_link", foreground=WHITE, font=("Segoe UI", 8, "underline"),
        )
        self.chat_view.tag_configure("md_bold", foreground=TEXT,
                                     font=("Segoe UI", 10, "bold"))
        self.chat_view.tag_configure("md_italic", foreground="#E6E6E6",
                                     font=("Segoe UI", 10, "italic"))
        self.chat_view.tag_configure("md_code", foreground=TEXT,
                                     background=GRAY, font=("Consolas", 9))
        self.chat_view.tag_configure("md_heading", foreground=TEXT,
                                     font=("Segoe UI", 11, "bold"))
        self.chat_view.tag_configure("md_quote", foreground=MUTED,
                                     lmargin1=12, lmargin2=12)

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill=tk.X, padx=16, pady=(4, 12))
        model_row = tk.Frame(bottom, bg=BG)
        model_row.pack(fill=tk.X, pady=(0, 6))
        self.model_button = tk.Button(
            model_row, text="", font=("Segoe UI", 8),
            command=self._open_model_picker,
            bg=BG, fg=MUTED, activebackground=GRAY, activeforeground=TEXT,
            relief=tk.FLAT, padx=8, pady=4, cursor="hand2", borderwidth=0,
        )
        self.model_button.pack(side=tk.LEFT)
        self.new_chat_button = tk.Button(
            model_row, text="New chat", font=("Segoe UI", 8),
            command=self._clear_chat,
            bg=BG, fg=MUTED, activebackground=GRAY, activeforeground=TEXT,
            relief=tk.FLAT, padx=8, pady=4, cursor="hand2", borderwidth=0,
        )
        self.new_chat_button.pack(side=tk.RIGHT)

        composer = tk.Frame(
            bottom, bg=ENTRY_BG, highlightthickness=1,
            highlightbackground=ENTRY_BORDER, highlightcolor=WHITE,
        )
        composer.pack(fill=tk.X)
        self.send_button = tk.Button(
            composer, text="↑", font=("Segoe UI", 13, "bold"),
            command=self.on_send, bg=WHITE, fg=BLACK,
            activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, width=3, cursor="hand2", borderwidth=0,
        )
        self.send_button.pack(side=tk.RIGHT, padx=(4, 5), pady=5, ipady=1)
        self.mic_button = tk.Button(
            composer, text="Mic", font=FONT_BOLD, command=self.on_mic,
            bg=ENTRY_BG, fg=TEXT, activebackground=WHITE,
            activeforeground=BLACK, relief=tk.FLAT, padx=12, pady=7,
            cursor="hand2", borderwidth=0,
        )
        self.mic_button.pack(side=tk.RIGHT, padx=(0, 4), pady=5)
        self.entry = tk.Entry(
            composer, bg=ENTRY_BG, fg=MUTED, insertbackground=TEXT,
            font=FONT, relief=tk.FLAT, borderwidth=0,
        )
        self._input_hint = f"Message {self.assistant_name}…"
        self.entry.insert(0, self._input_hint)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=12)
        self.entry.bind("<Return>", lambda _event: self.on_send())
        self.entry.bind("<FocusIn>", self._clear_input_hint)
        self.entry.bind("<FocusOut>", self._restore_input_hint)

        self.status_label = tk.Label(
            bottom, text=f"Say Hey {self.assistant_name} to start",
            font=("Segoe UI", 8), bg=BG, fg=MUTED, anchor="w",
        )
        self.status_label.pack(fill=tk.X, pady=(6, 0))
        self._update_model_button()

    def _clear_input_hint(self, _event=None) -> None:
        if self.entry.get() == self._input_hint:
            self.entry.delete(0, tk.END)
            self.entry.configure(fg=TEXT)

    def _restore_input_hint(self, _event=None) -> None:
        if not self.entry.get().strip():
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self._input_hint)
            self.entry.configure(fg=MUTED)

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
        self.face.pack(anchor="center")

    def _refresh_shared_settings(self) -> None:
        """Read settings saved by the phone before opening desktop controls."""
        importlib.reload(config)
        self.ai_provider = chat_providers.normalize_provider(
            getattr(config, "AI_PROVIDER", self.ai_provider))
        self.model = chat_providers.model_for(config, self.ai_provider)
        name = str(getattr(config, "ASSISTANT_NAME", self.assistant_name)
                   or DEFAULT_ASSISTANT_NAME)
        if name != self.assistant_name:
            self.assistant_name = name
            self._apply_shared_name()
        avatar = str(getattr(config, "AVATAR_MODE", self.avatar_mode)).lower()
        if avatar in AVATAR_MODES and avatar != self.avatar_mode:
            self._switch_avatar(avatar)
        speed = str(getattr(config, "VOICE_SPEED", self._speed)).lower()
        if speed in ("slow", "normal", "fast"):
            self._speed = speed
        self._update_model_button()

    def _open_settings(self) -> None:
        """Open the small chat/voice preferences sheet."""
        self._refresh_shared_settings()
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.lift()
            self._settings_win.focus_force()
            return
        win = tk.Toplevel(self.root)
        win.title("Chat settings")
        win.configure(bg=BG)
        win.geometry("400x430")
        win.resizable(False, False)
        win.transient(self.root)
        self._settings_win = win
        body = tk.Frame(win, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=22, pady=18)
        tk.Label(body, text="Chat settings", font=FONT_SECTION,
                 bg=BG, fg=TEXT).pack(anchor="w")

        def label(text: str) -> None:
            tk.Label(body, text=text, font=FONT_SMALL,
                     bg=BG, fg=MUTED).pack(anchor="w", pady=(14, 3))

        label("ASSISTANT NAME")
        self.name_entry = tk.Entry(
            body, bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
            font=FONT, relief=tk.FLAT, borderwidth=0,
            highlightthickness=1, highlightbackground=ENTRY_BORDER,
            highlightcolor=WHITE,
        )
        self.name_entry.insert(0, self.assistant_name)
        self.name_entry.pack(fill=tk.X, ipady=7)

        label("AVATAR")
        self._avatar_mode_var = tk.StringVar(value=self.avatar_mode)
        for text, value in (("Animated portrait", AVATAR_MODE_PORTRAIT),
                            ("Classic Grok Bot", AVATAR_MODE_GROKBOT),
                            ("Static image", AVATAR_MODE_IMAGE)):
            tk.Radiobutton(
                body, text=text, value=value, variable=self._avatar_mode_var,
                bg=BG, fg=TEXT, selectcolor=GRAY, activebackground=BG,
                activeforeground=WHITE, highlightthickness=0, borderwidth=0,
                font=FONT,
            ).pack(anchor="w", pady=1)

        label("VOICE SPEED")
        self._speed_var = tk.StringVar(value=self._speed)
        speed_row = tk.Frame(body, bg=BG)
        speed_row.pack(fill=tk.X)
        for text, value in (("Slow", "slow"), ("Normal", "normal"),
                            ("Fast", "fast")):
            tk.Radiobutton(
                speed_row, text=text, value=value, variable=self._speed_var,
                bg=BG, fg=TEXT, selectcolor=GRAY, activebackground=BG,
                activeforeground=WHITE, highlightthickness=0, borderwidth=0,
                font=FONT,
            ).pack(side=tk.LEFT, expand=True, anchor="w")

        tk.Label(
            body, text="Choose a provider and model from the chat bar.",
            font=("Segoe UI", 8), bg=BG, fg=MUTED,
        ).pack(anchor="w", pady=(14, 0))
        tk.Button(
            body, text="Save settings", font=FONT_BOLD, command=self.save_settings,
            bg=WHITE, fg=BLACK, activebackground=WHITE, activeforeground=BLACK,
            relief=tk.FLAT, pady=8, cursor="hand2", borderwidth=0,
        ).pack(fill=tk.X, pady=(20, 0))

    def _open_model_picker(self) -> None:
        """Open the provider/model picker inspired by Coucou's chat picker."""
        self._refresh_shared_settings()
        if self._model_picker_win is not None and self._model_picker_win.winfo_exists():
            self._model_picker_win.lift()
            self._model_picker_win.focus_force()
            return
        win = tk.Toplevel(self.root)
        win.title("Choose a model")
        win.configure(bg=BG)
        win.geometry("460x590")
        win.resizable(False, False)
        win.transient(self.root)
        self._model_picker_win = win
        body = tk.Frame(win, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=18)
        tk.Label(body, text="Choose a model", font=FONT_SECTION,
                 bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(
            body, text="Pick a provider, connect it once, then choose its model.",
            font=("Segoe UI", 8), bg=BG, fg=MUTED,
        ).pack(anchor="w", pady=(3, 12))

        self._picker_provider_var = tk.StringVar(value=self.ai_provider)
        self._picker_model_var = tk.StringVar(value=self.model)
        self._picker_key_cleared = False
        self._picker_status_var = tk.StringVar(value="")
        self._picker_provider_values = list(chat_providers.PROVIDER_IDS)
        self._picker_key_field = None
        self._picker_base_field = None
        self._picker_key_saved = None

        tk.Label(body, text="PROVIDER", font=FONT_SMALL,
                 bg=BG, fg=MUTED).pack(anchor="w", pady=(4, 2))
        provider_combo = ttk.Combobox(
            body, state="readonly", textvariable=self._picker_provider_var,
            values=[f"{provider} · {chat_providers.PROVIDER_LABELS[provider]}"
                    for provider in self._picker_provider_values],
            font=FONT,
        )
        provider_combo.pack(fill=tk.X, ipady=4)
        provider_combo.set(
            f"{self.ai_provider} · {chat_providers.PROVIDER_LABELS[self.ai_provider]}")
        provider_combo.bind("<<ComboboxSelected>>", self._picker_provider_changed)
        self._provider_combo = provider_combo

        tk.Label(body, text="MODEL", font=FONT_SMALL,
                 bg=BG, fg=MUTED).pack(anchor="w", pady=(14, 2))
        self._picker_model_combo = ttk.Combobox(
            body, state="normal", textvariable=self._picker_model_var,
            font=FONT,
        )
        self._picker_model_combo.pack(fill=tk.X, ipady=4)

        self._picker_key_label = tk.Label(body, text="API KEY", font=FONT_SMALL,
                                          bg=BG, fg=MUTED)
        self._picker_key_label.pack(anchor="w", pady=(14, 2))
        self._picker_key_field = tk.Entry(
            body, show="•", bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
            font=FONT, relief=tk.FLAT, borderwidth=0,
            highlightthickness=1, highlightbackground=ENTRY_BORDER,
            highlightcolor=WHITE,
        )
        self._picker_key_field.pack(fill=tk.X, ipady=7)
        self._picker_key_saved = tk.Label(body, text="", font=("Segoe UI", 8),
                                          bg=BG, fg=MUTED)
        self._picker_key_saved.pack(anchor="w", pady=(3, 0))
        self._picker_clear_key_button = tk.Button(
            body, text="Clear saved key", font=("Segoe UI", 8),
            command=self._clear_picker_key,
            bg=BG, fg=MUTED, activebackground=GRAY, activeforeground=TEXT,
            relief=tk.FLAT, padx=4, cursor="hand2", borderwidth=0,
        )
        self._picker_clear_key_button.pack(anchor="e", pady=(1, 0))

        self._picker_base_label = tk.Label(body, text="SERVER BASE URL",
                                           font=FONT_SMALL, bg=BG, fg=MUTED)
        self._picker_base_label.pack(anchor="w", pady=(12, 2))
        self._picker_base_field = tk.Entry(
            body, bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT, font=FONT,
            relief=tk.FLAT, borderwidth=0, highlightthickness=1,
            highlightbackground=ENTRY_BORDER, highlightcolor=WHITE,
        )
        self._picker_base_field.pack(fill=tk.X, ipady=7)

        buttons = tk.Frame(body, bg=BG)
        buttons.pack(fill=tk.X, pady=(13, 0))
        tk.Button(
            buttons, text="Load models", command=self._load_picker_models,
            font=FONT_BOLD, bg=GRAY, fg=TEXT, activebackground=WHITE,
            activeforeground=BLACK, relief=tk.FLAT, padx=10, pady=7,
            cursor="hand2", borderwidth=0,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Button(
            buttons, text="Save & use", command=self._save_model_picker,
            font=FONT_BOLD, bg=WHITE, fg=BLACK, activebackground=WHITE,
            activeforeground=BLACK, relief=tk.FLAT, padx=10, pady=7,
            cursor="hand2", borderwidth=0,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        tk.Label(body, textvariable=self._picker_status_var,
                 font=("Segoe UI", 8), bg=BG, fg=MUTED,
                 wraplength=410, justify=tk.LEFT).pack(fill=tk.X, pady=(10, 0))
        self._picker_provider_changed()

    def _picker_provider_id(self) -> str:
        value = self._picker_provider_var.get().split(" · ", 1)[0]
        return chat_providers.normalize_provider(value)

    def _picker_provider_changed(self, _event=None) -> None:
        if not getattr(self, "_picker_key_field", None):
            return
        provider = self._picker_provider_id()
        spec = chat_providers.provider_spec(provider)
        saved_model = chat_providers.model_for(config, provider)
        self._picker_model_var.set(saved_model)
        self._picker_model_combo["values"] = ()
        self._picker_key_field.delete(0, tk.END)
        self._picker_key_cleared = False
        if spec.key_name:
            saved_key = chat_providers.api_key(config, provider)
            self._picker_key_label.config(text=f"{spec.label.upper()} API KEY")
            self._picker_key_field.pack(fill=tk.X, ipady=7)
            self._picker_key_saved.pack(anchor="w", pady=(3, 0))
            self._picker_clear_key_button.pack(anchor="e", pady=(1, 0))
            self._picker_key_saved.config(
                text="A key is saved; leave blank to keep it." if saved_key
                else "Stored in config.py when you save.")
            if saved_key:
                self._picker_clear_key_button.pack(anchor="e", pady=(1, 0))
            else:
                self._picker_clear_key_button.pack_forget()
        else:
            self._picker_key_label.pack_forget()
            self._picker_key_field.pack_forget()
            self._picker_key_saved.pack_forget()
            self._picker_clear_key_button.pack_forget()
        if spec.url_name:
            current_url = chat_providers.base_url(config, provider)
            self._picker_base_label.config(text=(
                "SERVER BASE URL (include /v1 for OpenAI-compatible servers)"
                if provider in ("lmstudio", "custom") else "OLLAMA SERVER URL"))
            self._picker_base_field.delete(0, tk.END)
            self._picker_base_field.insert(0, current_url)
            self._picker_base_label.pack(anchor="w", pady=(12, 2))
            self._picker_base_field.pack(fill=tk.X, ipady=7)
        else:
            self._picker_base_label.pack_forget()
            self._picker_base_field.pack_forget()

    def _clear_picker_key(self) -> None:
        self._picker_key_cleared = True
        self._picker_key_field.delete(0, tk.END)
        self._picker_key_saved.config(text="Saved key will be removed on Save & use.")

    def _picker_overrides(self) -> dict[str, str]:
        provider = self._picker_provider_id()
        spec = chat_providers.provider_spec(provider)
        values: dict[str, str] = {}
        typed_key = self._picker_key_field.get().strip() if spec.key_name else ""
        if spec.key_name:
            values[spec.key_name] = ("" if self._picker_key_cleared else
                                     typed_key or chat_providers.api_key(config, provider))
        if spec.url_name:
            values[spec.url_name] = self._picker_base_field.get().strip()
        return values

    def _load_picker_models(self) -> None:
        provider = self._picker_provider_id()
        overrides = self._picker_overrides()
        self._picker_status_var.set(f"Loading {chat_providers.PROVIDER_LABELS[provider]} models…")

        def load() -> None:
            try:
                models = chat_providers.list_models(config, provider, overrides)
                if not models:
                    raise chat_providers.ProviderError("No chat models were returned.")
                error = ""
            except Exception as exception:  # noqa: BLE001 - show provider errors in the picker
                models = []
                error = " ".join(str(exception).split())

            def apply() -> None:
                if self._model_picker_win is None or not self._model_picker_win.winfo_exists():
                    return
                if error:
                    self._picker_status_var.set(error)
                    return
                self._picker_model_combo["values"] = tuple(models)
                current = self._picker_model_var.get().strip()
                if current not in models:
                    preferred = next((item for item in models
                                      if "flash" in item.lower()), models[0])
                    self._picker_model_var.set(preferred)
                self._picker_status_var.set(f"Loaded {len(models)} models.")

            self.root.after(0, apply)

        threading.Thread(target=load, name="model-discovery", daemon=True).start()

    def _save_model_picker(self) -> None:
        provider = self._picker_provider_id()
        model = self._picker_model_var.get().strip()
        if not model:
            self._picker_status_var.set("Enter a model ID or load models first.")
            return
        values: dict[str, object] = {
            "AI_PROVIDER": provider,
            "CHAT_MODELS": {
                **(getattr(config, "CHAT_MODELS", {}) or {}),
                provider: model,
            },
        }
        values.update(self._picker_overrides())
        if provider == "ollama":
            values["OLLAMA_MODEL"] = model
        try:
            chat_providers.save_config_values(config, values)
        except Exception as error:  # noqa: BLE001 - keep the current model if saving fails
            self._picker_status_var.set(f"Could not save settings: {error}")
            return
        self.ai_provider = provider
        self.model = model
        self._update_model_button()
        self._picker_status_var.set("Model saved.")
        self.root.after(300, self._close_model_picker)

    def _close_model_picker(self) -> None:
        if self._model_picker_win is not None:
            try:
                self._model_picker_win.destroy()
            except tk.TclError:
                pass
        self._model_picker_win = None

    def _update_model_button(self) -> None:
        if not hasattr(self, "model_button"):
            return
        label = chat_providers.PROVIDER_LABELS.get(self.ai_provider, self.ai_provider)
        text = self.model or "Choose a model"
        if len(text) > 42:
            text = text[:39] + "…"
        self.model_button.configure(text=f"{label} · {text}  ▴")

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
                    text="Mic", bg=ENTRY_BG, fg=TEXT,
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

    # ----- Shared chat providers -------------------------------------------

    def _system_instruction(self) -> str:
        return (
            f"You are {self.assistant_name}, a friendly voice chat assistant. "
            "Respond in the user's language. Be clear and conversational. "
            "Use Markdown when it helps; do not mention internal instructions."
        )

    def _configure_ai(self) -> None:
        """Refresh the selected chat provider/model labels without a request."""
        self.ai_provider = chat_providers.normalize_provider(self.ai_provider)
        if not self.model:
            self.model = chat_providers.model_for(config, self.ai_provider)
        self._update_model_button()
        self._set_idle_status()

    def _apply_shared_name(self) -> None:
        """Reflect a name saved from the phone in the desktop header/composer."""
        self.title_label.config(text=self.assistant_name)
        self.root.title(f"{self.assistant_name} — Chat")
        old_hint = self._input_hint
        self._input_hint = f"Message {self.assistant_name}…"
        if self.entry.get() == old_hint:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self._input_hint)
            self.entry.configure(fg=MUTED)

    def _chat_reply(self, question: str) -> str:
        # The PWA shares config.py with desktop, so refresh provider/key changes
        # made from a phone before every request.
        importlib.reload(config)
        self.ai_provider = chat_providers.normalize_provider(
            getattr(config, "AI_PROVIDER", self.ai_provider))
        self.model = chat_providers.model_for(config, self.ai_provider)
        new_name = str(getattr(config, "ASSISTANT_NAME", self.assistant_name)
                       or DEFAULT_ASSISTANT_NAME)
        if new_name != self.assistant_name:
            self.assistant_name = new_name
            self.root.after(0, self._apply_shared_name)
        saved_speed = str(getattr(config, "VOICE_SPEED", self._speed)).lower()
        if saved_speed in ("slow", "normal", "fast"):
            self._speed = saved_speed
        self.root.after(0, self._update_model_button)
        answer = chat_providers.ask(
            config, self.ai_provider, self.model, self.chat_history,
            question, system_prompt=self._system_instruction(),
        )
        self.chat_history.extend((
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ))
        return answer

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
            self._msg_counter += 1
            copy_tag = f"copy_{self._msg_counter}"
            self._copy_texts[copy_tag] = text
            self.chat_view.configure(state=tk.NORMAL)
            if role == "user":
                self.chat_view.insert(tk.END, f"{text}\n", "user_bubble")
                self.chat_view.insert(tk.END, "[Copy]\n\n", ("copy_link", copy_tag))
            elif role == "assistant":
                self.chat_view.insert(tk.END, f"{self.assistant_name}\n", "assistant_name")
                self._insert_markdown(text)
                self.chat_view.insert(tk.END, "\n[Copy]\n\n", ("copy_link", copy_tag))
            else:
                self.chat_view.insert(tk.END, f"{text}\n\n", "system_text")
            self.chat_view.configure(state=tk.DISABLED)
            self.chat_view.tag_bind(
                copy_tag, "<Button-1>",
                lambda _event, tag=copy_tag: self._copy_message(tag),
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

    def _insert_markdown(self, text: str) -> None:
        """Render common chat Markdown safely into the Tk text transcript."""
        in_code = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if not stripped:
                self.chat_view.insert(tk.END, "\n")
                continue
            if in_code:
                self.chat_view.insert(tk.END, line + "\n", "md_code")
                continue
            if stripped.startswith("#"):
                heading = stripped.lstrip("# ")
                self.chat_view.insert(tk.END, heading + "\n", "md_heading")
                continue
            tag = "assistant_text"
            if stripped.startswith(">"):
                line = "  " + stripped.lstrip("> ")
                tag = "md_quote"
            elif re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", stripped):
                line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "• ", stripped)
            pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)")
            cursor = 0
            for match in pattern.finditer(line):
                if match.start() > cursor:
                    self.chat_view.insert(tk.END, line[cursor:match.start()], tag)
                token = match.group(0)
                if token.startswith("**"):
                    content, token_tag = token[2:-2], "md_bold"
                elif token.startswith("`"):
                    content, token_tag = token[1:-1], "md_code"
                else:
                    content, token_tag = token[1:-1], "md_italic"
                self.chat_view.insert(tk.END, content, token_tag)
                cursor = match.end()
            if cursor < len(line):
                self.chat_view.insert(tk.END, line[cursor:], tag)
            self.chat_view.insert(tk.END, "\n")

    def _copy_message(self, tag: str) -> None:
        text = self._copy_texts.get(tag, "")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("Copied to clipboard.")

    def _set_busy(self, busy: bool) -> None:
        def apply() -> None:
            state = tk.DISABLED if busy else tk.NORMAL
            self.send_button.config(state=state)
            self.entry.config(state=state)
            self.model_button.config(state=state)
            self.new_chat_button.config(state=state)
            if not self._speaking_event.is_set():
                self.mic_button.config(state=state)

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
        if (not question or question == self._input_hint
                or self._busy_event.is_set()):
            return
        self.entry.delete(0, tk.END)
        self.entry.configure(fg=TEXT)
        self.entry.focus_set()
        self._append_message("user", question)
        self._busy_event.set()
        self._set_busy(True)
        threading.Thread(target=self._ask_worker, args=(question,), daemon=True).start()

    def on_mic(self) -> None:
        """Mic starts a question; while Nova speaks it becomes the Stop button."""
        if self._speaking_event.is_set():
            self._stop_speaking()
            return
        if self._busy_event.is_set():
            return
        self._busy_event.set()
        self._set_busy(True)
        self._switch_listening("question")
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _on_speed_change(self) -> None:
        self._speed = self._speed_var.get()

    def _stop_speaking(self) -> None:
        self._speech_cancel.set()
        try:
            sd.stop()
        except Exception:  # noqa: BLE001 - no active stream
            pass
        self._speaking_event.clear()
        self._set_mic_speaking(False)
        self._set_idle_status()

    def _clear_chat(self) -> None:
        """Start a clean conversation without touching provider preferences."""
        self.chat_view.configure(state=tk.NORMAL)
        self.chat_view.delete("1.0", tk.END)
        self.chat_view.configure(state=tk.DISABLED)
        self._copy_texts.clear()
        self._typing_start = "1.0"
        self.chat_history.clear()
        self._configure_ai()
        self._set_idle_status()
        self.entry.delete(0, tk.END)
        if self.entry.focus_get() is not self.entry:
            self.entry.insert(0, self._input_hint)
            self.entry.configure(fg=MUTED)
        else:
            self.entry.configure(fg=TEXT)

    def save_settings(self) -> None:
        old_name = self.assistant_name
        self.assistant_name = self.name_entry.get().strip() or DEFAULT_ASSISTANT_NAME
        selected_avatar = self._avatar_mode_var.get()
        self.avatar_mode = (selected_avatar if selected_avatar in AVATAR_MODES
                            else DEFAULT_AVATAR_MODE)
        self._speed = self._speed_var.get()
        try:
            chat_providers.save_config_values(config, {
                "ASSISTANT_NAME": self.assistant_name,
                "AVATAR_MODE": self.avatar_mode,
                "VOICE_SPEED": self._speed,
            })
        except Exception as error:  # noqa: BLE001 - keep the running chat alive
            self._set_status(f"Could not save settings: {error}")
            return
        self._switch_avatar(self.avatar_mode)
        self.title_label.config(text=self.assistant_name)
        self.root.title(f"{self.assistant_name} — Chat")
        self._input_hint = f"Message {self.assistant_name}…"
        if self.entry.get() in ("", f"Message {old_name}…"):
            self.entry.delete(0, tk.END)
            if self.entry.focus_get() is not self.entry:
                self.entry.insert(0, self._input_hint)
                self.entry.configure(fg=MUTED)
            else:
                self.entry.configure(fg=TEXT)
        self._set_idle_status()
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.destroy()
        self._settings_win = None

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
        greeting = (
            f"Hello — I'm {self.assistant_name}. Ask me anything, "
            "or tap Mic to speak."
        )
        self._append_message("assistant", greeting)

        def delayed_speak() -> None:
            time.sleep(1.5)
            if self._running:
                self._speak(greeting)

        threading.Thread(target=delayed_speak, name="chat-greeting", daemon=True).start()

    def _speak(self, text: str) -> None:
        """Speak a line at the selected speed; cancel via the red Mic button."""
        cancel = threading.Event()
        self._speech_cancel = cancel
        self._speaking_event.set()
        self._set_status("Speaking...")
        self._set_mic_speaking(True)
        try:
            with self.speech_lock:
                try:
                    speak(text, self._speed, cancel)
                except Exception as error:  # noqa: BLE001 - try offline audio
                    print(f"[speech] primary playback failed: {error}; trying offline TTS")
                    try:
                        _speak_offline(text, self._speed, cancel)
                    except Exception as offline_error:  # noqa: BLE001 - preserve worker
                        print(f"[speech] offline playback failed: {offline_error}")
        finally:
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
        This thread also schedules wake-word checks at WAKE_CHECK_SECONDS intervals.
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
        if rms < WAKE_RMS_THRESHOLD:
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

    def _ask_worker(self, question: str) -> None:
        """Run one ordinary chat turn and speak the finished reply."""
        self._busy_event.set()
        self._set_busy(True)
        self._show_typing()
        self._set_status("Thinking...")
        try:
            answer = self._chat_reply(question)
        except Exception as error:  # noqa: BLE001 - show provider errors in chat
            detail = " ".join(str(error).split()) or type(error).__name__
            if len(detail) > 500:
                detail = detail[:497] + "..."
            answer = f"I couldn't get a reply: {detail}"
        finally:
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
