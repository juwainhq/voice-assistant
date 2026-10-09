# Nova — Voice Assistant

A voice assistant you can talk to on your **desktop** and on your **phone**, powered by
Google Gemini (`gemini-3.5-flash-lite`). Say "Hey Nova" and ask anything — it can chat,
search the web, remember things about you, and open your apps.

---

## How it works

Two front-ends share **one brain** (`config.py` + `memory.json`), so your settings,
memories and Allowed Apps stay in sync everywhere.

```
        DESKTOP (main.py)                    MOBILE (mobile_server.py + web/)
 ┌────────────────────────────┐       ┌─────────────────────────────────┐
 │ tkinter window + tray icon │       │ phone browser (installable PWA) │
 │ own always-on mic stream   │       │ browser mic + "Hey Nova" wake   │
 │ gTTS / pyttsx3 speech      │       │ phone speech-synthesis voice    │
 └────────────┬───────────────┘       └──────────────┬──────────────────┘
              │            SHARED BRAIN              │
              └────────► config.py + memory.json ◄────┘
                                │
                    ┌───────────▼────────────┐
                    │  Command router        │
                    │  1. memory commands    │
                    │  2. local commands     │
                    │  3. open [app]         │
                    │  4. "search for ..."   │
                    │  5. Gemini chat        │
                    └────────────────────────┘
```

### The voice pipeline (both apps)

1. **Wake word** — a lightweight listener watches for the assistant's name
   ("Nova", "Hey Nova", "Nova, I have a question"). On the desktop it runs on its
   **own independent mic stream** (`sd.InputStream`), so it never blocks the main
   microphone; on the phone the browser's speech recognition does the same job.
2. **Activation** — a confirmation beep plays and the app switches to
   `Listening...` (the mic button listens for your question).
3. **Understanding** — your speech is transcribed (Google speech recognition on
   desktop; the browser on mobile) and routed:
   - **Memory commands** — "my name is …", "remember that …", "forget …",
     "what do you remember" — handled locally and stored in `memory.json`.
   - **Local commands** — "stop", "clear chat", "what time is it", the date,
     "help", "repeat that", "goodbye" — answered instantly, no internet needed.
   - **"open [app]"** — launches the program on the PC if it's on the
     Allowed Apps list (anything else answers *"That app isn't on my allowed list."*).
   - **"search for …"** — DuckDuckGo results are fetched and given to Gemini as context.
   - **Anything else** — sent to Gemini. Things remembered about you are injected
     into the system prompt, so Nova uses them naturally.
4. **Reply** — the answer appears in the chat and is **spoken out loud**
   (desktop: gTTS with a 3-second timeout and an offline **pyttsx3** fallback;
   mobile: the phone's built-in voice). The animated Grok Bot face reacts to every
   state: idle, listening, thinking, talking, happy, sad.

### The desktop app (`main.py`)

- tkinter window (900×620, dark monochrome) with header, full-width chat and input row.
- Header: avatar + name on the left; **status dot** (gray = idle, white = listening,
  blinking = speaking) and a **⚙ gear** button that opens the settings popup
  (API key, name, voice speed, Allowed Apps).
- Mic button turns **red (#FF4444)** while Nova speaks — tap it to stop mid-sentence.
- Typing dots while Gemini thinks, `[Copy]` links on every message, Clear chat.
- Closing the window **minimizes to the system tray** — click the icon to reopen.
- Voice speed: slow / normal / fast (gTTS slow mode, or faster playback).

### The mobile web app (`mobile_server.py` + `web/`)

- A small standard-library web server on your PC + an installable **PWA** for phones.
- Tap the Mic (or type), or turn on the **☾ wake word** for hands-free "Hey Nova".
- Replies are spoken by the **phone's own speech synthesis**.
- Settings (gear) save straight into the same `config.py`; "open [app]" launches
  programs **on the PC** where the server runs.
- Works over your local Wi-Fi — nothing is exposed to the internet.

### The Grok Bot face (both apps)

A 1:1 Grok Bot model: a smooth pearl head with **two black vertical capsule eyes**
and soft oval blush — no mouth, brows, glints or shine line. The whole expression
lives in the capsules: they grow (listening), drift (thinking), lean outward (happy),
droop (sad), chatter (talking) and squash to dots (blink). The ball itself only
breathes — it never bounces.

---

## Quick start

### Desktop (Windows, Python 3.13/3.14)

```bash
pip install -r requirements.txt
python main.py
```

1. Click the **⚙ gear** and paste your **Gemini API key**
   (from <https://aistudio.google.com/apikey>) — it is stored in `config.py`.
2. Click **Mic** (or type) and talk. Say **"Hey Nova"** any time to go hands-free.

### On your phone

```bash
python mobile_server.py
```

The console prints your PC's address — open `http://<your-pc-ip>:8080` in your
phone's browser (same Wi-Fi). Use **"Add to Home Screen"** to install Nova like an
app. Voice input works best in Chrome/Android; on iOS Safari it is more limited —
typing always works.

> No extra packages: the mobile server uses only the standard library plus the
> existing `google-genai` and `requests` dependencies.

---

## Features

- **Chat** with conversation history, typing dots, and `[Copy]` on every message.
- **Voice in** — mic button, text fallback, and the "Hey Nova" wake word (always-on).
- **Voice out** — every reply is spoken; stop it any time with the red mic button.
- **Memory** — names and facts persist in `memory.json` between sessions.
- **Web search** — "search for …" answers with DuckDuckGo context and sources.
- **App launcher** — "open [app]" runs programs from your Allowed Apps list.
- **Settings** — Gemini API key, assistant name (wake word follows it), voice speed,
  Allowed Apps — saved to `config.py`.
- **Tray icon** — closing the desktop window keeps Nova running in the background.
- **Mobile PWA** — same assistant in your pocket over local Wi-Fi.

## Voice commands

| Say | What happens |
| --- | --- |
| "Hey Nova" / "Nova" (+ your question) | Wake (and ask in one breath) |
| "stop" / "be quiet" | Silence speech |
| "clear chat" | Reset the conversation |
| "what time is it" / "what's the date" | Instant answer |
| "search for …" | Web search answer |
| "open [app name]" | Launch an allowed app on the PC |
| "my name is Ali" / "call me Ali" | Remember your name |
| "remember that …" / "forget that …" | Store / delete a fact |
| "what's my name" / "what do you remember" | Recall memories |
| "help" | List everything Nova can do |
| "repeat that" | Say the last answer again |
| "goodbye" | Quit (desktop) |

---

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Desktop app (tkinter, mic, TTS, wake listener, tray) |
| `mobile_server.py` | Mobile web server + shared assistant brain |
| `web/` | Phone web app (PWA: HTML/JS/CSS, manifest, service worker) |
| `config.py` | `GEMINI_API_KEY`, `ASSISTANT_NAME`, `ALLOWED_APPS` (edited by Settings) |
| `memory.json` | Long-term memory (auto-created, gitignored) |
| `requirements.txt` | Python dependencies |
| `preview/` | Browser mockups used while designing the UI |

---

## Updates (key points)

> Every change to this project is recorded here as key points, newest first.

**2026-10-10 — Mobile release**
- Added `mobile_server.py` + `web/` — a phone-first installable PWA (manifest + service worker).
- Voice in the browser: tap-to-talk Mic and an always-on **"Hey Nova"** wake mode.
- Replies spoken by the phone's built-in speech synthesis (speed: slow/normal/fast).
- Same brain everywhere: settings, memory and Allowed Apps are shared with the desktop app via `config.py` + `memory.json`.
- "open [app]" from the phone launches the program on the PC; settings save to `config.py` from the phone's gear menu.
- Fixed memory recall for spoken "whats my name" (no apostrophe) in both apps.

**2026-10-10 — Desktop overhaul**
- Avatar is now a **1:1 Grok Bot model**: twin capsule eyes + blush, no shine line, mouth or brows; emotions morph the capsules (grow, drift, lean, droop, chatter, blink) on the calm breathing pearl head.
- Wake listener rewritten onto its **own always-on `sd.InputStream`** — no more lock contention with the main mic; restarts itself after every activation and never dies.
- Complete UI redesign: 900×620 full-width layout, settings **popup** behind a ⚙ gear, header **status dot** (gray/white/blinking), user messages right-aligned white, assistant left-aligned gray.
- **No separate Stop button** — the Mic button turns red (#FF4444) while speaking; click to stop mid-sentence.
- Speech fallback: gTTS with a **3-second timeout** falls back to offline **pyttsx3** (`requirements.txt` updated).
- Fixed `TclError: unknown option "-rmargin1"` on Python 3.14 Tk.

**2026-10-09 — Avatar design pass**
- Live avatar picker (`preview/avatars.html`) comparing calm pearl head / kawaii eyes + calm motion / kawaii bubbly ball.
- Calm pearl-bubble-head avatar applied: still while idle, motion only when the emotion calls for it.

**Earlier — core features**
- Long-term memory: name + facts in `memory.json`, voice commands to remember/forget/ask, memories injected into Gemini's system prompt, personal greetings.
- Wake word + activation beep, always-on listener that survives errors.
- Monochrome design system (pure black, sharp corners, white highlights).
- Gemini chat (`gemini-3.5-flash-lite`, `google-genai`), typing indicator dots, `[Copy]` links, Clear chat.
- Voice speed slider, Stop control, DuckDuckGo "search for …", Allowed Apps launcher.
- System-tray minimize (pystray), settings persisted to `config.py`.
