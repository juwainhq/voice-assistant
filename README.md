# Nova — Voice Assistant

A voice assistant you can talk to on your **desktop** and on your **phone**. The desktop
can use Google Gemini (`gemini-3.5-flash-lite`) or a local Ollama model; the mobile
companion uses Gemini. Say "Hey Nova" and ask anything — Nova can chat, search the web,
remember things about you, and open your apps.

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
                    │  5. Gemini / Ollama  │
                    └────────────────────────┘
```

### The voice pipeline (both apps)

1. **Wake word** — a lightweight listener watches for the assistant's name
   ("Nova", "Hey Nova", "Nova, I have a question"). On the desktop there is only
   ever **one open mic stream** (`sd.InputStream`): `_mic_thread` keeps it
   running and pushes every chunk into `_mic_queue`; `_audio_router_thread`
   routes chunks by `_listening_mode` — a rolling 3-second `_wake_buffer` deque
   (idle) or the `_question_chunks` list (while recording) — so Windows audio
   drivers never see two streams.
   On the phone the browser's speech recognition does the same job.
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
   - **"search for …"** — DuckDuckGo results are fetched and given to the selected AI as context.
   - **Anything else** — sent to the selected AI. Desktop supports Gemini or Local AI
     (Ollama); mobile uses Gemini. Things remembered about you are added to the prompt.
4. **Reply** — the answer appears in the chat and is **spoken out loud**
   (desktop: gTTS behind a hard 3-second `threading.Timer` cap, with an
   immediate offline **pyttsx3** fallback — engine created once at startup,
   `runAndWait()` always runs in its own thread; mobile: the phone's built-in
   voice). The animated portrait reacts to every state: idle, listening,
   thinking, talking, happy, and sad.

### The desktop app (`main.py`)

- tkinter window (900×620, dark monochrome) with header, full-width chat and input row.
- Header: avatar + name on the left; **status dot** (gray = idle, white = listening,
  blinking = speaking) and a **⚙ gear** button that opens the settings popup
  (AI provider, Gemini API key or Ollama model, name, voice speed, Allowed Apps).
- Mic button turns **red (#FF4444)** while Nova speaks — tap it to stop mid-sentence.
- Typing dots while the selected AI thinks, `[Copy]` links on every message, Clear chat.
- Closing the window **minimizes to the system tray** — click the icon to reopen.
- Voice speed: slow / normal / fast (gTTS slow mode, or faster playback).

### The mobile web app (`mobile_server.py` + `web/`)

- A small standard-library web server on your PC + an installable **PWA** for phones.
- Tap the Mic (or type), or turn on the **☾ wake word** for hands-free "Hey Nova".
- Replies are spoken by the **phone's own speech synthesis**.
- Settings (gear) save straight into the same `config.py`; "open [app]" launches
  programs **on the PC** where the server runs.
- Works over your local Wi-Fi — nothing is exposed to the internet.

### Animated desktop portrait

The desktop avatar now uses the supplied `avatar.png` as its character design,
not as a still display: the eyes blink naturally, their catchlights shift with
its gaze, the face breathes subtly, and the mouth smiles or opens/closes while
Nova speaks. Listening shows an animated sound wave; thinking shows animated dots.
The earlier still-image version remains available with `USE_IMAGE_AVATAR = True`,
and the original animated Grok Bot remains available with `AVATAR_STYLE = "grokbot"`.

### Mobile and classic Grok Bot

The mobile PWA keeps its existing animated Grok Bot SVG. The classic desktop
`BubblyFace` is also retained in `main.py` as a switchable alternative.

---

## Quick start

### Desktop (Windows, Python 3.13/3.14)

```bash
pip install -r requirements.txt
python main.py
```

1. Click the **⚙ gear** and choose **Gemini API** (paste a key from
   <https://aistudio.google.com/apikey>) or **Local AI (Ollama)**.
2. For Local AI, install and start Ollama, then download a model such as
   `ollama pull phi3:mini`; Settings defaults to `phi3:mini`. Ollama must be running
   at `http://localhost:11434`. If it is unavailable, Nova falls back to Gemini
   (which requires a saved API key) and displays a chat notice.
3. Click **Mic** (or type) and talk. Say **"Hey Nova"** any time to go hands-free.

### On your phone

```bash
python mobile_server.py
```

The console prints your PC's address — open `http://<your-pc-ip>:8080` in your
phone's browser (same Wi-Fi). Use **"Add to Home Screen"** to install Nova like an
app. Voice input works best in Chrome/Android; on iOS Safari it is more limited —
typing always works.

> No Ollama connection is used by the mobile companion; it continues to use Gemini.
> The mobile server uses only the standard library plus `google-genai` and `requests`.

---

## Features

- **Chat** with conversation history, typing dots, and `[Copy]` on every message.
- **Voice in** — mic button, text fallback, and the "Hey Nova" wake word (always-on).
- **Voice out** — every reply is spoken; stop it any time with the red mic button.
- **Memory** — names and facts persist in `memory.json` between sessions.
- **Web search** — "search for …" answers with DuckDuckGo context and sources.
- **App launcher** — "open [app]" runs programs from your Allowed Apps list.
- **Settings** — choose **Gemini API** or **Local AI (Ollama)**. Gemini uses an API
  key; Ollama uses a configurable model (default `phi3:mini`). Ollama failures fall
  back to Gemini with a chat notice. Provider, model, assistant name, voice speed and
  Allowed Apps are saved to `config.py`.
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
| `config.py` | `GEMINI_API_KEY`, `AI_PROVIDER`, `OLLAMA_MODEL`, `ASSISTANT_NAME`, `ALLOWED_APPS` |
| `avatar.png` | User-provided portrait artwork used by the animated desktop avatar |
| `memory.json` | Long-term memory (auto-created, gitignored) |
| `requirements.txt` | Python dependencies |
| `preview/` | Browser mockups used while designing the UI |

---

## Updates (key points)

> Every change to this project is recorded here as key points, newest first.

**2026-10-10 — Animated desktop portrait based on the supplied avatar design**
- The default header avatar now uses the supplied `avatar.png` as its design reference and animates it: natural blinking/closed eyelids, shifting eye catchlights, subtle breathing, changing smiles, and an open/close speaking mouth.
- Listening and thinking keep their animated wave/dot indicators; the avatar changes expression through idle, happy, sad, thinking, listening, and talking states.
- The old still-image mode remains selectable with `USE_IMAGE_AVATAR = True`; the previous animated Grok Bot remains selectable with `AVATAR_STYLE = "grokbot"`. Missing artwork still falls back to the classic animated face.
- No image upload was altered; the submitted artwork remains in `avatar.jpg`/`avatar.png`. Mobile keeps its existing animated Grok Bot SVG.


**2026-10-10 — Desktop Local AI option (Ollama) with Gemini fallback**
- Added a provider radio to Settings: **Gemini API** or **Local AI (Ollama)**. The selected provider shows only its relevant field — the Gemini key or the Ollama model name.
- Ollama uses the Python `ollama` library at `http://localhost:11434`; the configurable model defaults to `phi3:mini`. Provider/model persist in `config.py`; existing config files default safely to Gemini.
- Local conversations use the same assistant prompt and remembered facts. Search results remain context for the selected model; chat history is retained per provider and reset by Clear chat.
- If the Ollama package/server/model errors, Nova posts a Notice in chat and automatically tries Gemini. If Gemini is not configured or also fails, Nova explains that in the chat rather than crashing.
- Added `ollama` to `requirements.txt`. This option is desktop-only; the mobile companion remains on Gemini, and its config save preserves the desktop provider/model settings.

**2026-10-10 — Mic pipeline v2: `_mic_queue` + `_audio_router_thread` (exact spec)**
- `_mic_thread` opens exactly one `sd.InputStream` and runs permanently, pushing raw chunks into `_mic_queue`.
- `_audio_router_thread` reads `_mic_queue` and routes chunks by the `_listening_mode` string (starts at `"wake"`): into the rolling `_wake_buffer` (deque, 3 s) in wake mode, or `_question_chunks` (list) in question mode.
- Wake checker runs every 2 seconds: it transcribes the wake buffer and `re.search`es for the assistant name; on a hit it switches `_listening_mode` to `"question"`, plays the beep, and collects the question (0.8 s silence cutoff).
- The Mic button switches `_listening_mode` to `"question"` directly; after capture the mode returns to `"wake"` so the wake listener never stops.
- Removed the leftover multi-stream-era helpers (`_open_input_stream`, `record_question`, the old wake worker) — search, memory, app launcher, UI and tray are unchanged.

**2026-10-10 — Static image avatar with emotion overlays**
- The header avatar is now a **static `avatar.png`** (loaded from the app folder via `PhotoImage`/PIL `ImageTk`) shown at 64×64 — swap in your own image any time.
- Animated emoji-style overlay at the avatar's bottom-right: hidden when idle, an animated **sound wave** when listening, animated **"..." dots** when thinking, and pulsing **mouth-open dots** when speaking.
- The animated drawn face (BubblyFace) is kept in `main.py` as the classic fallback; the current avatar-mode switches are documented in the newest key point.

**2026-10-10 — UI polish (desktop)**
- Animated face in the header slimmed down (72 → 48) to give the chat more room.
- User messages now sit in a **#141414 bubble** with 6px padding (visually distinct from assistant replies).
- Assistant messages show the assistant name in **#8C8C8C** directly before the message text ("Nova: …").
- Status line is always visible — it now shows **"Say Nova to start"** when idle instead of going blank.
- The gear button is labeled **"⚙ Settings"** so it's obvious.
- Thin **#242424 separator line** added between the header and the chat window.

**2026-10-10 — Speech robustness (gTTS never hangs)**
- Every gTTS call now runs in a worker thread behind a hard **3-second `threading.Timer`** cap — a hanging or failing internet request can no longer stall the app.
- On failure or timeout the reply is spoken **immediately** with the offline **pyttsx3** engine.
- The pyttsx3 fallback engine is **initialized once at startup** (with its base speaking rate captured so speed changes never compound), and `runAndWait()` always executes in its own thread since it blocks.
- Fallback speech is still cuttable mid-sentence (the red Mic/Stop calls `engine.stop()`), and temp mp3 files are cleaned on every path (failure, timeout, cancel).

**2026-10-10 — Wake word rework (single shared mic stream)**
- Rewrote the desktop wake system around **one shared `sd.InputStream`** that runs in a background thread for the whole session — Windows drivers often reject two streams at once, so only one is ever open.
- The stream feeds two queues: a rolling **3-second wake buffer** and a **question queue**; one `threading.Event` (`_wake_mode`) switches between them.
- Idle: the wake buffer is transcribed every 2 seconds and checked for the assistant's name. Mic button or wake hit: the same stream feeds the question queue for the full question (0.8 s silence cutoff).
- The stream self-heals: if it ever stops, it is automatically reopened. Removed the old second-stream wake path (`_wake_listen_once`) and the stream-opening `record_question()`.

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
