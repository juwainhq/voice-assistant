# Nova — Voice Chat

Nova is a chat-only voice assistant for desktop and phone. Type a message or
speak; replies appear in a conversation and are spoken aloud. Choose one of
seven chat providers and models from either interface. Nova does **not** launch
apps, keep long-term memories, or perform web or image searches.

## How it works

```text
Desktop (main.py)                      Phone (mobile_server.py + web/)
┌────────────────────────────┐         ┌──────────────────────────────┐
│ Tk chat, shared mic stream │         │ Installable browser chat    │
│ Google STT · gTTS / pyttsx3│         │ Browser STT · phone speech   │
└─────────────┬──────────────┘         └──────────────┬───────────────┘
              │                                        │
              └────────── shared config.py ───────────┘
                                 │
                  Anthropic · Gemini · OpenAI
                 OpenRouter · Ollama · LM Studio
                  Custom OpenAI-compatible API
```

Both clients call the shared adapters in `chat_providers.py`. Provider, model,
assistant name, avatar, voice speed, API keys, and local server addresses are
stored in `config.py`. Chat history is held for the current conversation and
cleared with **New chat**; no long-term memory file is used.

### Chat providers

The picker supports **Anthropic**, **Google Gemini**, **OpenAI**, **OpenRouter**,
**Ollama**, **LM Studio**, and **Custom OpenAI-compatible** servers. Models can
be loaded from the provider or entered by ID. The default remains Google Gemini
with `gemini-3.5-flash-lite`.

- Add cloud API keys in the model picker. Keys are stored as plain text in
  `config.py` on the PC; do not share or commit real keys.
- Ollama defaults to `http://localhost:11434` and `phi3:mini`.
- LM Studio defaults to `http://localhost:1234/v1`.
- For a custom server, enter its OpenAI-compatible `/v1` base URL and model ID;
  an API key is optional for keyless local servers.
- The phone server sends requests from the PC, so `localhost` refers to that PC,
  not the phone. Keep local model servers reachable from the machine running
  Nova.

### Desktop voice and conversation

The desktop keeps one microphone stream open. `_mic_thread` owns the sole
`sounddevice.InputStream` and puts audio chunks on `_mic_queue`. The
`_audio_router_thread` routes them to the rolling 2-second `_wake_buffer` or to
`_question_chunks`, based on `_listening_mode` (`"wake"` or `"question"`). Wake
checks run every 0.5 seconds and only transcribe when RMS is at least `0.008`.
Saying **“Hey Nova”** switches to question mode, plays a short tone, and captures
the follow-up; the Mic button starts question mode directly. Capture stops after
0.8 seconds of silence. If the microphone or speech recognition is unavailable,
type in the composer instead. PyAudio is not used.

Every assistant reply is spoken. gTTS is capped by a 3-second
`threading.Timer`; a failure or timeout immediately falls back to the pyttsx3
engine initialized once at startup. Blocking `runAndWait()` runs in its own
thread, and temporary MP3 files are deleted after playback. The red
**#FF4444** Stop button interrupts speech. Voice speed is configurable.

The desktop uses a dark, centered chat layout with Markdown replies, copy
controls, a provider/model picker, New chat, Settings, and the system tray.
Exact active statuses are **Listening...**, **Thinking...**, and **Speaking...**.

### Phone chat

`mobile_server.py` serves the PWA in `web/` and uses the same provider settings
and chat adapters as desktop. On the phone, speech recognition and spoken
responses use browser-provided speech APIs; typed input remains available if
voice recognition is unsupported or microphone permission is denied. The wake
word can be toggled on or off in the header. The phone UI also has provider/model
settings and New chat.

Run the server on the PC and open its LAN address on a phone connected to the
same Wi-Fi. Requests go from the server to the selected provider; the phone
never calls `localhost` to reach a model server.

## Avatars

The desktop Settings selector remains **Animated portrait**, **Classic Grok
Bot**, and **Static image**. The mouthless animated portrait is the default; it
uses gentle body/head movement and eye, brow, gaze, and blush expressions. The
classic `BubblyFace` option and static-image option remain available, and the
selection persists in `config.py` as `AVATAR_MODE`. `USE_IMAGE_AVATAR` remains as
a legacy rollback switch. The phone keeps its existing animated SVG avatar.

## Quick start

### Desktop

```bash
pip install -r requirements.txt
python main.py
```

Open **Settings → Provider & model**, choose a provider, add its API key or local
server URL if needed, choose a model, then save. Click Mic or type a message;
say “Hey Nova” for hands-free desktop listening.

### Phone

Start the server on the PC:

```bash
python mobile_server.py
```

Open the printed `http://<PC-LAN-address>:8080` URL on a phone on the same
network. Use the browser's **Add to Home Screen** action to install the PWA.
Browser voice input and PWA installation vary by platform; typing is always
available. The server is intended for a trusted local network, not direct public
internet exposure.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Desktop chat, avatar selector, shared microphone router, TTS, and tray |
| `mobile_server.py` | Local web server and phone-chat API |
| `chat_providers.py` | Shared adapters, model listing, and config persistence |
| `config.py` | Provider/model, key, voice, name, and avatar settings |
| `web/` | Phone PWA (HTML, CSS, JavaScript, manifest, service worker) |
| `requirements.txt` | Python dependencies |
| `avatar.png` | Supplied static avatar for the Static image choice; replace with your own art if desired |

`ALLOWED_APPS` remains in `config.py` for old config-file compatibility but is
not read by the chat-only app. No Coucou names, character art, sounds, or other
branding assets are used.

## Upstream attribution

The provider-first chat-picker interaction is inspired by
[Louis-CFM/coucou](https://github.com/Louis-CFM/coucou). Coucou is MIT-licensed;
the required copyright and license notice is included in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Nova's provider adapters,
voice pipeline, avatars, and interface are implemented for this project.

## Updates

> Dated newest-first summary of the current chat-only work.

**2026-10-10 — Chat-only desktop and phone revamp**
- Replaced the desktop's active command router with ordinary provider-backed
  chat; removed memory commands, app launching, and web/image search from the
  active experience.
- Added the same provider/model settings to desktop and phone: Anthropic,
  Google Gemini, OpenAI, OpenRouter, Ollama, LM Studio, and custom
  OpenAI-compatible endpoints.
- Refreshed the phone PWA as a responsive chat surface with Markdown replies,
  provider settings, voice input/output, and New chat.
- Preserved the one-stream desktop microphone design, typed fallback, TTS
  timeout/fallback rules, status text, stop-button color, and all three avatar
  choices. Added Coucou MIT attribution without copying product assets.
