"""Nova — mobile web companion for the voice assistant.

Run this on the same PC as the desktop app:

    python mobile_server.py

then open http://<your-pc-ip>:8080 on your phone (same Wi-Fi network), or use
the printed URL. The page is a phone-first installable web app:

    - Tap the Mic (or type) to talk to Nova — the browser listens.
    - Turn on the wake word (moon button) and just say "Hey Nova".
    - Replies are spoken by the phone's built-in speech synthesis.
    - Settings (gear): Gemini API key, assistant name, voice speed, apps.

It shares config.py and memory.json with the desktop app, so names, memories
and the Allowed Apps list stay in sync. "open [app]" launches the program on
this PC, exactly like the desktop version.

The helper functions below mirror the ones in main.py (kept in sync by hand
so this server has no audio/GUI dependencies and runs on any machine).
"""

import json
import os
import re
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - Gemini is optional for local commands
    genai = None
    types = None

try:
    import requests
except ImportError:  # pragma: no cover - web search needs requests
    requests = None

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# ----------------------------------------------------------------------------
# Constants and helpers — mirror main.py
# ----------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ASSISTANT_NAME = "Nova"
PLACEHOLDER_KEY = "YOUR_GEMINI_API_KEY_HERE"
NO_APP_REPLY = "That app isn't on my allowed list."

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "memory.json")

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".webmanifest": "application/manifest+json",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def extract_search_query(text: str):
    """Return the query when the message asks to "search for ..."."""
    parts = re.split(r"\bsearch for\b", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip(" .:!?")
    return None


def extract_open_app(text: str):
    """Return the app name when the message asks to "open [app name]"."""
    match = re.match(
        r"^\s*(?:please\s+)?(?:can\s+you\s+)?open\s+(.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" .:!?")
    return None


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return a plain-text summary of the results."""
    if requests is None:
        return "(web search is unavailable: the requests package is missing)"
    headers = {"User-Agent": "Mozilla/5.0 (voice-assistant)"}
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1",
                    "skip_disambig": "1"},
            headers=headers,
            timeout=10,
        )
        data = response.json()
    except Exception as error:  # noqa: BLE001 - report and continue
        return f"(web search failed: {error})"

    parts = []
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

    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=10,
        )
        page = response.text
    except Exception as error:  # noqa: BLE001 - report and continue
        return f"(web search failed: {error})"

    results = []
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
                allowed_apps=None) -> None:
    """Persist the settings back to config.py."""
    apps = allowed_apps if allowed_apps is not None else {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    content = (
        '"""Configuration settings for the voice assistant."""\n\n'
        "# Get your Gemini API key from Google AI Studio: "
        "https://aistudio.google.com/apikey\n"
        "# NOTE: Do not commit a real API key to a public repository.\n"
        f"GEMINI_API_KEY = {api_key!r}\n\n"
        "# Name the assistant introduces itself with "
        "(editable in the app's Settings).\n"
        f"ASSISTANT_NAME = {assistant_name!r}\n\n"
        "# Apps the assistant may open with 'open [app name]'.\n"
        f"ALLOWED_APPS = {apps!r}\n"
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


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


def parse_memory_command(text: str):
    """Parse memory commands exactly as spoken.

    Returns (action, payload) with action one of:
    set_name, remember, forget, forget_name, forget_all, recall_name,
    recall_all. Returns None when the utterance is not a memory command.
    """
    s = text.strip()

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

    if re.match(r"^forget\s+(?:that\s+)?my name\b.*$", s, flags=re.IGNORECASE):
        return "forget_name", ""

    m = re.match(r"^forget\s+(?:that\s+)?(?P<fact>.+)$", s, flags=re.IGNORECASE)
    if m:
        fact = m.group("fact").strip(" .!?\"'")
        if fact.lower() in ("everything", "all", "all my memories",
                            "what you know", "what you know about me",
                            "your memories", "memories"):
            return "forget_all", ""
        return "forget", fact

    m = re.match(
        r"^(?:do not|don't)\s+remember\s+(?:that\s+)?(?P<fact>.+)$",
        s, flags=re.IGNORECASE)
    if m:
        return "forget", m.group("fact").strip(" .!?\"'")

    if re.match(r"^(?:what(?:'s|s| is) my name|who am i|do you know my name)\??$",
                s, flags=re.IGNORECASE):
        return "recall_name", ""

    if re.match(r"^(?:what do you (?:remember|know about me)"
                r"|what are (?:my )?(?:your )?memories|what do you know)\??$",
                s, flags=re.IGNORECASE):
        return "recall_all", ""

    m = re.match(
        r"^(?:please\s+)?remember\s+(?:that\s+)?(?P<fact>.+)$",
        s, flags=re.IGNORECASE)
    if m:
        return "remember", m.group("fact").strip(" .!?\"'")

    return None


# ----------------------------------------------------------------------------
# The assistant brain (same behavior as the desktop app)
# ----------------------------------------------------------------------------

class NovaCore:
    """Gemini chat + local commands, shared with the desktop app's files."""

    def __init__(self) -> None:
        self.api_key = getattr(config, "GEMINI_API_KEY", "")
        self.name = getattr(config, "ASSISTANT_NAME", DEFAULT_ASSISTANT_NAME)
        self.allowed_apps = dict(getattr(config, "ALLOWED_APPS", {}) or {})
        self.memory = load_memory()
        self.client = None
        self.chat = None
        self.last_answer = ""
        self.lock = threading.Lock()
        self.configure_gemini()

    # -- Gemini ----------------------------------------------------------

    @property
    def api_ready(self) -> bool:
        return bool(self.api_key) and self.api_key != PLACEHOLDER_KEY

    def configure_gemini(self) -> None:
        if not self.api_ready or genai is None:
            self.client = None
            self.chat = None
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
                    f"You are {self.name}, a friendly voice assistant. "
                    "Keep answers clear and conversational." + memory_block
                ),
            ),
            history=list(history),
        )

    # -- Memory commands ---------------------------------------------------

    def apply_memory(self, action: str, payload: str) -> str:
        if action == "set_name":
            name = payload.strip()
            if not name:
                return "I didn't catch a name. Say 'my name is' followed by your name."
            self.memory["user_name"] = name
            save_memory(self.memory)
            self.configure_gemini()
            return f"Got it — I'll call you {name}."
        if action == "remember":
            fact = payload.strip()
            if not fact:
                return "What should I remember? Say 'remember that' and then the fact."
            if fact.lower() in {f.lower() for f in self.memory["facts"]}:
                return "I already remember that."
            self.memory["facts"].append(fact)
            save_memory(self.memory)
            self.configure_gemini()
            return f"Okay, I'll remember that {fact}."
        if action == "forget":
            before = len(self.memory["facts"])
            self.memory["facts"] = [
                f for f in self.memory["facts"]
                if payload.strip().lower() not in f.lower()
            ]
            save_memory(self.memory)
            self.configure_gemini()
            if len(self.memory["facts"]) < before:
                return "Done — I forgot that."
            return f"I don't have a memory matching '{payload}'."
        if action == "forget_name":
            self.memory["user_name"] = ""
            save_memory(self.memory)
            self.configure_gemini()
            return "Okay, I forgot your name."
        if action == "forget_all":
            self.memory = {"user_name": "", "facts": []}
            save_memory(self.memory)
            self.configure_gemini()
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
                parts.append("I also remember: "
                             + "; ".join(self.memory["facts"]))
            if not parts:
                return ("I don't have any memories yet. "
                        "Say 'remember that' followed by anything and I'll keep it.")
            return "I remember that " + " and ".join(parts) + "."
        return "I couldn't apply that memory command."

    # -- Local commands -----------------------------------------------------

    def local_command(self, question: str):
        memory_cmd = parse_memory_command(question)
        if memory_cmd is not None:
            return self.apply_memory(*memory_cmd)
        q = question.lower().strip().strip(" .!?")
        if q in ("stop", "stop speaking", "be quiet", "quiet"):
            return "Okay, stopping."
        if q in ("clear chat", "clear the chat", "reset conversation",
                 "start over", "new chat"):
            self.chat = None
            self.configure_gemini()
            return "Chat cleared. What would you like to talk about?"
        if q in ("what time is it", "tell me the time", "the time please"):
            return f"It's {time.strftime('%I:%M %p').lstrip('0')}."
        if q in ("what's the date", "what is the date", "what day is it",
                 "what's today", "what is today", "today's date"):
            return f"Today is {time.strftime('%A, %B %d, %Y')}."
        if q in ("help", "what can you do", "your features", "features"):
            return (
                "You can ask me anything, say 'search for' to look something up, "
                "'open' plus an app name to launch a program on the PC, "
                "'what time is it', 'clear chat', 'repeat that', or 'stop' to "
                "silence me. Tell me 'my name is' or 'remember that' and I'll "
                "keep it in memory — ask 'what do you remember' to hear it back. "
                f"Just say 'Hey {self.name}' to start."
            )
        if q in ("repeat", "repeat that", "say that again", "come again"):
            return self.last_answer or "I haven't said anything yet."
        if q in ("goodbye", "exit app", "quit app", "bye"):
            return "Goodbye!"
        return None

    def open_app(self, app_name: str) -> str:
        for name, path in self.allowed_apps.items():
            if name.lower() == app_name.lower():
                try:
                    subprocess.Popen([path])
                    return f"Opening {name}."
                except Exception as error:  # noqa: BLE001 - report and continue
                    return f"I couldn't open {name}: {error}"
        return NO_APP_REPLY

    # -- Full pipeline -------------------------------------------------------

    def ask(self, question: str) -> str:
        with self.lock:
            return self._ask(question)

    def _ask(self, question: str) -> str:
        local_answer = self.local_command(question)
        if local_answer is not None:
            self.last_answer = local_answer
            return local_answer

        app_name = extract_open_app(question)
        if app_name is not None:
            self.last_answer = self.open_app(app_name)
            return self.last_answer

        prompt = question
        query = extract_search_query(question)
        if query:
            results = duckduckgo_search(query)
            prompt = (
                f"The user asked me to search the web for: {query}\n"
                f"DuckDuckGo results:\n{results}\n\n"
                "Based on those results, reply to the user's message below. "
                "Be concise and mention sources when useful.\n\n"
                f"User message: {question}"
            )

        if self.chat is None:
            if genai is None:
                answer = ("The Gemini library is not installed on the server — "
                          "run: pip install google-genai")
            else:
                answer = ("Set your Gemini API key in Settings to start chatting.")
        else:
            try:
                response = self.chat.send_message(prompt)
                answer = (response.text or "").strip() or "(no response)"
            except Exception as error:  # noqa: BLE001 - keep the app alive
                answer = f"Sorry, I ran into a problem: {error}"
        self.last_answer = answer
        return answer

    def clear_chat(self) -> str:
        with self.lock:
            self.chat = None
            self.configure_gemini()
        return "Chat cleared. What would you like to talk about?"


CORE = NovaCore()


# ----------------------------------------------------------------------------
# HTTP server
# ----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "NovaWeb/1.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers --------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str,
              cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            self._json(200, {
                "name": CORE.name,
                "api_ready": CORE.api_ready,
                "apps": CORE.allowed_apps,
                "user_name": CORE.memory.get("user_name", ""),
                "facts": CORE.memory.get("facts", []),
            })
            return
        self._serve_static(path)

    def _serve_static(self, path: str) -> None:
        if path in ("/", ""):
            path = "/index.html"
        rel = os.path.normpath(path.lstrip("/"))
        if rel.startswith("..") or os.path.isabs(rel):
            self._send(404, b"not found", "text/plain")
            return
        full = os.path.join(WEB_DIR, rel)
        if not os.path.isfile(full):
            self._send(404, b"not found", "text/plain")
            return
        ext = os.path.splitext(full)[1].lower()
        content_type = MIME_TYPES.get(ext, "application/octet-stream")
        cache = "no-store" if ext == ".html" else "public, max-age=300"
        with open(full, "rb") as file:
            self._send(200, file.read(), content_type, cache)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        body = self._read_json()
        if path == "/api/message":
            text = str(body.get("text", "")).strip()
            if not text:
                self._json(400, {"error": "empty message"})
                return
            reply = CORE.ask(text)
            self._json(200, {"reply": reply})
            return
        if path == "/api/clear":
            reply = CORE.clear_chat()
            self._json(200, {"reply": reply})
            return
        if path == "/api/settings":
            api_key = body.get("api_key")
            name = body.get("name")
            apps = body.get("apps")
            if api_key is not None:
                CORE.api_key = str(api_key).strip()
            if name is not None:
                CORE.name = str(name).strip() or DEFAULT_ASSISTANT_NAME
            if isinstance(apps, dict):
                CORE.allowed_apps = {
                    str(k).strip(): str(v).strip()
                    for k, v in apps.items() if str(k).strip() and str(v).strip()
                }
            save_config(CORE.api_key, CORE.name, CORE.allowed_apps)
            CORE.configure_gemini()
            self._json(200, {
                "ok": True,
                "name": CORE.name,
                "api_ready": CORE.api_ready,
                "apps": CORE.allowed_apps,
            })
            return
        self._json(404, {"error": "unknown endpoint"})


def lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Nova mobile web app running.")
    print(f"  On this PC:    http://127.0.0.1:{port}")
    print(f"  On your phone: http://{lan_ip()}:{port}   (same Wi-Fi)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
