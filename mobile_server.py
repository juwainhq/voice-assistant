"""Nova's local chat server for the phone PWA.

The phone client is a chat-only companion: text or browser voice input, phone
speech output, and the same provider/model settings used by the desktop app.
It runs only on the user's LAN and has no app-launch, memory, or web-search
commands.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote

import chat_providers
import config


WEB_DIR = Path(__file__).resolve().parent / "web"
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
DEFAULT_ASSISTANT_NAME = "Nova"


class NovaCore:
    """One plain-text conversation with whichever configured provider is active."""

    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []
        self.lock = threading.RLock()

    def refresh_config(self) -> None:
        """Pick up settings changes made by the desktop process."""
        importlib.reload(config)

    @property
    def name(self) -> str:
        return str(getattr(config, "ASSISTANT_NAME", DEFAULT_ASSISTANT_NAME)
                   or DEFAULT_ASSISTANT_NAME)

    @property
    def provider(self) -> str:
        return chat_providers.normalize_provider(
            getattr(config, "AI_PROVIDER", "google"))

    @property
    def model(self) -> str:
        return chat_providers.model_for(config, self.provider)

    @property
    def voice_speed(self) -> str:
        value = str(getattr(config, "VOICE_SPEED", "normal")).lower()
        return value if value in ("slow", "normal", "fast") else "normal"

    @property
    def provider_ready(self) -> bool:
        return chat_providers.provider_ready(config, self.provider)

    @property
    def key_saved(self) -> bool:
        return bool(chat_providers.api_key(config, self.provider))

    @property
    def server_url(self) -> str:
        return chat_providers.base_url(config, self.provider)

    def state(self) -> dict[str, Any]:
        with self.lock:
            history = [dict(message) for message in self.history[-40:]]
        return {
            "name": self.name,
            "provider": self.provider,
            "provider_label": chat_providers.PROVIDER_LABELS[self.provider],
            "model": self.model,
            "models": dict(getattr(config, "CHAT_MODELS", {}) or {}),
            "providers": [
                {"id": provider.id, "label": provider.label}
                for provider in chat_providers.PROVIDERS
            ],
            "api_ready": self.provider_ready,
            "key_saved": self.key_saved,
            "keys_saved": {
                provider.id: bool(chat_providers.api_key(config, provider.id))
                for provider in chat_providers.PROVIDERS if provider.key_name
            },
            "base_url": self.server_url,
            "base_urls": {
                provider.id: chat_providers.base_url(config, provider.id)
                for provider in chat_providers.PROVIDERS if provider.url_name
            },
            "speed": self.voice_speed,
            "history": history,
        }

    def ask(self, question: str) -> str:
        self.refresh_config()
        with self.lock:
            provider = self.provider
            model = self.model
            answer = chat_providers.ask(
                config, provider, model, self.history, question,
                system_prompt=(
                    f"You are {self.name}, a friendly voice chat assistant. "
                    "Respond in the user's language. Be clear and conversational. "
                    "Use Markdown when it helps; do not mention internal instructions."
                ),
            )
            self.history.extend((
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ))
            return answer

    def clear_chat(self) -> None:
        with self.lock:
            self.history.clear()


CORE = NovaCore()


def provider_overrides(body: dict[str, Any], provider: str) -> dict[str, str]:
    spec = chat_providers.provider_spec(provider)
    overrides: dict[str, str] = {}
    key = str(body.get("api_key", "") or "").strip()
    if spec.key_name and key:
        overrides[spec.key_name] = key
    if spec.url_name and "base_url" in body:
        overrides[spec.url_name] = str(body.get("base_url", "") or "").strip()
    return overrides


class Handler(BaseHTTPRequestHandler):
    server_version = "NovaChat/2.0"

    def log_message(self, _fmt: str, *_args: Any) -> None:
        pass

    def _send(self, code: int, body: bytes, content_type: str,
              cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, encoded, "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        try:
            length = min(int(self.headers.get("Content-Length", "0") or "0"), 65536)
            if length <= 0:
                return {}
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            return value if isinstance(value, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            CORE.refresh_config()
            self._json(200, CORE.state())
            return
        self._serve_static(path)

    def _serve_static(self, requested: str) -> None:
        decoded = unquote(requested)
        relative = decoded.lstrip("/") or "index.html"
        path = (WEB_DIR / relative).resolve()
        if path != WEB_DIR and WEB_DIR not in path.parents:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        if not path.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        content_type = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
        cache = "no-store" if path.suffix.lower() in (".html", ".js", ".css") else "public, max-age=300"
        self._send(200, path.read_bytes(), content_type, cache)

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        path = self.path.split("?", 1)[0]
        body = self._read_json()
        if path == "/api/message":
            question = str(body.get("text", "")).strip()
            if not question:
                self._json(400, {"error": "empty message"})
                return
            try:
                answer = CORE.ask(question)
                self._json(200, {"reply": answer})
            except Exception as error:  # noqa: BLE001 - report integration errors in the chat
                detail = " ".join(str(error).split()) or type(error).__name__
                self._json(200, {
                    "reply": f"I couldn't get a reply: {detail}",
                    "error": detail,
                })
            return

        if path == "/api/clear":
            CORE.clear_chat()
            self._json(200, {"ok": True})
            return

        if path == "/api/models":
            provider = chat_providers.normalize_provider(body.get("provider"))
            try:
                models = chat_providers.list_models(
                    config, provider, provider_overrides(body, provider))
                self._json(200, {"models": models})
            except Exception as error:  # noqa: BLE001 - show provider errors in picker
                self._json(400, {"error": " ".join(str(error).split())})
            return

        if path == "/api/settings":
            provider = chat_providers.normalize_provider(body.get("provider", CORE.provider))
            model = str(body.get("model", CORE.model) or "").strip()
            if not model:
                model = chat_providers.provider_spec(provider).default_model
            name = str(body.get("name", CORE.name) or DEFAULT_ASSISTANT_NAME).strip()
            name = name[:40] or DEFAULT_ASSISTANT_NAME
            speed = str(body.get("speed", CORE.voice_speed)).lower()
            if speed not in ("slow", "normal", "fast"):
                speed = "normal"
            models = dict(getattr(config, "CHAT_MODELS", {}) or {})
            models[provider] = model
            values: dict[str, Any] = {
                "AI_PROVIDER": provider,
                "CHAT_MODELS": models,
                "ASSISTANT_NAME": name,
                "VOICE_SPEED": speed,
            }
            spec = chat_providers.provider_spec(provider)
            if spec.key_name:
                if body.get("clear_key"):
                    values[spec.key_name] = ""
                elif str(body.get("api_key", "") or "").strip():
                    values[spec.key_name] = str(body["api_key"]).strip()
            if spec.url_name:
                values[spec.url_name] = str(body.get("base_url", "") or "").strip()
            if provider == "ollama":
                values["OLLAMA_MODEL"] = model
            try:
                chat_providers.save_config_values(config, values)
                self._json(200, {"ok": True, **CORE.state()})
            except Exception as error:  # noqa: BLE001 - do not replace current settings on error
                self._json(400, {"error": " ".join(str(error).split())})
            return

        self._json(404, {"error": "unknown endpoint"})


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    print("Nova Voice Chat PWA running.")
    print(f"  On this PC:    http://127.0.0.1:{port}")
    print(f"  On your phone: http://{lan_ip()}:{port}   (same Wi-Fi)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
