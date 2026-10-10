"""Chat providers shared by Nova's desktop and phone chat clients.

The provider picker follows the small, provider-first chat workflow in Coucou
(https://github.com/Louis-CFM/coucou, MIT), adapted to this Python app. No
Coucou character art, sounds, or other product features are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping
from urllib.parse import quote, urlsplit

import requests


GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_KEY_PLACEHOLDER = "YOUR_GEMINI_API_KEY_HERE"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"
DEFAULT_OLLAMA_MODEL = "phi3:mini"


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    api: str
    key_name: str | None = None
    url_name: str | None = None
    default_url: str = ""
    default_model: str = ""


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec("anthropic", "Anthropic", "anthropic", "ANTHROPIC_API_KEY",
                 default_model="claude-opus-5"),
    ProviderSpec("google", "Google Gemini", "google", "GEMINI_API_KEY",
                 default_model=GEMINI_MODEL),
    ProviderSpec("openai", "OpenAI", "openai", "OPENAI_API_KEY",
                 default_model="gpt-4o"),
    ProviderSpec("openrouter", "OpenRouter", "openai", "OPENROUTER_API_KEY",
                 default_model="openrouter/auto"),
    ProviderSpec("ollama", "Ollama", "ollama", url_name="OLLAMA_URL",
                 default_url=DEFAULT_OLLAMA_URL,
                 default_model=DEFAULT_OLLAMA_MODEL),
    ProviderSpec("lmstudio", "LM Studio", "openai", url_name="LMSTUDIO_URL",
                 default_url=DEFAULT_LMSTUDIO_URL),
    ProviderSpec("custom", "Custom OpenAI-compatible", "openai",
                 "CUSTOM_API_KEY", "CUSTOM_API_BASE_URL"),
)

PROVIDER_IDS = tuple(provider.id for provider in PROVIDERS)
PROVIDER_LABELS = {provider.id: provider.label for provider in PROVIDERS}
_PROVIDER_BY_ID = {provider.id: provider for provider in PROVIDERS}
_ALIASES = {"gemini": "google", "google-ai": "google", "lm_studio": "lmstudio"}

# Names written to config.py. Existing ALLOWED_APPS is retained for backward
# compatibility with older installations, but the chat-only app no longer uses it.
CONFIG_NAMES = frozenset({
    "AI_PROVIDER", "CHAT_MODELS", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY", "OPENROUTER_API_KEY", "CUSTOM_API_KEY",
    "OLLAMA_URL", "OLLAMA_MODEL", "LMSTUDIO_URL", "CUSTOM_API_BASE_URL",
    "ASSISTANT_NAME", "AVATAR_MODE", "VOICE_SPEED", "ALLOWED_APPS",
})
_CONFIG_LOCK = threading.Lock()


class ProviderError(RuntimeError):
    """An actionable provider/network error safe to show in the chat."""


def normalize_provider(provider: str | None) -> str:
    """Return a supported canonical provider id, defaulting to Google Gemini."""
    value = str(provider or "google").strip().lower()
    value = _ALIASES.get(value, value)
    return value if value in _PROVIDER_BY_ID else "google"


def provider_spec(provider: str | None) -> ProviderSpec:
    return _PROVIDER_BY_ID[normalize_provider(provider)]


def config_value(config_module: Any, name: str, fallback: Any = "") -> Any:
    return getattr(config_module, name, fallback)


def api_key(config_module: Any, provider: str) -> str:
    spec = provider_spec(provider)
    if not spec.key_name:
        return ""
    value = str(config_value(config_module, spec.key_name, "") or "").strip()
    if provider == "google" and value == GEMINI_KEY_PLACEHOLDER:
        return ""
    return value


def base_url(config_module: Any, provider: str) -> str:
    spec = provider_spec(provider)
    if not spec.url_name:
        return ""
    configured = str(config_value(config_module, spec.url_name, "") or "").strip()
    return configured or spec.default_url


def model_for(config_module: Any, provider: str) -> str:
    provider = normalize_provider(provider)
    models = config_value(config_module, "CHAT_MODELS", {})
    if isinstance(models, Mapping):
        chosen = str(models.get(provider, "") or "").strip()
        if chosen:
            return chosen
    # Read the two legacy setting names so existing config.py files keep working.
    if provider == "ollama":
        legacy = str(config_value(config_module, "OLLAMA_MODEL", "") or "").strip()
        return legacy or provider_spec(provider).default_model
    return provider_spec(provider).default_model


def provider_ready(config_module: Any, provider: str) -> bool:
    spec = provider_spec(provider)
    if spec.key_name and provider != "custom" and not api_key(config_module, provider):
        return False
    if spec.url_name and not base_url(config_module, provider):
        return False
    return True


def _validated_base(raw: str, provider: str) -> str:
    value = raw.strip().rstrip("/")
    if not value:
        raise ProviderError("Enter a server base URL first.")
    if "://" not in value:
        value = "http://" + value
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ProviderError("Enter a valid HTTP or HTTPS server base URL.")
    if provider in ("lmstudio", "custom") and not parsed.path.rstrip("/").endswith("/v1"):
        value += "/v1"
    return value.rstrip("/")


def _auth_headers(config_module: Any, provider: str) -> dict[str, str]:
    spec = provider_spec(provider)
    key = api_key(config_module, provider)
    if spec.key_name and not key and provider != "custom":
        raise ProviderError(f"Add your {spec.label} API key in Chat settings.")
    headers = {"Content-Type": "application/json"}
    if provider == "anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    if provider == "openrouter":
        headers["X-Title"] = "Nova Voice Chat"
    return headers


def _response_json(method: str, url: str, *, provider: str,
                   timeout: int = 20, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as error:
        detail = str(error)
        params = kwargs.get("params") or {}
        headers = kwargs.get("headers") or {}
        secrets = [str(value) for value in params.values() if value]
        secrets.extend(str(value).removeprefix("Bearer ") for name, value in headers.items()
                       if name.lower() in ("authorization", "x-api-key") and value)
        for secret in secrets:
            detail = detail.replace(secret, "[redacted]")
        raise ProviderError(
            f"{PROVIDER_LABELS[provider]} connection failed: {detail}"
        ) from error
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError) as error:
        if not response.ok:
            detail = f"HTTP {response.status_code}"
        else:
            detail = "the server returned invalid JSON"
        raise ProviderError(f"{PROVIDER_LABELS[provider]}: {detail}.") from error
    if not response.ok:
        error_data = data.get("error", {}) if isinstance(data, dict) else {}
        if isinstance(error_data, dict):
            detail = error_data.get("message") or error_data.get("detail")
        else:
            detail = error_data
        detail = str(detail or data.get("message", f"HTTP {response.status_code}"))
        raise ProviderError(f"{PROVIDER_LABELS[provider]}: {detail}")
    if not isinstance(data, dict):
        raise ProviderError(f"{PROVIDER_LABELS[provider]} returned an unexpected response.")
    return data


def _compat_base(config_module: Any, provider: str) -> str:
    spec = provider_spec(provider)
    if provider in ("openai", "openrouter"):
        return "https://api.openai.com/v1" if provider == "openai" else "https://openrouter.ai/api/v1"
    return _validated_base(base_url(config_module, provider), provider)


def list_models(config_module: Any, provider: str,
                overrides: Mapping[str, Any] | None = None) -> list[str]:
    """Return chat model IDs from the selected provider's model-list endpoint."""
    provider = normalize_provider(provider)
    spec = provider_spec(provider)
    settings = dict(overrides or {})

    class SettingsView:
        def __getattr__(self, name: str) -> Any:
            if name in settings:
                return settings[name]
            return getattr(config_module, name, "")

    view = SettingsView()
    if spec.key_name and not api_key(view, provider) and provider != "custom":
        raise ProviderError(f"Add your {spec.label} API key in Chat settings.")
    headers = _auth_headers(view, provider)

    if provider == "google":
        data = _response_json(
            "GET", "https://generativelanguage.googleapis.com/v1beta/models",
            provider=provider, params={"key": api_key(view, provider)}, headers={},
        )
        models = []
        for item in data.get("models", []):
            if "generateContent" not in item.get("supportedGenerationMethods", []):
                continue
            name = str(item.get("name", "")).removeprefix("models/")
            if name:
                models.append(name)
        return sorted(set(models))

    if provider == "anthropic":
        data = _response_json(
            "GET", "https://api.anthropic.com/v1/models", provider=provider,
            params={"limit": 100}, headers=headers,
        )
        return sorted({str(item.get("id", "")) for item in data.get("data", [])
                       if item.get("id")})

    if provider == "ollama":
        root = _validated_base(base_url(view, provider), provider)
        data = _response_json("GET", f"{root}/api/tags", provider=provider, headers={})
        return sorted({str(item.get("name", "")) for item in data.get("models", [])
                       if item.get("name")})

    root = _compat_base(view, provider)
    data = _response_json("GET", f"{root}/models", provider=provider, headers=headers)
    items = data.get("data", data.get("models", []))
    skip = ("embed", "whisper", "transcri", "dall-e", "moderation", "tts-")
    models = []
    for item in items if isinstance(items, list) else []:
        name = item.get("id") or item.get("name") if isinstance(item, dict) else item
        name = str(name or "").strip()
        if name and not any(term in name.lower() for term in skip):
            models.append(name)
    return sorted(set(models))


def ask(config_module: Any, provider: str, model: str,
        history: list[dict[str, str]], user_text: str,
        system_prompt: str | None = None,
        overrides: Mapping[str, Any] | None = None) -> str:
    """Send one turn and return its text. History is committed by the caller."""
    provider = normalize_provider(provider)
    spec = provider_spec(provider)
    model = str(model or model_for(config_module, provider)).strip()
    if not model:
        raise ProviderError("Choose a model in the model picker before chatting.")

    settings = dict(overrides or {})

    class SettingsView:
        def __getattr__(self, name: str) -> Any:
            if name in settings:
                return settings[name]
            return getattr(config_module, name, "")

    view = SettingsView()
    prompt = system_prompt or (
        "You are Nova, a friendly voice chat assistant. "
        "Respond in the user's language. Be clear and conversational. "
        "Use Markdown when it helps; do not mention internal instructions."
    )
    messages = [
        {"role": message.get("role", "user"),
         "content": str(message.get("content", ""))}
        for message in history[-40:]
        if message.get("role") in ("user", "assistant") and message.get("content")
    ]
    messages.append({"role": "user", "content": user_text})
    headers = _auth_headers(view, provider)

    if provider == "google":
        contents = [
            {"role": "model" if item["role"] == "assistant" else "user",
             "parts": [{"text": item["content"]}]}
            for item in messages
        ]
        body = {
            "systemInstruction": {"parts": [{"text": prompt}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 4096},
        }
        data = _response_json(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(model, safe='')}:generateContent",
            provider=provider,
            params={"key": api_key(view, provider)},
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=90,
        )
        candidates = data.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        answer = "".join(str(part.get("text", "")) for part in parts).strip()
        if not answer:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise ProviderError(f"Google Gemini returned no text{f' ({block})' if block else ''}.")
        return answer

    if provider == "anthropic":
        body = {
            "model": model,
            "max_tokens": 4096,
            "system": prompt,
            "messages": messages,
        }
        data = _response_json(
            "POST", "https://api.anthropic.com/v1/messages", provider=provider,
            json=body, headers=headers, timeout=90,
        )
        blocks = data.get("content") or []
        answer = "\n".join(str(block.get("text", "")) for block in blocks
                            if block.get("type") == "text").strip()
        if not answer:
            raise ProviderError("Anthropic returned an empty response.")
        return answer

    if provider == "ollama":
        root = _validated_base(base_url(view, provider), provider)
        body = {"model": model, "messages": [
            {"role": "system", "content": prompt}, *messages,
        ], "stream": False}
        data = _response_json(
            "POST", f"{root}/api/chat", provider=provider,
            json=body, headers={"Content-Type": "application/json"}, timeout=120,
        )
        answer = str((data.get("message") or {}).get("content", "")).strip()
        if not answer:
            raise ProviderError("Ollama returned an empty response.")
        return answer

    root = _compat_base(view, provider)
    body = {"model": model, "messages": [
        {"role": "system", "content": prompt}, *messages,
    ], "stream": False}
    data = _response_json(
        "POST", f"{root}/chat/completions", provider=provider,
        json=body, headers=headers, timeout=120,
    )
    choices = data.get("choices") or []
    answer = str(((choices[0].get("message") or {}).get("content", ""))
                 if choices else "").strip()
    if not answer:
        raise ProviderError(f"{spec.label} returned an empty response.")
    return answer


def save_config_values(config_module: Any, values: Mapping[str, Any]) -> None:
    """Atomically update known settings in config.py while preserving other lines."""
    unknown = set(values) - CONFIG_NAMES
    if unknown:
        raise ValueError(f"Unsupported config field(s): {', '.join(sorted(unknown))}")
    path = Path(getattr(config_module, "__file__", "config.py")).resolve()
    with _CONFIG_LOCK:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = '"""Configuration settings for Nova Voice Chat."""\n\n'
        for name, value in values.items():
            assignment = f"{name} = {value!r}"
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as error:
                raise ValueError(
                    "config.py must contain valid Python before settings can be saved"
                ) from error
            node = next((item for item in tree.body
                         if isinstance(item, (ast.Assign, ast.AnnAssign))
                         and any(isinstance(target, ast.Name) and target.id == name
                                 for target in (item.targets if isinstance(item, ast.Assign)
                                                else [item.target]))), None)
            if node is None:
                text = text.rstrip() + "\n\n" + assignment + "\n"
                continue
            lines = text.splitlines(keepends=True)
            start_line = node.lineno - 1
            end_line = getattr(node, "end_lineno", node.lineno)
            has_newline = bool(lines[end_line - 1].endswith(("\n", "\r")))
            lines[start_line:end_line] = [assignment + ("\n" if has_newline else "")]
            text = "".join(lines)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            try:
                mode = path.stat().st_mode & 0o777
                os.chmod(temp_name, mode)
            except OSError:
                pass
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    for name, value in values.items():
        setattr(config_module, name, value)
