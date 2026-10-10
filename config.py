"""Configuration settings for Nova Voice Chat."""

# Add keys for the providers you use in the Chat settings window. Do not commit
# real API keys: these fields are persisted here for compatibility with Nova's
# existing config.py setup. Local providers do not need a key.
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
ANTHROPIC_API_KEY = ""
OPENAI_API_KEY = ""
OPENROUTER_API_KEY = ""
CUSTOM_API_KEY = ""

# Provider ids: anthropic, google (Gemini), openai, openrouter, ollama,
# lmstudio, or custom. "gemini" remains accepted as a legacy alias.
AI_PROVIDER = "google"
CHAT_MODELS = {
    "anthropic": "claude-opus-5",
    "google": "gemini-3.5-flash-lite",
    "openai": "gpt-4o",
    "openrouter": "openrouter/auto",
    "ollama": "phi3:mini",
    "lmstudio": "",
    "custom": "",
}

# Local and custom OpenAI-compatible server addresses.
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"  # legacy setting; CHAT_MODELS takes precedence
LMSTUDIO_URL = "http://localhost:1234/v1"
CUSTOM_API_BASE_URL = ""

# Chat UI and voice preferences.
ASSISTANT_NAME = "Nova"
VOICE_SPEED = "normal"
AVATAR_MODE = "portrait"  # portrait, grokbot, or image
ALLOWED_APPS = {}  # retained for old config files; the chat-only app does not use it
