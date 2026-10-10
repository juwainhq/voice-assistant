"""Configuration settings for the voice assistant."""

# Get your Gemini API key from Google AI Studio: https://aistudio.google.com/apikey
# Replace the placeholder below with your real key.
# NOTE: Do not commit a real API key to a public repository.
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# Desktop AI provider: "gemini" or "ollama" (selectable in Settings).
AI_PROVIDER = "gemini"

# Local Ollama model (download with: ollama pull phi3:mini).
OLLAMA_MODEL = "phi3:mini"

# Desktop avatar selected in Settings: portrait, grokbot, or image.
AVATAR_MODE = "portrait"

# Name the assistant introduces itself with (editable in the app's Settings).
ASSISTANT_NAME = "Nova"
