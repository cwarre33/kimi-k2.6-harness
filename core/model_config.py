"""Ollama model client configuration constants."""

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "kimi-k2.6")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
OLLAMA_CLOUD_MODE = os.getenv("OLLAMA_CLOUD_MODE", "false").lower() in ("true", "1", "yes")
