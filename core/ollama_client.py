"""Async Ollama client for Kimi-K2.6 inference."""

import logging
from typing import Any, Dict, Optional

import httpx

from core.model_config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


def _normalize_base_url(base_url: str) -> str:
    """Normalize host-level Ollama URLs before appending API routes."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[:-4]
    return normalized


class OllamaClient:
    """Async httpx client for Ollama generate endpoint."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        api_key: str = OLLAMA_API_KEY,
        timeout: float = OLLAMA_TIMEOUT_SECONDS,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a generate request and return the model response text."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        try:
            response = await self._client.post("/api/generate", json=payload)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                "Could not connect to Ollama generate endpoint "
                f"{self.base_url}/api/generate for model {self.model} "
                f"(api_key_set={bool(self.api_key)}). "
                "Check OLLAMA_BASE_URL, OLLAMA_MODEL, and OLLAMA_API_KEY "
                "in the benchmark environment."
            ) from exc
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", ""))

    async def close(self) -> None:
        await self._client.aclose()
