"""Tests for the async Ollama client."""

import os
from unittest.mock import patch

import httpx
import pytest

from core.ollama_client import OllamaClient
from core import model_config


def test_defaults_are_sensible():
    assert model_config.OLLAMA_BASE_URL == "http://localhost:11434"
    assert model_config.OLLAMA_MODEL == "kimi-k2.6"
    assert model_config.OLLAMA_TIMEOUT_SECONDS == 300.0


@pytest.mark.asyncio
async def test_generate_returns_model_response(monkeypatch):
    monkeypatch.setattr(
        model_config, "OLLAMA_BASE_URL", "http://localhost:11434"
    )

    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert "/api/generate" in str(request.url)
        return httpx.Response(200, json={"response": "Run pytest -v", "done": True})

    transport = httpx.MockTransport(handler)
    client = OllamaClient()
    client._client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=transport
    )

    result = await client.generate("How do I run tests?")
    assert result == "Run pytest -v"


@pytest.mark.asyncio
async def test_generate_includes_api_key_when_configured(monkeypatch):
    monkeypatch.setattr(
        model_config, "OLLAMA_BASE_URL", "http://localhost:11434"
    )

    auth_header = None

    def handler(request: httpx.Request):
        nonlocal auth_header
        auth_header = request.headers.get("Authorization")
        return httpx.Response(200, json={"response": "ok", "done": True})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(api_key="secret-token")
    client._client = httpx.AsyncClient(
        base_url="http://localhost:11434",
        headers={"Authorization": "Bearer secret-token"},
        transport=transport,
    )

    await client.generate("test")
    assert auth_header == "Bearer secret-token"


@pytest.mark.asyncio
async def test_generate_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        model_config, "OLLAMA_BASE_URL", "http://localhost:11434"
    )

    def handler(request: httpx.Request):
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)
    client = OllamaClient()
    client._client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=transport
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate("test")
