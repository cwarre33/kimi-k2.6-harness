"""Tests for the async Ollama client."""

import httpx
import pytest

from core.ollama_client import OllamaClient
from core import model_config


def test_defaults_are_sensible():
    assert model_config.OLLAMA_BASE_URL == "http://localhost:11434"
    assert model_config.OLLAMA_MODEL == "kimi-k2.6"
    assert model_config.OLLAMA_TIMEOUT_SECONDS == 300.0


def test_base_url_strips_trailing_api_segment():
    client = OllamaClient(base_url="https://ollama.com/api")

    assert client.base_url == "https://ollama.com"
    assert str(client._client.base_url) == "https://ollama.com"


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


@pytest.mark.asyncio
async def test_generate_connect_error_includes_configuration_context():
    def handler(request: httpx.Request):
        raise httpx.ConnectError("All connection attempts failed")

    transport = httpx.MockTransport(handler)
    client = OllamaClient(
        base_url="http://localhost:11434",
        model="kimi-k2.6",
        api_key="secret-token",
    )
    client._client = httpx.AsyncClient(
        base_url="http://localhost:11434",
        headers={"Authorization": "Bearer secret-token"},
        transport=transport,
    )

    with pytest.raises(ConnectionError) as exc_info:
        await client.generate("test")

    message = str(exc_info.value)
    assert "http://localhost:11434" in message
    assert "kimi-k2.6" in message
    assert "api_key_set=True" in message
    assert "OLLAMA_BASE_URL" in message
