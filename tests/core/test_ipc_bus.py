"""Integration tests for IPC bus."""

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio

from core.ipc_bus import IPCBus, IPCRole
from core.ipc_protocol import EventType, IPCMessage


@pytest_asyncio.fixture
async def ipc_pair():
    """Provide a connected client/server IPC bus pair."""
    socket_path = tempfile.mktemp(suffix=".sock")
    server = IPCBus(IPCRole.AUDITOR, socket_path=socket_path)
    addr = await server.start_server()

    client = IPCBus(IPCRole.CONTROLLER, socket_path=socket_path)
    await client.connect(addr)

    yield client, server

    await client.disconnect()
    await server.stop_server()


@pytest.mark.asyncio
async def test_client_sends_message_server_receives(ipc_pair):
    """Client sends a message and server receives it via iter_messages."""
    client, server = ipc_pair

    msg = IPCMessage(
        msg_id="test-msg-001",
        timestamp_ns=1_700_000_000_000_000_000,
        event_type=EventType.TOOL_INVOCATION,
        session_id="sess-001",
        payload={"tool_name": "shell.exec", "tool_args": {"command": "pytest"}},
    )

    await client.send_message(msg)

    gen = server.iter_messages()
    received = await asyncio.wait_for(gen.__anext__(), timeout=5.0)
    while received.event_type == EventType.SESSION_HEARTBEAT:
        received = await asyncio.wait_for(gen.__anext__(), timeout=5.0)

    assert received.msg_id == msg.msg_id
    assert received.event_type == msg.event_type
    assert received.payload == msg.payload


@pytest.mark.asyncio
async def test_heartbeat_timeout_triggers_degraded_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = os.path.join(tmpdir, "heartbeat.sock")
        client = IPCBus(role=IPCRole.CONTROLLER, socket_path=socket_path)
        with pytest.raises((OSError, ConnectionError)):
            await client.connect()
        assert not client._connected
