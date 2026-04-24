"""Async IPC client/server with Unix domain sockets and TCP fallback."""

import asyncio
import os
import socket
import struct
from enum import Enum
from typing import Optional, Set

from core.ipc_config import (
    MAX_MESSAGE_SIZE_BYTES,
    SOCKET_TIMEOUT_SECONDS,
    TCP_FALLBACK_HOST,
    TCP_FALLBACK_PORT,
)
from core.ipc_protocol import (
    EventType,
    IPCMessage,
    deserialize_message,
    serialize_message,
)


class IPCRole(Enum):
    CONTROLLER = "controller"
    AUDITOR = "auditor"


class IPCBus:
    """Async IPC bus with message framing and TCP fallback."""

    def __init__(self, role: IPCRole, socket_path: Optional[str] = None):
        self.role = role
        self.socket_path = socket_path
        self._message_queue: asyncio.Queue[IPCMessage] = asyncio.Queue()
        self._intervention_queue: asyncio.Queue[IPCMessage] = asyncio.Queue()
        self._server: Optional[asyncio.Server] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._client_writers: Set[asyncio.StreamWriter] = set()
        self._listen_task: Optional[asyncio.Task] = None
        self._tasks: Set[asyncio.Task] = set()
        self._connected = False
        self._server_running = False
        self._tcp_address: Optional[tuple[str, int]] = None

    async def start_server(self) -> Optional[str | tuple[str, int]]:
        """Start the IPC server.

        Returns the bound address (socket path or TCP address tuple).
        """
        if self._server_running:
            return self.socket_path or self._tcp_address

        # Try Unix domain socket first
        if self.socket_path and hasattr(socket, "AF_UNIX"):
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            try:
                self._server = await asyncio.start_unix_server(
                    self._handle_client,
                    path=self.socket_path,
                )
                self._server_running = True
                return self.socket_path
            except (NotImplementedError, OSError, AttributeError):
                pass  # Fall through to TCP fallback

        # TCP fallback
        self._server = await asyncio.start_server(
            self._handle_client,
            host=TCP_FALLBACK_HOST,
            port=TCP_FALLBACK_PORT,
        )
        self._server_running = True
        self._tcp_address = self._server.sockets[0].getsockname()[:2]
        return self._tcp_address

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle an incoming client connection."""
        self._client_writers.add(writer)
        try:
            await self._read_loop(reader)
        finally:
            self._client_writers.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        """Read framed messages from a stream."""
        while True:
            try:
                msg = await self._read_one_message(reader)
                if msg is None:
                    break
                await self._message_queue.put(msg)
                if msg.event_type == EventType.INTERVENTION:
                    await self._intervention_queue.put(msg)
            except asyncio.IncompleteReadError:
                break
            except ConnectionResetError:
                break
            except Exception:
                break

    async def _read_one_message(
        self, reader: asyncio.StreamReader
    ) -> Optional[IPCMessage]:
        """Read a single framed message."""
        try:
            length_bytes = await asyncio.wait_for(
                reader.readexactly(4), timeout=SOCKET_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            return None
        (length,) = struct.unpack(">I", length_bytes)
        if length > MAX_MESSAGE_SIZE_BYTES:
            raise ValueError(f"Message size {length} exceeds maximum")
        payload = await reader.readexactly(length)
        return deserialize_message(payload)

    async def connect(self, address: Optional[str | tuple[str, int]] = None) -> None:
        """Connect to an IPC server."""
        if self._connected:
            return

        target = address or self.socket_path
        if target is None:
            raise ValueError("No address provided and no socket_path configured")

        if isinstance(target, str):
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(
                    path=target
                )
            except (NotImplementedError, OSError, AttributeError) as exc:
                raise ConnectionError(
                    f"Cannot connect to Unix socket {target}: {exc}"
                )
            self._connected = True
            self._listen_task = asyncio.create_task(self._read_loop(self._reader))
            self._listen_task.add_done_callback(self._tasks.discard)
            self._tasks.add(self._listen_task)
            return

        # TCP connection
        host, port = target
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=SOCKET_TIMEOUT_SECONDS,
        )
        self._connected = True
        self._listen_task = asyncio.create_task(self._read_loop(self._reader))
        self._listen_task.add_done_callback(self._tasks.discard)
        self._tasks.add(self._listen_task)

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        self._connected = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

    async def stop_server(self) -> None:
        """Stop the server."""
        self._server_running = False

        for writer in list(self._client_writers):
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        self._client_writers.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self.socket_path and os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    async def send_message(self, msg: IPCMessage) -> None:
        """Send a framed message."""
        data = serialize_message(msg)
        header = struct.pack(">I", len(data))
        frame = header + data

        if self._writer is not None:
            self._writer.write(frame)
            await self._writer.drain()
        elif self._client_writers:
            for writer in list(self._client_writers):
                writer.write(frame)
                await writer.drain()
        else:
            raise ConnectionError("Not connected")

    async def iter_messages(self):
        """Async generator yielding consumed messages from the queue."""
        while True:
            msg = await self._message_queue.get()
            yield msg

    async def receive_intervention(self) -> Optional[IPCMessage]:
        """Wait for an intervention message.

        Returns the intervention message, or None if the wait times out.
        """
        try:
            return await asyncio.wait_for(
                self._intervention_queue.get(), timeout=SOCKET_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            return None
