"""MessagePack-based IPC protocol for Controller-Auditor communication."""

import msgpack
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    REASONING_PLAN = "reasoning_plan"
    TOOL_INVOCATION = "tool_invocation"
    TOOL_RESULT = "tool_result"
    CHECKPOINT = "checkpoint"
    SESSION_HEARTBEAT = "heartbeat"
    INTERVENTION = "intervention"
    STATE_QUERY = "state_query"
    LOOP_WARNING = "loop_warning"


class InterventionSeverity(Enum):
    APPROVE = "approve"
    WARN = "warn"
    HALT = "halt"
    EMERGENCY_RESET = "emergency_reset"


@dataclass
class IPCMessage:
    msg_id: str
    timestamp_ns: int
    event_type: EventType
    session_id: str
    payload: Dict[str, Any]
    target_skill_id: Optional[str] = None
    reply_to: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.event_type, str):
            self.event_type = EventType(self.event_type)


def serialize_message(msg: IPCMessage) -> bytes:
    data = asdict(msg)
    data["event_type"] = msg.event_type.value
    return msgpack.packb(data, use_bin_type=True)


def deserialize_message(data: bytes) -> IPCMessage:
    unpacked = msgpack.unpackb(data, raw=False)
    return IPCMessage(
        msg_id=unpacked["msg_id"],
        timestamp_ns=unpacked["timestamp_ns"],
        event_type=EventType(unpacked["event_type"]),
        session_id=unpacked["session_id"],
        payload=unpacked["payload"],
        target_skill_id=unpacked.get("target_skill_id"),
        reply_to=unpacked.get("reply_to"),
    )
