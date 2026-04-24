"""Tests for MessagePack IPC protocol."""

from core.ipc_protocol import (
    IPCMessage,
    EventType,
    InterventionSeverity,
    serialize_message,
    deserialize_message,
)


def test_serialize_deserialize_roundtrip():
    msg = IPCMessage(
        msg_id="test-msg-001",
        timestamp_ns=1_700_000_000_000_000_000,
        event_type=EventType.TOOL_INVOCATION,
        session_id="sess-001",
        payload={"tool_name": "shell.exec", "tool_args": {"command": "pytest"}},
        target_skill_id=None,
        reply_to=None,
    )
    serialized = serialize_message(msg)
    assert isinstance(serialized, bytes)
    restored = deserialize_message(serialized)
    assert restored.msg_id == msg.msg_id
    assert restored.timestamp_ns == msg.timestamp_ns
    assert restored.event_type == msg.event_type
    assert restored.payload == msg.payload


def test_intervention_message_roundtrip():
    msg = IPCMessage(
        msg_id="intervention-001",
        timestamp_ns=1_700_000_000_000_000_001,
        event_type=EventType.INTERVENTION,
        session_id="sess-001",
        payload={
            "severity": InterventionSeverity.HALT.value,
            "trigger": "hallucination_detected",
            "trigger_details": {"confidence": 0.94},
            "suggested_correction": {"corrected_args": {"command": "pytest -v"}},
            "resume_requires_ack": True,
        },
    )
    serialized = serialize_message(msg)
    restored = deserialize_message(serialized)
    assert restored.payload["severity"] == InterventionSeverity.HALT.value
    assert restored.payload["resume_requires_ack"] is True
