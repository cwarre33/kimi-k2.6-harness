"""Tests for zstd compression wrapper."""

import secrets

import pytest

from core.compression import compress, decompress, CompressionError


def test_natural_language_roundtrip():
    """Compress and decompress natural language text."""
    original = (
        "The quick brown fox jumps over the lazy dog. In the heart of the ancient forest, "
        "where sunlight barely touched the moss-covered ground, there lived a creature of "
        "myth and legend. Travelers spoke of it in hushed tones around campfires, wondering "
        "whether the stories held any truth or were merely the product of overactive "
        "imaginations fueled by too many days on the road."
    )
    compressed = compress(original)
    assert isinstance(compressed, bytes)
    restored = decompress(compressed)
    assert restored == original


def test_compression_actually_reduces_size():
    """Large repetitive text should compress smaller than original."""
    original = "ab" * 2000
    compressed = compress(original)
    assert len(compressed) < len(original.encode("utf-8"))


def test_empty_string_roundtrip():
    """Compress and decompress an empty string."""
    compressed = compress("")
    restored = decompress(compressed)
    assert restored == ""


def test_unicode_and_emoji_roundtrip():
    """Multi-byte UTF-8 characters roundtrip correctly."""
    original = "Hello world! 🌍 ñ 中文 🔥"
    compressed = compress(original)
    restored = decompress(compressed)
    assert restored == original


def test_random_bytes_roundtrip():
    """Incompressible input still roundtrips correctly."""
    original = secrets.token_hex(1024)
    compressed = compress(original)
    restored = decompress(compressed)
    assert restored == original


def test_corrupted_data_raises_compression_error():
    """Decompressing invalid zstd data raises CompressionError."""
    with pytest.raises(CompressionError):
        decompress(b"not zstd data")
