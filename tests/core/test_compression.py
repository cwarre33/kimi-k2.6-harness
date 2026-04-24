"""Tests for core.compression module."""

import pytest

from core.compression import compress, decompress


NATURAL_LANGUAGE_TEXT = """
The quick brown fox jumps over the lazy dog. In the heart of the ancient forest,
where sunlight barely touched the moss-covered ground, there lived a creature of
myth and legend. Travelers spoke of it in hushed tones around campfires, wondering
whether the stories held any truth or were merely the product of overactive
imaginations fueled by too many days on the road.
"""


class TestCompressionRoundtrip:
    def test_natural_language_roundtrip(self):
        compressed = compress(NATURAL_LANGUAGE_TEXT)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        assert len(compressed) < len(NATURAL_LANGUAGE_TEXT.encode("utf-8"))
        decompressed = decompress(compressed)
        assert decompressed == NATURAL_LANGUAGE_TEXT

    def test_empty_string_roundtrip(self):
        compressed = compress("")
        assert isinstance(compressed, bytes)
        decompressed = decompress(compressed)
        assert decompressed == ""
