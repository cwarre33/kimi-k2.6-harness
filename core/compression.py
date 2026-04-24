"""zstd compression wrapper for thought trace storage."""

import zstandard

from core.memory_config import ZSTD_COMPRESSION_LEVEL


def compress(data: str) -> bytes:
    """Compress a string using zstd."""
    compressor = zstandard.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL)
    return compressor.compress(data.encode("utf-8"))


def decompress(data: bytes) -> str:
    """Decompress zstd bytes back to a string."""
    decompressor = zstandard.ZstdDecompressor()
    return decompressor.decompress(data).decode("utf-8")
