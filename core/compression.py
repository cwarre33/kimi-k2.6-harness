"""zstd compression wrapper for thought trace storage."""

import zstandard

from core.memory_config import ZSTD_COMPRESSION_LEVEL

_MAX_DECOMPRESSION_SIZE = 10 * 1024 * 1024  # 10 MiB safety limit

_COMPRESSOR = zstandard.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL)
_DECOMPRESSOR = zstandard.ZstdDecompressor()


class CompressionError(Exception):
    """Raised when compression or decompression fails."""


def compress(data: str) -> bytes:
    """Compress a string using zstd.

    Raises:
        CompressionError: If compression fails.
    """
    try:
        return _COMPRESSOR.compress(data.encode("utf-8"))
    except zstandard.ZstdError as exc:
        raise CompressionError(f"zstd compression failed: {exc}") from exc


def decompress(data: bytes) -> str:
    """Decompress zstd bytes back to a string.

    Args:
        data: zstd-compressed bytes.

    Returns:
        The original uncompressed string.

    Raises:
        CompressionError: If decompression fails or output exceeds safety limit.
    """
    try:
        raw = _DECOMPRESSOR.decompress(data, max_output_size=_MAX_DECOMPRESSION_SIZE)
    except zstandard.ZstdError as exc:
        raise CompressionError(f"zstd decompression failed: {exc}") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CompressionError(f"decompressed data is not valid utf-8: {exc}") from exc
