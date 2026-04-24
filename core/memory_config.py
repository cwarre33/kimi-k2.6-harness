from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryConfig:
    ZSTD_COMPRESSION_LEVEL: int = 3
    MAX_FAILURES_BEFORE_DEPRECATION: int = 3
    DEPRECATED_VACUUM_MAX_AGE_DAYS: int = 30
    DEFAULT_TOP_K_RETRIEVAL: int = 3
    MIN_SUCCESS_COUNT_FOR_RETRIEVAL: int = 1
    STALENESS_HALF_LIFE_DAYS: float = 7.0
    AUDIT_LOG_RETENTION_DAYS: int = 90

CONFIG = MemoryConfig()
