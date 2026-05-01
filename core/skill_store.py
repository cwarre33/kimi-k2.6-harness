"""Async SQLite skill store with zstd-compressed traces and staleness detection."""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from core.compression import compress
from core.memory_config import (
    DEPRECATED_VACUUM_MAX_AGE_DAYS,
    DEFAULT_TOP_K_RETRIEVAL,
    MAX_FAILURES_BEFORE_DEPRECATION,
    MIN_SUCCESS_COUNT_FOR_RETRIEVAL,
)


class SkillMatch:
    """A retrieved skill with metadata."""

    __slots__ = (
        "skill_id",
        "canonical_name",
        "task_pattern",
        "tool_sequence",
        "thought_trace_compressed",
        "relevance_score",
        "last_validated_at",
        "success_count",
        "failure_count",
    )

    def __init__(
        self,
        skill_id: str,
        canonical_name: str,
        task_pattern: str,
        tool_sequence: List[Dict[str, Any]],
        thought_trace_compressed: bytes,
        relevance_score: float,
        last_validated_at: Optional[datetime],
        success_count: int,
        failure_count: int,
    ):
        self.skill_id = skill_id
        self.canonical_name = canonical_name
        self.task_pattern = task_pattern
        self.tool_sequence = tool_sequence
        self.thought_trace_compressed = thought_trace_compressed
        self.relevance_score = relevance_score
        self.last_validated_at = last_validated_at
        self.success_count = success_count
        self.failure_count = failure_count


class SkillHashCollision(Exception):
    """Raised when a duplicate thought trace hash is detected."""


class SkillNotFound(Exception):
    """Raised when a requested skill does not exist."""


class SqliteSkillStore:
    """Persistent, semantically-retrievable skill storage backed by SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Open the database and apply the schema."""
        self._db = await aiosqlite.connect(self.db_path)
        cursor = await self._db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        db_version = row[0] if row else 0
        if db_version > 1:
            raise RuntimeError(
                f"Database schema version {db_version} exceeds harness supported version 1"
            )
        schema_path = Path(__file__).parent / "skill_store_schema.sql"
        await self._db.executescript(schema_path.read_text())
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def store_skill(
        self,
        canonical_name: str,
        task_pattern: str,
        tool_sequence: List[Dict[str, Any]],
        thought_trace: str,
        code_artifact: Optional[str],
        context_requirements: Dict[str, Any],
        tags: List[str],
        embedding_vector: Optional[List[float]] = None,
    ) -> str:
        """Persist a new skill or return the existing id on hash collision."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        trace_hash = hashlib.sha256(thought_trace.encode("utf-8")).hexdigest()

        cursor = await self._db.execute(
            "SELECT skill_id FROM skills WHERE thought_trace_hash = ?",
            (trace_hash,),
        )
        row = await cursor.fetchone()
        if row:
            existing_id = row[0]
            await self.validate_skill(existing_id, "success", session_id="dedup")
            return existing_id

        skill_id = str(uuid.uuid4())
        compressed_trace = compress(thought_trace)

        await self._db.execute(
            """
            INSERT INTO skills (skill_id, canonical_name, thought_trace_hash)
            VALUES (?, ?, ?)
            """,
            (skill_id, canonical_name, trace_hash),
        )

        await self._db.execute(
            """
            INSERT INTO skill_bodies (skill_id, task_pattern, tool_sequence,
                                      thought_trace_compressed, code_artifact,
                                      context_requirements)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                skill_id,
                task_pattern,
                json.dumps(tool_sequence),
                compressed_trace,
                code_artifact,
                json.dumps(context_requirements),
            ),
        )

        for tag in tags:
            await self._db.execute(
                "INSERT INTO skill_tags (skill_id, tag) VALUES (?, ?)",
                (skill_id, tag),
            )

        await self._db.commit()
        return skill_id

    async def retrieve_skills(
        self,
        query_text: str,
        query_embedding: Optional[List[float]],
        context_filter: Optional[Dict[str, Any]] = None,
        top_k: int = DEFAULT_TOP_K_RETRIEVAL,
        min_success_count: int = MIN_SUCCESS_COUNT_FOR_RETRIEVAL,
        exclude_deprecated: bool = True,
    ) -> List[SkillMatch]:
        """Retrieve top-k relevant skills ordered by composite score."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        conditions = ["s.success_count >= ?"]
        params: List[Any] = [min_success_count]

        if exclude_deprecated:
            conditions.append("s.is_deprecated = 0")

        where_clause = " AND ".join(conditions)
        like_pattern = f"%{query_text}%"

        query = f"""
            SELECT s.skill_id, s.canonical_name, s.success_count,
                   s.failure_count, s.last_validated_at,
                   b.task_pattern, b.tool_sequence, b.thought_trace_compressed
            FROM skills s
            JOIN skill_bodies b ON s.skill_id = b.skill_id
            JOIN skill_tags t ON s.skill_id = t.skill_id
            WHERE {where_clause}
              AND (b.task_pattern LIKE ? OR t.tag LIKE ?)
            GROUP BY s.skill_id
            ORDER BY s.success_count DESC, s.last_validated_at DESC
            LIMIT ?
        """

        params.extend([like_pattern, like_pattern, top_k])
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()

        matches: List[SkillMatch] = []
        for row in rows:
            (
                skill_id,
                canonical_name,
                success_count,
                failure_count,
                last_validated_at,
                task_pattern,
                tool_sequence,
                trace_compressed,
            ) = row
            relevance = float(success_count) / (1.0 + float(failure_count))
            matches.append(
                SkillMatch(
                    skill_id=skill_id,
                    canonical_name=canonical_name,
                    task_pattern=task_pattern,
                    tool_sequence=json.loads(tool_sequence),
                    thought_trace_compressed=trace_compressed,
                    relevance_score=relevance,
                    last_validated_at=last_validated_at,
                    success_count=success_count,
                    failure_count=failure_count,
                )
            )

        return matches

    async def validate_skill(
        self, skill_id: str, outcome: str, session_id: str
    ) -> None:
        """Update success/failure counters and trigger deprecation."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        cursor = await self._db.execute(
            "SELECT skill_id FROM skills WHERE skill_id = ?", (skill_id,)
        )
        if not await cursor.fetchone():
            raise SkillNotFound(f"Skill {skill_id} not found")

        if outcome == "success":
            await self._db.execute(
                """
                UPDATE skills
                SET success_count = success_count + 1,
                    last_validated_at = CURRENT_TIMESTAMP
                WHERE skill_id = ?
                """,
                (skill_id,),
            )
        else:
            await self._db.execute(
                "UPDATE skills SET failure_count = failure_count + 1 WHERE skill_id = ?",
                (skill_id,),
            )
            await self._db.execute(
                """
                UPDATE skills
                SET is_deprecated = 1,
                    deprecation_reason = 'Too many failures'
                WHERE skill_id = ?
                  AND failure_count >= ?
                """,
                (skill_id, MAX_FAILURES_BEFORE_DEPRECATION),
            )

        await self._db.execute(
            """
            INSERT INTO skill_retrievals
                (skill_id, session_id, rank_position, was_accepted)
            VALUES (?, ?, ?, ?)
            """,
            (skill_id, session_id, 0, outcome == "success"),
        )
        await self._db.commit()

    async def get_skill_by_hash(
        self, thought_trace_hash: str
    ) -> Optional[SkillMatch]:
        """Deduplication check before storage."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        cursor = await self._db.execute(
            """
            SELECT s.skill_id, s.canonical_name, s.success_count,
                   s.failure_count, s.last_validated_at,
                   b.task_pattern, b.tool_sequence, b.thought_trace_compressed
            FROM skills s
            JOIN skill_bodies b ON s.skill_id = b.skill_id
            WHERE s.thought_trace_hash = ?
            """,
            (thought_trace_hash,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        (
            skill_id,
            canonical_name,
            success_count,
            failure_count,
            last_validated_at,
            task_pattern,
            tool_sequence,
            trace_compressed,
        ) = row
        return SkillMatch(
            skill_id=skill_id,
            canonical_name=canonical_name,
            task_pattern=task_pattern,
            tool_sequence=json.loads(tool_sequence),
            thought_trace_compressed=trace_compressed,
            relevance_score=float(success_count) / (1.0 + float(failure_count)),
            last_validated_at=last_validated_at,
            success_count=success_count,
            failure_count=failure_count,
        )

    async def vacuum_deprecated(
        self, max_age_days: int = DEPRECATED_VACUUM_MAX_AGE_DAYS
    ) -> int:
        """Hard-delete deprecated skills older than max_age_days."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        cursor = await self._db.execute(
            """
            DELETE FROM skills
            WHERE is_deprecated = 1
              AND julianday('now') - julianday(created_at) > ?
            """,
            (max_age_days,),
        )
        await self._db.commit()
        return cursor.rowcount
