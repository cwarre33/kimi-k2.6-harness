import hashlib
import pytest
import pytest_asyncio
import aiosqlite
from pathlib import Path

from core.skill_store import SqliteSkillStore

SCHEMA_PATH = Path(__file__).parent.parent.parent / "core" / "skill_store_schema.sql"

@pytest_asyncio.fixture
async def skill_store(tmp_path):
    db_path = tmp_path / "skills.db"
    store = SqliteSkillStore(str(db_path))
    await store.initialize()
    yield store
    await store.close()

@pytest.mark.asyncio
async def test_schema_creates_all_tables(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        schema = SCHEMA_PATH.read_text()
        await db.executescript(schema)
        await db.commit()

        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in await cursor.fetchall()}

    expected = {"skills", "skill_bodies", "skill_tags", "skill_retrievals"}
    assert expected.issubset(tables)

@pytest.mark.asyncio
async def test_schema_creates_all_indexes(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        schema = SCHEMA_PATH.read_text()
        await db.executescript(schema)
        await db.commit()

        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row[0] for row in await cursor.fetchall()}

    expected = {
        "idx_retrievals_skill_id",
        "idx_skills_deprecated",
        "idx_tags_tag",
        "idx_skills_validated",
    }
    assert expected.issubset(indexes)

@pytest.mark.asyncio
async def test_foreign_key_cascade_delete(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        schema = SCHEMA_PATH.read_text()
        await db.executescript(schema)
        await db.commit()

        await db.execute(
            "INSERT INTO skills (skill_id, canonical_name) VALUES ('skill-1', 'Test Skill')"
        )
        await db.execute(
            "INSERT INTO skill_bodies (skill_id, task_pattern, tool_sequence) VALUES ('skill-1', 'pattern', '[]')"
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT 1 FROM skill_bodies WHERE skill_id = 'skill-1'"
        )
        assert await cursor.fetchone() is not None

        await db.execute("DELETE FROM skills WHERE skill_id = 'skill-1'")
        await db.commit()

        cursor = await db.execute(
            "SELECT 1 FROM skill_bodies WHERE skill_id = 'skill-1'"
        )
        assert await cursor.fetchone() is None


# ---- Task 3 tests ----

@pytest.mark.asyncio
async def test_store_and_retrieve_skill(skill_store):
    skill_id = await skill_store.store_skill(
        canonical_name="test-skill",
        task_pattern="Run pytest on modified files",
        tool_sequence=[{"tool": "shell.exec", "args": {"command": "pytest -v"}}],
        thought_trace="I need to run tests to verify the changes.",
        code_artifact=None,
        context_requirements={"dependencies": ["pytest"]},
        tags=["testing", "pytest"],
    )

    assert isinstance(skill_id, str)
    assert len(skill_id) > 0

    matches = await skill_store.retrieve_skills(
        query_text="pytest",
        query_embedding=None,
        top_k=3,
    )

    assert len(matches) == 1
    assert matches[0].canonical_name == "test-skill"
    assert matches[0].success_count == 0


@pytest.mark.asyncio
async def test_validate_skill_updates_counters(skill_store):
    skill_id = await skill_store.store_skill(
        canonical_name="validate-test",
        task_pattern="Test validation",
        tool_sequence=[],
        thought_trace="trace",
        code_artifact=None,
        context_requirements={},
        tags=["test"],
    )

    await skill_store.validate_skill(skill_id, "success", session_id="sess-1")
    await skill_store.validate_skill(skill_id, "failure", session_id="sess-2")

    matches = await skill_store.retrieve_skills(
        query_text="test",
        query_embedding=None,
        top_k=3,
    )

    assert matches[0].success_count == 1
    assert matches[0].failure_count == 1


# ---- Task 4 tests ----

@pytest.mark.asyncio
async def test_vacuum_removes_old_deprecated_skills(skill_store):
    old_skill_id = await skill_store.store_skill(
        canonical_name="old-skill",
        task_pattern="Old pattern",
        tool_sequence=[],
        thought_trace="old trace",
        code_artifact=None,
        context_requirements={},
        tags=["old"],
    )

    await skill_store.validate_skill(old_skill_id, "failure", session_id="s1")
    await skill_store.validate_skill(old_skill_id, "failure", session_id="s2")
    await skill_store.validate_skill(old_skill_id, "failure", session_id="s3")

    await skill_store._db.execute(
        "UPDATE skills SET created_at = datetime('now', '-31 days') WHERE skill_id = ?",
        (old_skill_id,),
    )
    await skill_store._db.commit()

    deleted_count = await skill_store.vacuum_deprecated(max_age_days=30)
    assert deleted_count == 1

    result = await skill_store.get_skill_by_hash(
        hashlib.sha256("old trace".encode("utf-8")).hexdigest()
    )
    assert result is None


# ---- Task 5 tests ----

@pytest.mark.asyncio
async def test_schema_version_read(skill_store):
    cursor = await skill_store._db.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_schema_version_mismatch_raises(tmp_path):
    db_path = tmp_path / "skills.db"
    store = SqliteSkillStore(str(db_path))
    store._db = await aiosqlite.connect(str(db_path))
    await store._db.execute("PRAGMA user_version = 2")
    await store._db.commit()
    await store._db.close()
    store._db = None

    with pytest.raises(RuntimeError, match="Database schema version 2 exceeds harness supported version 1"):
        await store.initialize()

    await store.close()
