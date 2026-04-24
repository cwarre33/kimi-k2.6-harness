import pytest
import aiosqlite
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent.parent.parent / "core" / "skill_store_schema.sql"

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
