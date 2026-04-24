import pytest
import aiosqlite
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent.parent.parent / "core" / "skill_store_schema.sql"

@pytest.mark.asyncio
async def test_schema_creates_all_tables(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(db_path) as db:
        schema = SCHEMA_PATH.read_text()
        await db.executescript(schema)
        await db.commit()

        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in await cursor.fetchall()}

    expected = {"skills", "skill_bodies", "skill_tags", "skill_retrievals"}
    assert expected.issubset(tables)
