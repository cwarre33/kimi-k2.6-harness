CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    embedding_model_version TEXT NOT NULL DEFAULT 'kimi-k2.6-embedding-001',
    skill_schema_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_validated_at TIMESTAMP,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    is_deprecated BOOLEAN DEFAULT 0,
    deprecation_reason TEXT,
    thought_trace_hash TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS skill_bodies (
    skill_id TEXT PRIMARY KEY,
    task_pattern TEXT NOT NULL,
    tool_sequence TEXT NOT NULL,
    thought_trace_compressed BLOB,
    code_artifact TEXT,
    context_requirements TEXT,
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS skill_tags (
    skill_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    PRIMARY KEY (skill_id, tag),
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS skill_retrievals (
    retrieval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query_embedding_id TEXT,
    rank_position INTEGER NOT NULL,
    was_accepted BOOLEAN NOT NULL,
    execution_time_ms INTEGER,
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
);

CREATE INDEX IF NOT EXISTS idx_skills_deprecated ON skills(is_deprecated, failure_count);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON skill_tags(tag);
CREATE INDEX IF NOT EXISTS idx_skills_validated ON skills(last_validated_at);
