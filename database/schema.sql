-- ============================================================
-- Digital Attendance System — Face Enrollment Schema
-- ============================================================

CREATE TABLE IF NOT EXISTS students (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    register_number     TEXT NOT NULL UNIQUE,
    name                TEXT NOT NULL,
    department          TEXT NOT NULL,
    year                TEXT NOT NULL,
    section             TEXT NOT NULL,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id          INTEGER NOT NULL,
    embedding           BLOB NOT NULL,       -- serialized float32 vector (json or numpy bytes)
    embedding_dim       INTEGER NOT NULL,
    pose_label          TEXT,                -- front / left / right / up / down
    image_path          TEXT NOT NULL,
    quality_score       REAL,
    enrolled_at         TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_student ON face_embeddings(student_id);
CREATE INDEX IF NOT EXISTS idx_students_register_number ON students(register_number);
