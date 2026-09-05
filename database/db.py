"""
database/db.py
--------------
SQLite access layer for the face-enrollment module of the Digital Attendance
System. Keeps all SQL in one place so the rest of the backend never writes
raw queries.
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def init_db() -> None:
    """Create the database file and tables if they don't already exist."""
    with get_connection() as conn:
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())
        conn.commit()


@contextmanager
def get_connection():
    """Context-managed SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------- students

def get_student_by_register_number(register_number: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE register_number = ?",
            (register_number,),
        ).fetchone()
        return dict(row) if row else None


def create_student(name: str, register_number: str, department: str,
                    year: str, section: str) -> int:
    """Insert a new student record. Returns the new student's id.
    Raises sqlite3.IntegrityError if register_number already exists."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO students (name, register_number, department, year, section)
               VALUES (?, ?, ?, ?, ?)""",
            (name, register_number, department, year, section),
        )
        conn.commit()
        return cur.lastrowid


def delete_student(student_id: int) -> None:
    """Removes a student and (via cascade) their embeddings."""
    with get_connection() as conn:
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()


# ---------------------------------------------------------- face embeddings

def save_face_embedding(student_id: int, embedding: List[float], image_path: str,
                         pose_label: str = "", quality_score: float = 0.0) -> int:
    """Persist one face embedding vector for a student."""
    blob = json.dumps(embedding).encode("utf-8")
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO face_embeddings
               (student_id, embedding, embedding_dim, pose_label, image_path, quality_score)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (student_id, blob, len(embedding), pose_label, image_path, quality_score),
        )
        conn.commit()
        return cur.lastrowid


def get_all_embeddings() -> List[Dict[str, Any]]:
    """Returns every stored embedding together with its owning student.
    Used for duplicate-face checks against the whole enrolled population."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT fe.id, fe.student_id, fe.embedding, fe.pose_label,
                      s.name, s.register_number
               FROM face_embeddings fe
               JOIN students s ON s.id = fe.student_id"""
        ).fetchall()

    result = []
    for row in rows:
        result.append({
            "embedding_id": row["id"],
            "student_id": row["student_id"],
            "name": row["name"],
            "register_number": row["register_number"],
            "pose_label": row["pose_label"],
            "embedding": json.loads(row["embedding"].decode("utf-8")),
        })
    return result


def get_embeddings_for_student(student_id: int) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM face_embeddings WHERE student_id = ?",
            (student_id,),
        ).fetchall()
    return [dict(r) for r in rows]
