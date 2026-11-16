"""文档表 CRUD（生成的简历、总结等）。"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional


class DocumentRepository:
    """生成文档的数据访问对象。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, title: str, content: str, doc_type: str = "resume") -> int:
        cursor = self._conn.execute(
            "INSERT INTO documents(title, content, doc_type) VALUES (?, ?, ?)",
            (title, content, doc_type),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def list_all(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, title, doc_type, created_at, "
            "LENGTH(content) AS content_length FROM documents ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def get(self, document_id: int) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT id, title, doc_type, content, created_at FROM documents "
            "WHERE id = ?",
            (document_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def delete(self, document_id: int) -> None:
        self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self._conn.commit()
