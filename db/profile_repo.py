"""档案表 CRUD。"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional


class ProfileRepository:
    """结构化个人档案的数据访问对象。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, category: str, content: str, source: str = "chat") -> int:
        cursor = self._conn.execute(
            "INSERT INTO profiles(category, content, source) VALUES (?, ?, ?)",
            (category.strip(), content.strip(), source),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def list_all(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, category, content, source, created_at FROM profiles "
            "ORDER BY category, id"
        ).fetchall()
        return [dict(row) for row in rows]

    def list_by_category(self, category: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, category, content, source, created_at FROM profiles "
            "WHERE category = ? ORDER BY id",
            (category,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get(self, profile_id: int) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT id, category, content, source, created_at FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()
        return dict(row) if row else None

    def update(
        self, profile_id: int, category: str, content: str, source: str = "chat"
    ) -> bool:
        """更新已有档案，返回是否命中记录（不新增、不改变 created_at）。"""

        cursor = self._conn.execute(
            "UPDATE profiles SET category = ?, content = ?, source = ? WHERE id = ?",
            (category.strip(), content.strip(), source, profile_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """档案内容模糊检索（档案量小，直接使用 LIKE，天然支持中文短词）。"""

        keyword = query.strip()
        if not keyword:
            return []
        rows = self._conn.execute(
            "SELECT id, category, content, source, created_at FROM profiles "
            "WHERE content LIKE ? OR category LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{keyword}%", f"%{keyword}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, profile_id: int) -> None:
        self._conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS total FROM profiles").fetchone()
        return int(row["total"])
