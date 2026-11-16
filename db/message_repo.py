"""消息表 CRUD 及短词回退检索。

短词回退：查询词长度小于 3 时强制使用模糊匹配（LIKE），
大于等于 3 时使用 FTS5 MATCH（trigram tokenizer）。
"""

from __future__ import annotations

import sqlite3
from typing import Any

import config


def is_short_query(query: str) -> bool:
    """纯函数：判断查询词是否需要走 LIKE 兜底。"""

    return len(query.strip()) < config.SHORT_QUERY_LENGTH


def quote_fts_query(query: str) -> str:
    """纯函数：把用户输入包装为 FTS5 安全的短语查询。"""

    escaped = query.strip().replace('"', '""')
    return f'"{escaped}"'


class MessageRepository:
    """消息表数据访问对象，内部持有数据库连接，对外暴露独立 CRUD 方法。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add_message(self, session_id: str, role: str, content: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def list_messages(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, session_id, role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        # 按时间正序返回，便于直接渲染聊天记录
        return [dict(row) for row in reversed(rows)]

    def count_messages(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS total FROM messages").fetchone()
        return int(row["total"])

    def clear_session(self, session_id: str) -> None:
        """删除指定会话的全部消息（messages_fts 由 AFTER DELETE 触发器同步清理）。"""

        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def search(self, query: str, limit: int = config.SEARCH_TOP_K) -> list[dict[str, Any]]:
        """按关键词检索历史消息，短词自动回退到 LIKE。"""

        keyword = query.strip()
        if not keyword:
            return []

        if is_short_query(keyword):
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, created_at FROM messages "
                "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{keyword}%", limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT messages.id, messages.session_id, messages.role, "
                "messages.content, messages.created_at "
                "FROM messages_fts JOIN messages ON messages.id = messages_fts.rowid "
                "WHERE messages_fts MATCH ? ORDER BY bm25(messages_fts) LIMIT ?",
                (quote_fts_query(keyword), limit),
            ).fetchall()
        return [dict(row) for row in rows]
