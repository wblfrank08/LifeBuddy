"""建表语句、FTS5 触发器与向量虚表初始化。

本文件是 SQL 语句的唯一来源之一（另一处为各 ``*_repo.py``）。
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

import config

# --------------------------------------------------------------------------- #
# 基础表
# --------------------------------------------------------------------------- #
DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT    NOT NULL,
        role       TEXT    NOT NULL,
        content    TEXT    NOT NULL,
        created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)",
    """
    CREATE TABLE IF NOT EXISTS profiles (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        category   TEXT    NOT NULL,
        content    TEXT    NOT NULL,
        source     TEXT    NOT NULL DEFAULT 'chat',
        created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_profiles_category ON profiles(category, id)",
    """
    CREATE TABLE IF NOT EXISTS documents (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        title      TEXT    NOT NULL,
        doc_type   TEXT    NOT NULL DEFAULT 'resume',
        content    TEXT    NOT NULL,
        created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_docs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        filename     TEXT    NOT NULL,
        filepath     TEXT    NOT NULL,
        status       TEXT    NOT NULL DEFAULT 'processing',
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        error        TEXT    NOT NULL DEFAULT '',
        content_hash TEXT    NOT NULL DEFAULT '',
        created_at   TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_chunks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id      INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        content     TEXT    NOT NULL,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON knowledge_chunks(doc_id, id)",
)

# --------------------------------------------------------------------------- #
# 全文检索虚表（FTS5 trigram tokenizer：原生支持中英文子串搜索）
# --------------------------------------------------------------------------- #
FTS_STATEMENTS: tuple[str, ...] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content,
        content='messages',
        content_rowid='id',
        tokenize='trigram'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content,
        content='knowledge_chunks',
        content_rowid='id',
        tokenize='trigram'
    )
    """,
    # 外部内容表需要显式维护索引；INSERT/DELETE 触发器保证 FTS 与基础表同步
    """
    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON knowledge_chunks BEGIN
        INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON knowledge_chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    END
    """,
)

# --------------------------------------------------------------------------- #
# 向量检索虚表（sqlite-vec vec0）
# --------------------------------------------------------------------------- #
def vector_table_ddl() -> str:
    """按配置维度生成 vec0 虚表建表语句。"""

    return f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
        chunk_id  INTEGER PRIMARY KEY,
        embedding FLOAT[{config.EMBEDDING.dimension}]
    )
    """


def all_statements() -> Iterable[str]:
    """按依赖顺序返回全部建表/触发器语句。"""

    yield from DDL_STATEMENTS
    yield from FTS_STATEMENTS
    yield vector_table_ddl()


def ensure_column(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    """幂等迁移：为旧数据库补充新增列（CREATE TABLE IF NOT EXISTS 不会改结构）。"""

    existing = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def initialize_schema(conn: sqlite3.Connection) -> None:
    """在给定连接上完成全部建表、扩展虚表初始化与旧库迁移。"""

    for statement in all_statements():
        conn.execute(statement)
    # 旧版本数据库没有内容指纹列，补上后才能按文件内容判重
    ensure_column(
        conn,
        "knowledge_docs",
        "content_hash",
        "content_hash TEXT NOT NULL DEFAULT ''",
    )
