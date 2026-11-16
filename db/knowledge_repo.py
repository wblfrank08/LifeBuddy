"""知识库文档与分块 CRUD、全文检索（短词回退）及向量检索。

所有 SQL 语句集中在本文件；文本分块与向量化调度由 Service 层负责。
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

import config


def is_short_query(query: str) -> bool:
    """纯函数：查询词长度小于阈值时走 LIKE 兜底。"""

    return len(query.strip()) < config.SHORT_QUERY_LENGTH


def quote_fts_query(query: str) -> str:
    """纯函数：包装为 FTS5 安全的短语查询。"""

    escaped = query.strip().replace('"', '""')
    return f'"{escaped}"'


def build_fts_query(query: str) -> str:
    """纯函数：把用户问题拆成可召回的词项，用 OR 连接。

    FTS5 trigram 分词器以 3 字符建索引：整句短语查询（如 ``"Python FastAPI"``）
    几乎必然零命中，而少于 3 字符的中文词项也无法命中。这里把英文/数字词整体
    保留、中文串按 3 字滑窗切分，再以 OR 组合，保证「用完整句子提问」也能召回
    相关知识库分块。词项数量设上限，避免超长查询拖慢检索。
    """

    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query):
        if re.fullmatch(r"[A-Za-z0-9]+", token):
            if len(token) >= 2:
                terms.append(quote_fts_query(token))
        elif len(token) >= 3:
            terms.append(quote_fts_query(token))
            terms.extend(
                quote_fts_query(token[i : i + 3]) for i in range(len(token) - 2)
            )
    unique = list(dict.fromkeys(terms))[:16]
    return " OR ".join(unique) if unique else quote_fts_query(query)


class KnowledgeRepository:
    """知识库文档、分块、向量记录的数据访问对象。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------ #
    # 文档
    # ------------------------------------------------------------------ #
    @staticmethod
    def _remove_files(paths: list[str]) -> None:
        """删除已落盘的上传副本，失败不影响数据库操作结果。"""

        for raw in paths:
            try:
                Path(raw).unlink(missing_ok=True)
            except OSError:
                pass

    def _filepaths(self, document_ids: list[int]) -> list[str]:
        paths: list[str] = []
        for doc_id in document_ids:
            row = self._conn.execute(
                "SELECT filepath FROM knowledge_docs WHERE id = ?", (doc_id,)
            ).fetchone()
            if row:
                paths.append(row["filepath"])
        return paths

    def _delete_rows(self, document_ids: list[int]) -> None:
        """在同一事务内删除文档及其分块、向量（FTS 由触发器同步清理）。"""

        for doc_id in document_ids:
            self._conn.execute(
                "DELETE FROM chunks_vec WHERE chunk_id IN "
                "(SELECT id FROM knowledge_chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            self._conn.execute(
                "DELETE FROM knowledge_chunks WHERE doc_id = ?", (doc_id,)
            )
            self._conn.execute(
                "DELETE FROM knowledge_docs WHERE id = ?", (doc_id,)
            )

    def add_document(
        self, filename: str, filepath: str, content_hash: str = ""
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO knowledge_docs(filename, filepath, status, content_hash) "
            "VALUES (?, ?, 'processing', ?)",
            (filename, filepath, content_hash),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def find_document(self, filename: str) -> Optional[dict[str, Any]]:
        """按文件名查找同名资料（用于导入前判重，同名只保留一份）。"""

        row = self._conn.execute(
            "SELECT id, filename, content_hash, status, chunk_count "
            "FROM knowledge_docs WHERE filename = ? ORDER BY id DESC LIMIT 1",
            (filename,),
        ).fetchone()
        return dict(row) if row else None

    def deduplicate_documents(self) -> int:
        """清理同名重复文档，仅保留 id 最小的一条，返回删除条数。

        用于修复历史版本「重复导入」遗留的数据；本身幂等，可反复调用。
        """

        rows = self._conn.execute(
            "SELECT id FROM knowledge_docs WHERE id NOT IN "
            "(SELECT MIN(id) FROM knowledge_docs GROUP BY filename)"
        ).fetchall()
        ids = [int(row["id"]) for row in rows]
        if not ids:
            return 0

        paths = self._filepaths(ids)
        try:
            self._conn.execute("BEGIN")
            self._delete_rows(ids)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._remove_files(paths)
        return len(ids)

    def set_document_status(
        self,
        document_id: int,
        status: str,
        chunk_count: int = 0,
        error: str = "",
    ) -> None:
        self._conn.execute(
            "UPDATE knowledge_docs SET status = ?, chunk_count = ?, error = ? WHERE id = ?",
            (status, chunk_count, error, document_id),
        )
        self._conn.commit()

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, filename, filepath, status, chunk_count, error, created_at "
            "FROM knowledge_docs ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: int) -> None:
        """删除文档及其全部分块、向量（FTS 由触发器同步清理）。"""

        paths = self._filepaths([document_id])
        try:
            self._conn.execute("BEGIN")
            self._delete_rows([document_id])
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._remove_files(paths)

    # ------------------------------------------------------------------ #
    # 分块与向量
    # ------------------------------------------------------------------ #
    def add_chunk(self, document_id: int, chunk_index: int, content: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO knowledge_chunks(doc_id, chunk_index, content) VALUES (?, ?, ?)",
            (document_id, chunk_index, content),
        )
        return int(cursor.lastrowid or 0)

    def add_vector(self, chunk_id: int, embedding: bytes) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, embedding),
        )

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        """回滚未提交的入库事务（用于向量化失败时保持数据一致）。"""

        self._conn.rollback()

    def fetch_chunks(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        """按 chunk_id 批量取回分块内容，用于混合检索结果去重后补全。"""

        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"SELECT c.id, c.content, c.doc_id, d.filename "
            f"FROM knowledge_chunks AS c JOIN knowledge_docs AS d ON d.id = c.doc_id "
            f"WHERE c.id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
        return [dict(row) for row in rows]

    def count_chunks(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS total FROM knowledge_chunks").fetchone()
        return int(row["total"])

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search_by_keyword(
        self, query: str, limit: int = config.SEARCH_TOP_K
    ) -> list[dict[str, Any]]:
        """全文检索知识库分块，短词自动回退到 LIKE。"""

        keyword = query.strip()
        if not keyword:
            return []

        if is_short_query(keyword):
            rows = self._conn.execute(
                "SELECT c.id AS chunk_id, c.content, c.doc_id, d.filename "
                "FROM knowledge_chunks AS c JOIN knowledge_docs AS d ON d.id = c.doc_id "
                "WHERE c.content LIKE ? ORDER BY c.id DESC LIMIT ?",
                (f"%{keyword}%", limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT c.id AS chunk_id, c.content, c.doc_id, d.filename, "
                "bm25(chunks_fts) AS score "
                "FROM chunks_fts JOIN knowledge_chunks AS c ON c.id = chunks_fts.rowid "
                "JOIN knowledge_docs AS d ON d.id = c.doc_id "
                "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (build_fts_query(keyword), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_by_vector(
        self, embedding: bytes, limit: int = config.SEARCH_TOP_K
    ) -> list[dict[str, Any]]:
        """基于 sqlite-vec vec0 的 KNN 相似度检索。"""

        rows = self._conn.execute(
            "SELECT v.chunk_id, v.distance, c.content, c.doc_id, d.filename "
            "FROM chunks_vec AS v "
            "JOIN knowledge_chunks AS c ON c.id = v.chunk_id "
            "JOIN knowledge_docs AS d ON d.id = c.doc_id "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (embedding, limit),
        ).fetchall()
        return [dict(row) for row in rows]
