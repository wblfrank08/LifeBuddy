"""SQLite 连接单例管理及 sqlite-vec 向量扩展加载。

连接是典型的「需要维护内部状态」对象，因此按开发文档 3.1 用 Class 封装，
并以线程安全的懒加载方式保证全局只有一个连接实例。
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Optional

import sqlite_vec

import config
from db import schema


class DatabaseConnection:
    """数据库连接的单例容器。"""

    _instance: Optional["DatabaseConnection"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @classmethod
    def get_instance(cls) -> "DatabaseConnection":
        """返回全局唯一的连接实例（首次调用时建立连接并初始化表结构）。"""

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = cls(str(config.DB_PATH))
                    instance.connect()
                    cls._instance = instance
        return cls._instance

    def connect(self) -> sqlite3.Connection:
        """建立连接：创建数据目录 -> 加载 sqlite-vec 扩展 -> 初始化表结构。"""

        if self._conn is None:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")

            # 连接建立后立即加载向量检索扩展
            conn.enable_load_extension(True)
            try:
                sqlite_vec.load(conn)
            finally:
                conn.enable_load_extension(False)

            schema.initialize_schema(conn)
            conn.commit()
            self._conn = conn
        return self._conn

    @property
    def connection(self) -> sqlite3.Connection:
        """供各 Repository 使用的原始连接。"""

        return self.connect()
