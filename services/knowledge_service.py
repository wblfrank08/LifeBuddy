"""知识库文件读取、分块、调用本地 Embedding 服务调度。

无状态工具方法（分块、编码读取、向量序列化、响应解析）使用 Function，
需要持有 Repository 与 HTTP 客户端的入库流程使用 Class。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# 开发文档 1.2：网络请求统一复用 PydanticAI / OpenAI SDK 自带的 httpx2，
# 不单独安装原版 httpx，不引入任何额外依赖。
import httpx2 as httpx

import config
from db.knowledge_repo import KnowledgeRepository


class KnowledgeServiceError(RuntimeError):
    """知识库处理失败。UI 层捕获后用彩色 ``st.toast`` 展示明确错误信息。"""


# --------------------------------------------------------------------------- #
# 无状态纯函数
# --------------------------------------------------------------------------- #
def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """按段落合并、长段落固定长度滑动窗口分块，块间保留重叠度。"""

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    paragraphs = [para.strip() for para in normalized.split("\n") if para.strip()]
    raw_chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if buffer:
                raw_chunks.append(buffer)
                buffer = ""
            start = 0
            while start < len(para):
                raw_chunks.append(para[start : start + chunk_size])
                start += chunk_size - overlap
        elif not buffer:
            buffer = para
        elif len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n{para}"
        else:
            raw_chunks.append(buffer)
            buffer = para
    if buffer:
        raw_chunks.append(buffer)

    # 相邻块补充上一段尾部，形成滑动窗口重叠
    merged: list[str] = []
    for index, piece in enumerate(raw_chunks):
        if index > 0 and overlap > 0:
            piece = f"{raw_chunks[index - 1][-overlap:]}{piece}"
        cleaned = piece.strip()
        if cleaned:
            merged.append(cleaned)
    return merged


def read_text_file(path: Path) -> str:
    """读取 .md/.txt 文本，自动兼容常见中文编码。"""

    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise KnowledgeServiceError("文件编码无法识别，请转换为 UTF-8 后重新上传")


def extract_embedding(payload: Any) -> list[float]:
    """兼容 llama.cpp /embedding 与 OpenAI /v1/embeddings 两种响应格式。"""

    candidate: Any = payload
    if isinstance(candidate, list) and candidate:
        candidate = candidate[0]
    if isinstance(candidate, dict) and "data" in candidate:
        data = candidate["data"]
        if isinstance(data, list) and data:
            candidate = data[0]
    if isinstance(candidate, dict) and "embedding" in candidate:
        candidate = candidate["embedding"]
    if isinstance(candidate, list) and candidate and isinstance(candidate[0], list):
        candidate = candidate[0]
    if not isinstance(candidate, list) or not candidate:
        raise KnowledgeServiceError("向量化服务返回的数据格式无法解析")
    try:
        return [float(value) for value in candidate]
    except (TypeError, ValueError) as exc:
        raise KnowledgeServiceError("向量化服务返回了非数值向量") from exc


def serialize_vector(values: list[float]) -> bytes:
    """复用 Streamlit 内置的 numpy，把向量序列化为 vec0 需要的 float32 字节。"""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size != config.EMBEDDING.dimension:
        raise KnowledgeServiceError(
            f"向量维度异常：期望 {config.EMBEDDING.dimension} 维，实际 {array.size} 维"
        )
    return array.tobytes()


# --------------------------------------------------------------------------- #
# 业务服务类
# --------------------------------------------------------------------------- #
class KnowledgeService:
    """知识库入库与向量化调度。通过构造函数注入 Repository。"""

    def __init__(
        self,
        knowledge_repo: KnowledgeRepository,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._repo = knowledge_repo
        self._client = http_client or httpx.Client(timeout=config.EMBEDDING.timeout)

    # ------------------------------------------------------------------ #
    # 本地 Embedding 服务调用（严禁调用远程 API）
    # ------------------------------------------------------------------ #
    def embed(self, text: str) -> list[float]:
        url = config.EMBEDDING.base_url.rstrip("/") + config.EMBEDDING.embedding_path
        try:
            response = self._client.post(url, json={"content": text})
            response.raise_for_status()
            payload = response.json()
        except httpx.ConnectError as exc:
            raise KnowledgeServiceError(
                f"无法连接本地向量化服务（{url}）。请确认 bge-small-q8-zh-v1.5.gguf "
                f"已在 {config.EMBEDDING.base_url} 启动。"
            ) from exc
        except httpx.TimeoutException as exc:
            raise KnowledgeServiceError(
                "本地向量化服务响应超时：首次请求需要加载模型，请稍后重试。"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise KnowledgeServiceError(f"本地向量化服务返回异常：{exc}") from exc
        return extract_embedding(payload)

    def embed_bytes(self, text: str) -> bytes:
        """返回可直接写入 vec0 虚表的 float32 字节串。"""

        return serialize_vector(self.embed(text))

    # ------------------------------------------------------------------ #
    # 文件入库
    # ------------------------------------------------------------------ #
    def ingest_file(self, filename: str, content_bytes: bytes) -> dict[str, Any]:
        """幂等导入：读取 -> 分块 -> 逐块本地向量化 -> 文本入 FTS / 向量入 vec0。

        同名同内容视为重复，直接跳过；同名但内容有更新时先移除旧版本再重新入库。
        这样即使 UI 层因 rerun 重复调用，也绝不会堆积重复文档。
        """

        suffix = Path(filename).suffix.lower()
        if suffix not in (".md", ".txt"):
            raise KnowledgeServiceError("仅支持上传 .md 与 .txt 格式的文件")

        safe_name = Path(filename).name or "untitled.txt"
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # 先清理历史遗留的同名重复记录，再判重（二者都幂等）
        self._repo.deduplicate_documents()
        existing = self._repo.find_document(safe_name)
        if existing:
            if (
                existing["content_hash"] == content_hash
                and existing["status"] == "ready"
            ):
                return {
                    "document_id": int(existing["id"]),
                    "filename": safe_name,
                    "chunk_count": int(existing["chunk_count"]),
                    "duplicate": True,
                }
            # 内容已更新，或上次导入失败：删除旧记录后重新入库
            self._repo.delete_document(int(existing["id"]))

        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = config.UPLOAD_DIR / f"{int(time.time() * 1000)}_{safe_name}"
        target.write_bytes(content_bytes)

        text = read_text_file(target)
        chunks = chunk_text(text)
        if not chunks:
            target.unlink(missing_ok=True)
            raise KnowledgeServiceError("文件内容为空，未产生任何可入库的文本块")

        document_id = self._repo.add_document(safe_name, str(target), content_hash)
        try:
            for index, chunk in enumerate(chunks):
                chunk_id = self._repo.add_chunk(document_id, index, chunk)
                self._repo.add_vector(chunk_id, self.embed_bytes(chunk))
            self._repo.commit()
        except KnowledgeServiceError as exc:
            self._repo.rollback()
            self._repo.set_document_status(document_id, "failed", 0, str(exc))
            raise

        self._repo.set_document_status(document_id, "ready", len(chunks))
        return {
            "document_id": document_id,
            "filename": safe_name,
            "chunk_count": len(chunks),
            "duplicate": False,
        }

    # ------------------------------------------------------------------ #
    # 查询与维护
    # ------------------------------------------------------------------ #
    def list_documents(self) -> list[dict[str, Any]]:
        return self._repo.list_documents()

    def deduplicate(self) -> int:
        """清理同名重复文档，返回删除条数（用于修复历史数据）。"""

        return self._repo.deduplicate_documents()

    def delete_document(self, document_id: int) -> None:
        self._repo.delete_document(document_id)

    def count_chunks(self) -> int:
        return self._repo.count_chunks()
