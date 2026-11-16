"""混合检索策略调度：组合关键词全文检索与向量相似度检索。"""

from __future__ import annotations

from typing import Any

import config
from db.knowledge_repo import KnowledgeRepository
from db.message_repo import MessageRepository
from services.knowledge_service import KnowledgeService, KnowledgeServiceError

RRF_K = 60  # Reciprocal Rank Fusion 的平滑常数


def _merge_chunks(
    keyword_chunks: list[dict[str, Any]],
    vector_chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """纯函数：RRF 合并两路结果、按 chunk_id 去重并重排序。"""

    scores: dict[int, float] = {}
    payloads: dict[int, dict[str, Any]] = {}

    for rank, item in enumerate(keyword_chunks, start=1):
        chunk_id = int(item["chunk_id"])
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        payloads.setdefault(chunk_id, dict(item))

    for rank, item in enumerate(vector_chunks, start=1):
        chunk_id = int(item["chunk_id"])
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        payloads.setdefault(chunk_id, dict(item))

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    merged: list[dict[str, Any]] = []
    for chunk_id in ordered_ids:
        item = payloads[chunk_id]
        item["score"] = round(scores[chunk_id], 6)
        # 裁剪上下文，避免超长文本挤爆本地小模型窗口
        item["content"] = str(item.get("content", ""))[: config.TOOL_CONTEXT_MAX_CHARS]
        merged.append(item)
    return merged


class SearchService:
    """同时发起关键词与向量检索，合并去重重排序后返回。"""

    def __init__(
        self,
        message_repo: MessageRepository,
        knowledge_repo: KnowledgeRepository,
        knowledge_service: KnowledgeService,
    ) -> None:
        self._message_repo = message_repo
        self._knowledge_repo = knowledge_repo
        self._knowledge_service = knowledge_service

    def hybrid_search(
        self,
        query: str,
        top_k: int = config.SEARCH_TOP_K,
        use_vector: bool = True,
    ) -> dict[str, Any]:
        """返回历史消息命中、知识库分块命中，以及向量检索的降级提示。

        ``use_vector=False`` 供每轮对话的**确定性预检索**使用：只走 FTS5/LIKE
        关键词，零网络依赖，避免本地 Embedding 服务未启动时每轮等待超时。
        """

        keyword_messages = self._message_repo.search(query, top_k)
        keyword_chunks = self._knowledge_repo.search_by_keyword(query, top_k)

        vector_chunks: list[dict[str, Any]] = []
        vector_error = ""
        if use_vector:
            try:
                embedding = self._knowledge_service.embed_bytes(query)
                vector_chunks = self._knowledge_repo.search_by_vector(embedding, top_k)
            except KnowledgeServiceError as exc:
                # 仅占位配置下向量服务可能未启动：降级为纯关键词检索，不中断问答
                vector_error = str(exc)

        chunks = _merge_chunks(keyword_chunks, vector_chunks, top_k)
        return {
            "messages": keyword_messages,
            "chunks": chunks,
            "vector_error": vector_error,
        }

    @staticmethod
    def format_context(result: dict[str, Any]) -> str:
        """把检索结果转化为给模型阅读的自然语言上下文（纯数据转化）。

        知识库文件清单放在**首行**：既是模型标注来源时唯一可引用的名称，
        又不会被尾部 ``TOOL_CONTEXT_MAX_CHARS`` 裁剪掉。
        """

        lines: list[str] = []
        chunks = result.get("chunks", [])
        sources = list(
            dict.fromkeys(
                str(item.get("filename", "")).strip()
                for item in chunks
                if str(item.get("filename", "")).strip()
            )
        )
        if sources:
            lines.append(f"【本轮检索到的知识库文件】{'、'.join(sources)}")
            lines.append("（以下资料均来自上述文件；回答只能依据这些资料，不得使用通用知识。）")
        for message in result["messages"]:
            lines.append(f"[历史对话·{message['role']}] {message['content']}")
        for index, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[资料{index}·{chunk.get('filename', '知识库')}] {chunk['content']}"
            )
        if not lines:
            return "未检索到相关资料。"
        if result.get("vector_error"):
            lines.append(f"（提示：{result['vector_error']}）")
        text = "\n".join(lines)
        return text[: config.TOOL_CONTEXT_MAX_CHARS]
