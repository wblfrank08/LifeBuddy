"""聊天流程控制、确定性预检索与消息落库调度。

对话页有两种模式，共用同一套预检索框架但注入不同的上下文：
- 知识查询：只注入知识库分块，零命中时由 UI 直接拦截为固定文案；
- 自然对话：注入个人档案 + 相关历史对话 + 知识库资料（均仅供参考），
  不拦截零命中，模型可自由聊天、使用通用知识并自动建档。
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

import config
from db.message_repo import MessageRepository
from services.profile_service import ProfileService
from services.search_service import SearchService


def to_model_messages(history: list[dict[str, Any]]) -> list[ModelRequest | ModelResponse]:
    """纯函数：把 DB 聊天记录转换为 PydanticAI 的 ModelMessage 序列。"""

    messages: list[ModelRequest | ModelResponse] = []
    for item in history:
        content = str(item["content"]).strip()
        if not content:
            continue
        if item["role"] == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif item["role"] == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=content)]))
    return messages


def _source_names(chunks: list[dict[str, Any]]) -> list[str]:
    """纯函数：按出现顺序提取去重后的知识库文件名。"""

    return list(
        dict.fromkeys(
            str(item.get("filename", "")).strip()
            for item in chunks
            if str(item.get("filename", "")).strip()
        )
    )


def format_turn_context(
    result: dict[str, Any],
    max_chars: int = config.TURN_CONTEXT_MAX_CHARS,
) -> str:
    """纯函数：知识查询模式——只注入知识库分块的《本轮检索资料》。

    文件清单放在首行，保证尾部裁剪不会把它截掉——它既是来源标注唯一
    可引用的名称，也是 UI 层兜底依据。
    """

    chunks = result.get("chunks", [])
    sources = _source_names(chunks)
    lines: list[str] = ["《本轮检索资料》"]
    lines.append(
        f"【本轮检索到的知识库文件】{'、'.join(sources)}"
        if sources
        else "【本轮检索到的知识库文件】无（本轮未命中知识库资料）"
    )

    if chunks:
        lines.append("【知识库资料】")
        for index, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[资料{index}·{chunk.get('filename', '知识库')}] {chunk['content']}"
            )
    else:
        lines.append("【知识库资料】无")
    return "\n".join(lines)[:max_chars]


def format_natural_context(
    result: dict[str, Any],
    profiles: list[dict[str, Any]],
    max_chars: int = config.TURN_CONTEXT_MAX_CHARS,
) -> str:
    """纯函数：自然对话模式——把档案/历史/知识库组织成供参考的资料。

    与知识查询的区别：这里明确标注「仅供参考」，允许模型自由聊天、用通用
    知识作答，知识库无命中也不影响作答。个人档案放在最前，便于自动建档与
    更正信息时取用。
    """

    lines: list[str] = ["《本轮检索资料》（系统预先检索，仅供参考，可自由作答）"]

    lines.append("【个人档案】")
    if profiles:
        for item in profiles:
            lines.append(
                f"- [id={item['id']}][{item['category']}] {item['content']}"
            )
    else:
        lines.append("（暂无档案记录）")

    related = result.get("messages", [])
    lines.append("【相关历史对话】")
    if related:
        for message in related:
            lines.append(f"[{message['role']}] {message['content']}")
    else:
        lines.append("（本轮未命中相关历史对话）")

    chunks = result.get("chunks", [])
    lines.append("【知识库资料】")
    if chunks:
        for index, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[资料{index}·{chunk.get('filename', '知识库')}] {chunk['content']}"
            )
    else:
        lines.append("（本轮未命中知识库资料）")

    return "\n".join(lines)[:max_chars]


class ChatService:
    """聊天业务服务：构造函数注入所需 Repository / Service，内部不含任何 SQL。"""

    def __init__(
        self,
        message_repo: MessageRepository,
        profile_service: ProfileService,
        search_service: SearchService,
    ) -> None:
        self._message_repo = message_repo
        self._profile_service = profile_service
        self._search_service = search_service

    def build_turn_context(
        self, prompt: str, natural: bool = False
    ) -> dict[str, Any]:
        """代码层**确定性预检索**：先查后答，不依赖本地小模型自主发起工具调用。

        ``natural=True`` 走自然对话模式（注入全量参考上下文、不作零命中拦截）；
        默认 ``False`` 走知识查询模式（只注入知识库，``has_knowledge`` 供 UI
        在零命中时直接输出固定文案、完全不调用 LLM）。
        预检索跳过向量服务，零网络依赖，避免 Embedding 未启动时每轮等待超时。
        """

        result = self._search_service.hybrid_search(prompt, use_vector=False)
        chunks = result.get("chunks", [])
        sources = _source_names(chunks)
        if natural:
            profiles = self._profile_service.list_profiles()
            context = format_natural_context(result, profiles)
        else:
            context = format_turn_context(result)
        return {
            "context": context,
            "sources": sources,
            "has_knowledge": bool(chunks),
        }

    def record_user_message(self, session_id: str, content: str) -> int:
        return self._message_repo.add_message(session_id, "user", content)

    def record_assistant_message(self, session_id: str, content: str) -> int:
        return self._message_repo.add_message(session_id, "assistant", content)

    def get_history(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self._message_repo.list_messages(session_id, limit)

    def clear_session(self, session_id: str) -> None:
        """清空指定会话的聊天记录；页面显示状态由 UI 层同步清空。"""

        self._message_repo.clear_session(session_id)

    def prepare_turn(
        self, session_id: str, prompt: str, history_limit: int = 20
    ) -> list[ModelRequest | ModelResponse]:
        """落库用户消息，并返回用于 ``agent.run_stream`` 的历史消息（不含本轮输入）。"""

        self.record_user_message(session_id, prompt)
        rows = self.get_history(session_id, history_limit * 2 + 1)
        return to_model_messages(rows[:-1])

    def session_stats(self, session_id: str) -> dict[str, int]:
        return {
            "messages": len(self.get_history(session_id)),
            "profiles": self._profile_service.list_profiles().__len__(),
        }
