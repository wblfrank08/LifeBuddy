"""LifeBuddy 程序启动入口：st.navigation 路由 + 全局依赖组装。

单向依赖：UI -> Agent -> Service -> DB。
所有 Repository 与 Service 都在此处实例化后注入，Service 内部禁止自行创建 Repo。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agent.agent_core import Deps, build_agent_session
from db.connection import DatabaseConnection
from db.document_repo import DocumentRepository
from db.knowledge_repo import KnowledgeRepository
from db.message_repo import MessageRepository
from db.profile_repo import ProfileRepository
from services.chat_service import ChatService
from services.document_service import DocumentService
from services.knowledge_service import KnowledgeService
from services.profile_service import ProfileService
from services.search_service import SearchService
from ui import chat_view, document_view, knowledge_view, profile_view


@st.cache_resource
def build_runtime() -> dict[str, Any]:
    """组装全局单例资源（DB 连接、Repository、Service、模型会话工厂）。

    缓存后页面刷新不会重复初始化连接与服务。

    注意：这里缓存的是**模型会话工厂**，而不是 Agent 实例。Agent 内含的
    ``httpx.AsyncClient`` 会绑定创建它的 event loop；若长期缓存，第二轮对话
    复用上一轮已关闭 loop 的连接池会抛出 ``RuntimeError: Event loop is closed``。
    真正的模型会话由 UI 层在每轮对话的 ``asyncio.run`` 内创建并关闭。
    """

    conn = DatabaseConnection.get_instance().connection

    message_repo = MessageRepository(conn)
    profile_repo = ProfileRepository(conn)
    document_repo = DocumentRepository(conn)
    knowledge_repo = KnowledgeRepository(conn)

    knowledge_service = KnowledgeService(knowledge_repo)
    profile_service = ProfileService(profile_repo, message_repo)
    document_service = DocumentService(document_repo, profile_service)
    search_service = SearchService(message_repo, knowledge_repo, knowledge_service)
    chat_service = ChatService(message_repo, profile_service, search_service)

    deps = Deps(
        chat_service=chat_service,
        profile_service=profile_service,
        document_service=document_service,
        search_service=search_service,
        knowledge_service=knowledge_service,
    )
    return {"deps": deps, "agent_factory": build_agent_session}


def init_session_state() -> None:
    """跨页面共享状态统一由 st.session_state 管理。"""

    if "runtime" not in st.session_state:
        st.session_state["runtime"] = build_runtime()
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "lifebuddy-main"
    if "messages" not in st.session_state:
        history = st.session_state["runtime"]["deps"].chat_service.get_history(
            st.session_state["session_id"]
        )
        st.session_state["messages"] = [
            {"role": row["role"], "content": row["content"]} for row in history
        ]


def main() -> None:
    st.set_page_config(page_title="LifeBuddy", layout="wide", page_icon=":material/book:")
    init_session_state()

    pages = [
        st.Page(chat_view.render_chat, title="智能对话", icon=":material/chat:", default=True),
        st.Page(profile_view.render_profile, title="个人档案", icon=":material/badge:"),
        st.Page(knowledge_view.render_knowledge, title="知识库", icon=":material/library_books:"),
        st.Page(document_view.render_documents, title="简历中心", icon=":material/description:"),
    ]
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()

    with st.sidebar:
        st.caption("LifeBuddy V1.0 | By Maxwell")


main()
