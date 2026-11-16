"""知识库页面视图：上传文件、查看入库状态、右上角「知识查询」弹窗。

防重复导入：``st.file_uploader`` 的值会在 rerun 后保留，若直接在主层级
消费它，任何一次 rerun 都会把同一文件再入库一次。这里用 ``on_change``
回调 + session_state 标记，确保只有「用户真正选择/更换文件」时才触发导入；
Service 层另做同名幂等兜底。

「知识查询」弹窗是独立的严格知识文档问答：只依据上传的知识库作答，零命中
由代码层回复固定文案，历史与主对话隔离，且关闭弹窗后不保留。

布局上聊天记录在上、输入框在下；关闭弹窗由 ``on_dismiss`` 清空历史，
重新打开即为全新会话。
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import streamlit as st

import config
from agent.agent_core import NO_KNOWLEDGE_REPLY, SYSTEM_PROMPT, Deps
from services.chat_service import to_model_messages
from services.knowledge_service import KnowledgeServiceError
from ui.chat_view import consume_stream
from ui.notify import toast_error, toast_success, toast_warning

_PENDING_KEY = "_knowledge_upload_pending"
_DEDUP_KEY = "_knowledge_dedup_done"
_KB_HISTORY_KEY = "kb_query_history"


def _mark_upload_pending() -> None:
    """仅在文件选择发生变化时打标；普通 rerun 不会调用此回调。"""

    st.session_state[_PENDING_KEY] = True


def _clear_kb_history() -> None:
    """清空知识查询历史：弹窗关闭（或重新打开）后不保留上一轮会话。"""

    st.session_state.pop(_KB_HISTORY_KEY, None)


@st.dialog("知识查询", width="large", on_dismiss=_clear_kb_history)
def _knowledge_query_dialog(deps: Deps, agent_factory: Any) -> None:
    """严格依据知识库文档的独立问答（large 弹窗）。

    布局：聊天记录在上、输入框在下。先用 ``st.container`` 占位、再放
    ``st.chat_input``，最后回过头向容器回填历史与本轮回答——既让输入框位于
    弹窗底部，又保证消息不会渲染到输入框之外（dialog 内 chat_input 是 inline，
    不会像主页面那样自动吸附底部）。

    弹窗关闭时由 ``on_dismiss`` 清空历史，重新打开即为全新会话。
    """

    st.caption(
        "严格依据已上传的知识库文档作答，没有相关资料时会明确告知；"
        "此处的问答记录与「对话」页互相独立，关闭本窗口后不会保留。"
    )
    history: list[dict[str, str]] = st.session_state.setdefault(_KB_HISTORY_KEY, [])

    # 1) 聊天记录容器先占位（视觉在上，固定高度可滚动）
    chat_area = st.container(height=460, autoscroll=True)

    # 2) 输入框随后放置（视觉在下）
    question = st.chat_input("输入你想从知识库查询的问题…", key="kb_chat_input")

    previous = list(history)
    if question:
        history.append({"role": "user", "content": question})

    # 3) 回填聊天记录：容器对象可在创建后再次 with 写入，元素仍渲染在容器
    #    原本的位置（上方），不会跑到输入框之外。先渲染全部历史（含本轮用户
    #    消息），再在本轮把助手回答流式写入同一个容器。
    with chat_area:
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question:
            # 严格知识查询：只注入知识库分块，零命中不调用 LLM
            turn = deps.chat_service.build_turn_context(question, natural=False)
            deps.last_knowledge_sources = turn["sources"]

            reply = ""
            with st.chat_message("assistant"):
                answer_slot = st.empty()
                if not turn["has_knowledge"]:
                    reply = NO_KNOWLEDGE_REPLY
                    answer_slot.markdown(reply)
                else:
                    model_prompt = (
                        f"{turn['context']}\n\n"
                        "【以上是系统从知识库检索到的资料，请只依据它回答；"
                        "资料不足就按规则回复固定文案，不要输出查询过渡语。】\n\n"
                        f"【用户问题】{question}"
                    )
                    # 严格问答不需要任何工具（资料已预检索注入），彻底杜绝
                    # 模型改用档案或通用知识作答。
                    session_factory = partial(
                        agent_factory, system_prompt=SYSTEM_PROMPT, with_tools=False
                    )
                    try:
                        reply = asyncio.run(
                            consume_stream(
                                session_factory,
                                model_prompt,
                                deps,
                                to_model_messages(previous),
                                answer_slot,
                                with_source_line=True,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - 连接失败必须明确报错
                        toast_error(
                            f"生成回答失败：{exc}。请确认网络可访问 "
                            f"{config.LLM.base_url}、config.py 中 LLM.api_key 已填写，"
                            "或修改模型地址。"
                        )
            if reply:
                history.append({"role": "assistant", "content": reply})


@st.fragment
def _knowledge_body() -> None:
    """知识库页的局部交互区：去重清理、文件上传与文件列表。

    上传与删除都在此 fragment 内完成；筛选 / 上传等局部交互由 Streamlit 自动
    触发 fragment 级重跑，避免整页重绘。
    """

    deps: Deps = st.session_state["runtime"]["deps"]
    knowledge_service = deps.knowledge_service

    # 清理历史版本遗留的同名重复记录（每会话一次，幂等）
    if not st.session_state.get(_DEDUP_KEY):
        removed = knowledge_service.deduplicate()
        st.session_state[_DEDUP_KEY] = True
        if removed:
            toast_success(f"已清理 {removed} 条重复导入记录")

    # 单文件上传：accept_multiple_files=False 时 uploaded 是单个 UploadedFile，
    # 不能再按列表遍历（UploadedFile 可迭代会按字节行拆分，导致 .name 报错）。
    uploaded = st.file_uploader(
        "选择资料文件（仅支持 .md 与 .txt）",
        type=["md", "txt"],
        accept_multiple_files=False,
        key="knowledge_uploader",
        on_change=_mark_upload_pending,
        max_upload_size=1,
    )

    pending = st.session_state.pop(_PENDING_KEY, False)
    if uploaded and pending:
        try:
            with st.spinner(f"正在消化 {uploaded.name}…"):
                result = knowledge_service.ingest_file(
                    uploaded.name, uploaded.getvalue()
                )
            if result.get("duplicate"):
                toast_warning(f"{result['filename']} 已存在，已跳过重复导入")
            else:
                toast_success(
                    f"{result['filename']} 入库完成，共 {result['chunk_count']} 个分块"
                )
        except KnowledgeServiceError as exc:
            toast_error(str(exc))
    elif uploaded:
        st.caption("文件已导入。若要重新导入同名文件，请先删除下方对应记录。")

    st.subheader("已上传文件")
    documents = knowledge_service.list_documents()
    if not documents:
        st.caption("暂无文件，上传一份旧简历或学习笔记试试。")
        return
    for document in documents:
        document_id = document["id"]
        col_name, col_status, col_chunks, col_action = st.columns([3, 1, 1, 1])
        col_name.write(document["filename"])
        col_status.write(document["status"])
        col_chunks.write(f"{document['chunk_count']} 块")
        if col_action.button(
            "删除", key=f"delete_knowledge_{document_id}", width="stretch"
        ):
            knowledge_service.delete_document(document_id)
            # 列表在删除前已查询（见上方 list_documents），本帧渲染的仍是旧列表，
            # 故用 st.rerun(scope="fragment") 让片段局部重跑并重新拉取列表；
            # 该调用不会取消紧随其后的彩色 toast，满足「局部刷新 + 轻提示」。
            toast_success("文件已删除")
            st.rerun(scope="fragment")
        if document["error"]:
            toast_error(document["error"])


def render_knowledge() -> None:
    """渲染知识库页（无状态函数）。"""

    deps: Deps = st.session_state["runtime"]["deps"]
    agent_factory = st.session_state["runtime"]["agent_factory"]

    # ---- header：标题 + 右上角知识查询入口 ------------------------------- #
    title_col, query_col = st.columns([6, 1.2], vertical_alignment="center")
    title_col.header("知识库")
    if query_col.button(
        "知识查询", key="kb_query_open", width="stretch", type="secondary"
    ):
        # 打开前清空：与 on_dismiss 双保险，保证每次点开都是全新会话
        _clear_kb_history()
        _knowledge_query_dialog(deps, agent_factory)
    st.caption("导入 .md / .txt 资料，LifeBuddy 会自动分块、建立索引，供知识查询检索。")

    _knowledge_body()
