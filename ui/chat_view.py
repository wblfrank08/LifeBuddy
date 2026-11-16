"""聊天页面视图：自然对话、流式输出与输入捕获。

对话页固定为自然对话：预检索资料仅供参考，模型可自由聊天、用通用知识
作答、自动建档，知识库零命中也照常回答。严格的「知识库文档问答」入口
放在知识库页的「知识查询」弹窗中（见 knowledge_view.py），两者互不干扰。

``consume_stream`` 同时被知识查询弹窗复用，``with_source_line=True`` 时
按真实文件名确定性补全来源行。

严格遵守：
- ``st.chat_input`` 放在页面主层级，绝不放入 ``st.fragment``；
- 视图层定义 async generator，内部 ``async with agent.run_stream(...)``，
  使用 ``stream_text(delta=True)`` 获取增量文本；
- 工具调用导致生成器暂停时用 ``st.spinner("正在查询资料...")`` 展示等待；
- 每轮对话单独创建并关闭模型会话，保证 HTTP 客户端与 event loop 同生命周期。
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, AsyncGenerator, Callable

import streamlit as st

import config
from agent.agent_core import NATURAL_SYSTEM_PROMPT, Deps, prepend_system_prompt
from ui.notify import toast_error, toast_success


def response_stream(
    agent: Any,
    prompt: str,
    deps: Deps,
    message_history: list[Any],
) -> AsyncGenerator[str, None]:
    """异步生成器：增量产出模型文本（本函数不做任何 UI 渲染）。"""

    async def generator() -> AsyncGenerator[str, None]:
        async with agent.run_stream(
            prompt, deps=deps, message_history=message_history
        ) as response:
            async for delta in response.stream_text(delta=True):
                if delta:
                    yield delta

    return generator()


async def consume_stream(
    session_factory: Callable[[], Any],
    prompt: str,
    deps: Deps,
    message_history: list[Any],
    answer_slot: Any,
    with_source_line: bool = True,
) -> str:
    """手动驱动异步生成器，用 ``st.empty()`` 占位更新实现打字机效果。

    开发文档「避坑指南 7」：Streamlit 1.67 的 ``st.write_stream`` 与 PydanticAI
    异步生成器结合时会在 ``ContextVar`` 跨上下文处报错，因此退化为手动
    ``async for`` 循环 + 占位更新；增量文本仍由 ``stream_text(delta=True)`` 保证。

    关键点：整个模型会话（含 ``httpx.AsyncClient``）必须在**当前
    ``asyncio.run`` 的 event loop 内**创建、使用并在 finally 中关闭。若把会话
    放进 ``@st.cache_resource`` 长期缓存，下一轮 ``asyncio.run`` 会复用绑定到
    上一轮已关闭 loop 的连接池，抛出 ``RuntimeError: Event loop is closed``。

    ``with_source_line=True``（知识查询弹窗）时做「知识来源兜底」：本轮命中
    知识库且模型漏写来源行则按真实文件名补齐；自然对话传 False，不强制来源行。
    """

    session = session_factory()
    try:
        # 显式前置 System Prompt：PydanticAI 传入非空 message_history 时不会
        # 再自动注入 Agent 的指令，否则第 2 轮起会丢失对应模式的全部规则。
        history = prepend_system_prompt(
            getattr(session, "system_prompt", ""), message_history
        )
        stream = response_stream(session.agent, prompt, deps, history)
        accumulated = ""
        iterator = stream.__aiter__()
        first = True
        while True:
            if first:
                with st.spinner(":green[正在思考...]"):
                    try:
                        delta = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
            else:
                try:
                    delta = await iterator.__anext__()
                except StopAsyncIteration:
                    break

            if not delta:
                continue
            first = False
            accumulated += delta
            answer_slot.markdown(accumulated)

        reply = accumulated
        if (
            with_source_line
            and reply.strip()
            and "> 知识来源：" not in reply
            and deps.last_knowledge_sources
        ):
            names = "、".join(dict.fromkeys(deps.last_knowledge_sources))
            reply = reply.rstrip() + f"\n\n> 知识来源：{names}"
            answer_slot.markdown(reply)
        return reply
    finally:
        # loop 被 asyncio.run 关闭之前主动释放连接池，杜绝跨 loop 残留
        try:
            await session.aclose()
        except Exception:  # noqa: BLE001 - 关闭失败不应覆盖本轮的原始异常
            pass


def render_chat() -> None:
    """渲染对话主页（无状态函数，供 st.navigation 调度）。"""

    runtime = st.session_state["runtime"]
    agent_factory = runtime["agent_factory"]
    deps: Deps = runtime["deps"]
    chat_service = deps.chat_service
    session_id = st.session_state["session_id"]

    # ---- header：标题 + 清空会话 ------------------------------------------ #
    title_col, clear_col = st.columns([6, 1.2], vertical_alignment="center")
    title_col.header("智能对话")
    if clear_col.button("清空会话", key="chat_clear", width="stretch"):
        # 同时清理页面状态与数据库，避免刷新页面后历史重现
        chat_service.clear_session(session_id)
        st.session_state["messages"] = []
        # 清空按钮位于页面主层级，点击本身即触发整页重跑，列表自然清空；
        # 显式 st.rerun() 反而会在脚本结束前取消紧随其后的成功 toast。
        toast_success("会话已清空")

    # 必须位于页面主层级：放入 st.fragment 会让输入框失去底部吸附定位。
    prompt = st.chat_input("输入你想说的话…", key="chat_input_main")

    model_prompt = ""
    if prompt:
        # 先查后答：代码层确定性预检索（跳过向量、零网络依赖），资料仅供参考，
        # 不拦截零命中，也不强制来源行。必须早于 prepare_turn（落库本轮消息）。
        turn = chat_service.build_turn_context(prompt, natural=True)
        deps.last_knowledge_sources = []
        tail = (
            "【以上是系统预先检索的资料，仅供参考：可以自由地用你的知识"
            "自然作答，涉及用户个人信息时依据档案且不要编造，"
            "不要输出查询过渡语。】"
        )
        model_prompt = f"{turn['context']}\n\n{tail}\n\n【用户问题】{prompt}"
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # 固定自然对话：带工具以支持自动建档，不使用知识查询的 System Prompt
    session_factory = partial(agent_factory, system_prompt=NATURAL_SYSTEM_PROMPT)

    # 固定高度 + 自动滚动的聊天显示区；chat_input 仍在主层级，不受影响
    with st.container(height=1280, autoscroll=True):
        # 每条消息（含历史）统一经 st.empty() 占位渲染，使「本轮流式占位」与
        # 「下一轮历史渲染」在同一位置产生完全相同的元素结构，避免幽灵消息。
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.empty().markdown(message["content"])

        if prompt:
            profiles_before = len(deps.profile_service.list_profiles())
            failed = False
            with st.chat_message("assistant"):
                answer_slot = st.empty()
                reply = ""
                try:
                    # 无论是否调用模型，都要把用户消息落库
                    history = chat_service.prepare_turn(session_id, prompt)
                    reply = asyncio.run(
                        consume_stream(
                            session_factory,
                            model_prompt,
                            deps,
                            history,
                            answer_slot,
                            with_source_line=False,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - 连接失败必须明确报错
                    failed = True
                    toast_error(
                        f"生成回答失败：{exc}。请确认网络可访问 "
                        f"{config.LLM.base_url}、config.py 中 LLM.api_key 已填写，"
                        "或修改模型地址。"
                    )

            if reply:
                chat_service.record_assistant_message(session_id, reply)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": reply}
                )
                # 主动建档成功时用 toast 轻量提示
                if len(deps.profile_service.list_profiles()) > profiles_before:
                    toast_success("已自动更新个人档案")

    # 本轮回复落库后无需显式 st.rerun()：chat_input 触发的脚本重跑已让页面自然
    # 进入最终状态；显式 rerun 会取消上面的成功 / 错误 toast，并破坏轻提示要求。
    # 每条消息（含本轮）均经 st.empty() 占位渲染，结构一致，无幽灵消息残留。
