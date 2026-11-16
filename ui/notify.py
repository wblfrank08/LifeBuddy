"""统一的彩色轻提示：用 ``st.toast`` 替代 ``st.error`` / ``st.info`` / ``st.warning``。

开发文档 4.5 原本要求异常用 ``st.error`` 在页面内强提示；用户最新要求改为统一
``st.toast``，并用 Streamlit 的彩色 Markdown 语法给文字上色，使成功/警告/错误
在视觉上一目了然：

- 成功：``:green[...]``
- 警告：``:orange[...]``
- 错误：``:red[...]``

toast 会在数秒后自动消失，因此只承载「操作结果反馈」，不承载需要长期驻留的
空状态说明（空状态仍用 ``st.caption`` 等原生组件）。
"""

from __future__ import annotations

import streamlit as st

_QUEUED_KEY = "_queued_toasts"


def toast_success(message: str) -> None:
    """操作成功：绿色提示。"""

    st.toast(f":green[{message}]")


def toast_warning(message: str) -> None:
    """需要注意：橙色提示。"""

    st.toast(f":orange[{message}]")


def toast_error(message: str) -> None:
    """操作失败 / 模型断片：红色提示。"""

    st.toast(f":red[{message}]")


def queue_toast(message: str, kind: str = "success") -> None:
    """在 ``st.rerun()`` 之前排队一条彩色 toast。

    ``st.rerun()`` 会取消紧随其后的 ``st.toast()``，但排队在 ``session_state``
    中的提示不会被取消。下一轮渲染早期调用 :func:`flush_queued_toasts` 即可在
    rerun 之后展示——从而同时满足「对话框关闭 / 列表刷新」与「彩色轻提示」。

    ``kind`` 取值：``success`` / ``warning`` / ``error``。
    """

    st.session_state.setdefault(_QUEUED_KEY, []).append((kind, message))


def flush_queued_toasts() -> None:
    """在页面 / 片段渲染早期调用，展示上一轮排队的 toast。

    rerun 之后的这一轮渲染里调用它，排队项便不会被取消，且只展示一次。
    """

    for kind, message in st.session_state.pop(_QUEUED_KEY, []):
        if kind == "error":
            toast_error(message)
        elif kind == "warning":
            toast_warning(message)
        else:
            toast_success(message)
