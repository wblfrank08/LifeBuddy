"""档案页面视图：按分类浏览档案，并提供编辑/删除入口与「刷新档案」。

档案数据查询放在 ``st.fragment`` 内部，筛选切换与「刷新档案」的局部重跑都会
重新调用 ``grouped()``，确保拿到最新数据，而不依赖外层闭包的旧快照。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agent.agent_core import Deps
from services.profile_service import PROFILE_CATEGORIES
from ui.notify import flush_queued_toasts, queue_toast, toast_success


@st.dialog("编辑档案", width="medium")
def _edit_profile_dialog(record: dict[str, Any], deps: Deps) -> None:
    """可编辑表单：修改分类/内容、保存或删除；原始数据仍以 JSON 折叠保留。"""

    categories = list(PROFILE_CATEGORIES)
    current = record["category"]
    index = categories.index(current) if current in categories else len(categories) - 1

    st.caption("修改后会直接更新这条档案，不会再新增一条重复记录。")
    with st.form(key=f"profile_edit_form_{record['id']}"):
        category = st.selectbox("分类", options=categories, index=index)
        content = st.text_area("内容", value=record["content"], height=120)
        save_col, delete_col = st.columns(2)
        saved = save_col.form_submit_button("保存修改", width="stretch")
        deleted = delete_col.form_submit_button("删除该档案", width="stretch")

    if saved:
        try:
            deps.profile_service.update_profile(record["id"], category, content or "")
            queue_toast("档案已更新", "success")
        except Exception as exc:  # noqa: BLE001 - 表单失败需明确提示
            queue_toast(f"保存失败：{exc}", "error")
        # 弹窗由 _filterable_section 内的按钮打开；rerun 关闭弹窗并刷新片段列表。
        # 成功 / 失败提示走「延迟 toast」，rerun 之后由 render_profile 的
        # flush_queued_toasts 展示，避免被 st.rerun() 取消。
        st.rerun()
    if deleted:
        deps.profile_service.delete(record["id"])
        queue_toast("档案已删除", "success")
        st.rerun()

    with st.expander("原始数据（JSON）", expanded=False):
        st.json(record, width="stretch")


def render_profile() -> None:
    """渲染个人档案页（无状态函数）。"""

    # 渲染早期 flush：展示上一轮对话框提交经 st.rerun() 排队的彩色 toast
    flush_queued_toasts()

    deps: Deps = st.session_state["runtime"]["deps"]

    # 局部交互封装在 st.fragment，避免切换筛选或刷新时整页重绘
    @st.fragment
    def _profile_section() -> None:
        # 数据查询置于片段内：任何局部重跑都会重新拉取最新档案
        grouped = deps.profile_service.grouped()

        # ---- header：标题 + 刷新档案 --------------------------------------- #
        title_col, refresh_col = st.columns([6, 1.2], vertical_alignment="center")
        title_col.header("个人档案")
        if refresh_col.button(
            "刷新档案", key="profile_refresh", width="stretch", type="secondary"
        ):
            # 局部重跑本片段即可重新获取档案数据，不整页重绘；
            # scope="fragment" 不会取消紧随其后的彩色 toast。
            toast_success("档案已刷新")
            st.rerun(scope="fragment")

        st.caption("系统在对话中自动为你沉淀的技能、经历与偏好。")

        st.subheader("分类浏览")
        options = list(PROFILE_CATEGORIES)
        choice = st.segmented_control(
            "筛选分类", options, key="profile_filter", default="基本信息", required= True
        )
        # 默认选中「基本信息」；用户手动取消选择时展示全部分类
        categories = [choice] if choice else list(grouped.keys())
        visible = False
        for category in categories:
            items = grouped.get(category, [])
            if not items:
                continue
            visible = True
            with st.container(height=1200, border=True):
                for item in items:
                    st.write(f"- {item['content']}")
                    st.caption(f"来源：{item['source']} · {item['created_at']}")
                    if st.button(
                        "编辑档案",
                        key=f"edit_profile_{item['id']}",
                        width="stretch",
                    ):
                        _edit_profile_dialog(item, deps)
            with st.bottom:
                st.caption(f"共 {len(items)}条 ")
        if not visible:
            st.caption("该分类下暂无记录。")

    _profile_section()
