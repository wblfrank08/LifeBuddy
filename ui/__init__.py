"""UI 渲染层：只负责渲染与事件捕获。

对外暴露的均为无状态 ``render_xxx()`` 函数，由 ``app.py`` 的
``st.navigation`` 路由调度。本层禁止编写 SQL 与业务规则。
"""
