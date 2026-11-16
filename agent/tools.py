"""@agent.tool 工具函数定义：仅做参数校验与转发。

所有业务逻辑都通过 ``ctx.deps`` 调用 Service 层，工具函数本身不持有状态。
返回给模型的上下文统一裁剪，避免超长文本挤爆本地小模型的窗口限制。
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

import config
from agent.agent_core import Deps
from services.profile_service import PROFILE_CATEGORIES


def clip(text: str) -> str:
    """纯函数：裁剪工具返回给模型的上下文长度。"""

    return str(text).strip()[: config.TOOL_CONTEXT_MAX_CHARS]


def register_tools(agent: Any) -> None:
    """把工具注册到 Agent 实例（函数内定义，规避模块级循环导入）。"""

    allowed = tuple(PROFILE_CATEGORIES)

    @agent.tool
    async def search_memory(ctx: RunContext[Deps], query: str) -> str:
        """按需补充检索历史对话与知识库。

        通常不需要调用：本轮资料已随用户消息附上，直接据此回答即可。
        """

        keyword = (query or "").strip()[:100]
        if not keyword:
            return "检索词为空，请提供要查询的关键词。"
        try:
            result = ctx.deps.search_service.hybrid_search(keyword)
            return clip(ctx.deps.search_service.format_context(result))
        except Exception as exc:  # noqa: BLE001 - 工具内需把异常转成自然语言
            return f"检索失败：{exc}"

    @agent.tool
    async def save_profile(ctx: RunContext[Deps], category: str, content: str) -> str:
        """归档一条**全新**的个人信息；高度相似的已有条目会自动合并更新。"""

        cleaned = (content or "").strip()[:500]
        if not cleaned:
            return "档案内容为空，未保存。"
        normalized = (category or "").strip()
        if normalized not in allowed:
            normalized = "其他"
        try:
            profile_id, created = ctx.deps.profile_service.upsert_profile(
                normalized, cleaned
            )
            action = "已归档" if created else "已合并更新已有档案"
            return f"{action}（编号 {profile_id}）到「{normalized}」：{cleaned}"
        except Exception as exc:  # noqa: BLE001
            return f"档案保存失败：{exc}"

    @agent.tool
    async def get_profile(ctx: RunContext[Deps], query: str = "") -> str:
        """查看已记录的个人档案，每条都带 id 编号，供 update_profile 使用。"""

        keyword = (query or "").strip()[:100]
        try:
            items = (
                ctx.deps.profile_service.search(keyword)
                if keyword
                else ctx.deps.profile_service.list_profiles()
            )
            if not items:
                return "目前还没有相关的档案记录。"
            lines = [
                f"[id={item['id']}][{item['category']}] {item['content']}"
                for item in items
            ]
            return clip("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            return f"档案读取失败：{exc}"

    @agent.tool
    async def update_profile(
        ctx: RunContext[Deps],
        profile_id: int,
        category: str = "",
        content: str = "",
    ) -> str:
        """修改已有档案（先用 get_profile 取得 id）；留空的字段保持原值。"""

        try:
            target = int(profile_id)
        except (TypeError, ValueError):
            return "档案编号无效，请先调用 get_profile 查看编号。"
        existing = ctx.deps.profile_service.get(target)
        if existing is None:
            return f"未找到编号 {target} 的档案，请先用 get_profile 查看。"
        new_category = (category or "").strip() or existing["category"]
        if new_category not in allowed:
            new_category = "其他"
        new_content = (content or "").strip()[:500] or existing["content"]
        try:
            ctx.deps.profile_service.update_profile(
                target, new_category, new_content
            )
            return f"已更新档案 {target}：[{new_category}] {new_content}"
        except Exception as exc:  # noqa: BLE001
            return f"档案更新失败：{exc}"

    @agent.tool
    async def delete_profile(ctx: RunContext[Deps], profile_id: int) -> str:
        """删除指定编号的档案（用户明确要求忘掉某条信息时使用）。"""

        try:
            target = int(profile_id)
        except (TypeError, ValueError):
            return "档案编号无效，请先调用 get_profile 查看编号。"
        try:
            if ctx.deps.profile_service.get(target) is None:
                return f"未找到编号 {target} 的档案。"
            ctx.deps.profile_service.delete(target)
            return f"已删除档案 {target}。"
        except Exception as exc:  # noqa: BLE001
            return f"档案删除失败：{exc}"

    @agent.tool
    async def get_resume_materials(ctx: RunContext[Deps]) -> str:
        """汇总分散在各处的档案素材，用于生成简历。"""

        try:
            materials = ctx.deps.profile_service.collect_resume_materials()
            if not materials["total"]:
                return "暂无档案素材，请先在对话中记录技能与经历。"
            # 简历要求全量列出所有经历与项目，用专属上限，不套用通用 clip()
            return str(materials["text"]).strip()[: config.RESUME_MATERIALS_MAX_CHARS]
        except Exception as exc:  # noqa: BLE001
            return f"素材汇总失败：{exc}"

    @agent.tool
    async def save_document(
        ctx: RunContext[Deps], title: str, content: str, doc_type: str = "resume"
    ) -> str:
        """把生成的简历或总结保存为 Markdown 文档。"""

        cleaned = (content or "").strip()
        if not cleaned:
            return "文档内容为空，未保存。"
        safe_type = doc_type if doc_type in ("resume", "summary") else "resume"
        try:
            document_id = ctx.deps.document_service.save_resume(
                (title or "个人简历").strip()[:100], cleaned, safe_type
            )
            return f"文档已保存，编号 {document_id}。请直接把正文输出给用户。"
        except Exception as exc:  # noqa: BLE001
            return f"文档保存失败：{exc}"

    @agent.tool
    async def list_knowledge(ctx: RunContext[Deps]) -> str:
        """查看知识库中已上传的文件及其入库状态。"""

        try:
            documents = ctx.deps.knowledge_service.list_documents()
            if not documents:
                return "知识库暂无文件。"
            lines = [
                f"{item['filename']}：{item['status']}，{item['chunk_count']} 个分块"
                for item in documents
            ]
            return clip("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            return f"知识库读取失败：{exc}"
