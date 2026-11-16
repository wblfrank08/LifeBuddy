"""档案数据聚合、简历素材提取。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import config
from db.message_repo import MessageRepository
from db.profile_repo import ProfileRepository

PROFILE_CATEGORIES: tuple[str, ...] = (
    "基本信息",
    "技能",
    "教育经历",
    "项目经验",
    "工作经历",
    "个人偏好",
    "其他",
)


def _normalize(text: str) -> str:
    """纯函数：仅保留中英文与数字，用于档案相似度比较。"""

    return "".join(
        char
        for char in text
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def is_similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """纯函数：判断两条档案是否在说同一件事（包含关系或字符级 Jaccard）。

    只用于**同一分类**内的去重合并，阈值偏保守，避免把不同技能误合并。
    """

    left, right = _normalize(a), _normalize(b)
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    set_a, set_b = set(left), set(right)
    return len(set_a & set_b) / len(set_a | set_b) >= threshold


def format_materials(
    grouped: dict[str, list[dict[str, Any]]],
    related_messages: list[dict[str, Any]],
) -> str:
    """纯函数：把结构化档案与相关对话线索组织成简历素材文本。"""

    lines: list[str] = []
    for category in PROFILE_CATEGORIES:
        items = grouped.get(category, [])
        if not items:
            continue
        lines.append(f"【{category}】")
        lines.extend(f"- {item['content']}" for item in items)
    # 归入自定义分类的档案
    known = set(PROFILE_CATEGORIES)
    for category, items in grouped.items():
        if category in known:
            continue
        lines.append(f"【{category}】")
        lines.extend(f"- {item['content']}" for item in items)

    if related_messages:
        lines.append("【近期对话线索】")
        lines.extend(
            f"- {item['content']}"
            for item in related_messages[: config.RESUME_RELATED_MESSAGES_LIMIT]
        )

    if not lines:
        return "暂无档案素材。请先在对话中聊聊你的技能、项目或经历。"
    return "\n".join(lines)


class ProfileService:
    """档案业务服务：通过构造函数注入所需的 Repository 实例。"""

    def __init__(
        self,
        profile_repo: ProfileRepository,
        message_repo: MessageRepository,
    ) -> None:
        self._profile_repo = profile_repo
        self._message_repo = message_repo

    def add_profile(
        self, category: str, content: str, source: str = "chat"
    ) -> int:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("档案内容不能为空")
        normalized = category.strip() or "其他"
        return self._profile_repo.add(normalized, cleaned, source)

    def upsert_profile(
        self, category: str, content: str, source: str = "chat"
    ) -> tuple[int, bool]:
        """归档信息：同分类下已有高度相似条目则**更新**，否则新增。

        返回 ``(档案 id, 是否为新增)``。避免同一件事换种说法反复堆积成多条。
        """

        cleaned = content.strip()
        if not cleaned:
            raise ValueError("档案内容不能为空")
        normalized = category.strip() or "其他"
        for item in self._profile_repo.list_by_category(normalized):
            if is_similar(item["content"], cleaned):
                self._profile_repo.update(item["id"], normalized, cleaned, source)
                return int(item["id"]), False
        return self._profile_repo.add(normalized, cleaned, source), True

    def get(self, profile_id: int) -> Optional[dict[str, Any]]:
        return self._profile_repo.get(profile_id)

    def update_profile(
        self,
        profile_id: int,
        category: str,
        content: str,
        source: str = "chat",
    ) -> bool:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("档案内容不能为空")
        normalized = category.strip() or "其他"
        return self._profile_repo.update(profile_id, normalized, cleaned, source)

    def list_profiles(self) -> list[dict[str, Any]]:
        return self._profile_repo.list_all()

    def grouped(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self._profile_repo.list_all():
            grouped[item["category"]].append(item)
        return dict(grouped)

    def search(self, query: str) -> list[dict[str, Any]]:
        return self._profile_repo.search(query)

    def delete(self, profile_id: int) -> None:
        self._profile_repo.delete(profile_id)

    def collect_resume_materials(self) -> dict[str, Any]:
        """汇总分散在各处的档案信息，作为生成简历的素材。"""

        grouped = self.grouped()
        related_messages = self._message_repo.search(
            "项目", config.RESUME_RELATED_MESSAGES_LIMIT
        )
        materials_text = format_materials(grouped, related_messages)
        return {
            "grouped": grouped,
            "related_messages": related_messages,
            "text": materials_text,
            "total": sum(len(items) for items in grouped.values()),
        }
