"""文档保存与生成调度。"""

from __future__ import annotations

from typing import Any, Optional

from db.document_repo import DocumentRepository
from services.profile_service import ProfileService


def strip_code_fence(text: str) -> str:
    """纯函数：去掉模型可能附加的 Markdown 代码块围栏与首尾空白。"""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def inject_job_intention(content: str, job_title: str) -> str:
    """纯函数：在简历标题下确定性插入模板样式的「求职意向：xxx」一行。

    该行由代码层注入而非交给模型，保证 100% 存在且格式统一；同时剔除模型
    可能自行输出的同类行（含旧版「应聘职位」），避免重复。职位名称为空时原样返回。
    """

    title = (job_title or "").strip()
    if not title:
        return content

    lines: list[str] = []
    for line in content.splitlines():
        probe = line.strip().lstrip(">*_- ").strip().lstrip("*").strip()
        if probe.startswith("求职意向") or probe.startswith("应聘职位"):
            continue
        lines.append(line)

    insert_at = next(
        (
            index + 1
            for index, line in enumerate(lines)
            if line.lstrip().startswith("# ")
        ),
        0,
    )
    lines[insert_at:insert_at] = ["", f"求职意向：{title}"]
    return "\n".join(lines).strip()


class DocumentService:
    """生成文档的保存与调度，通过构造函数注入 Repository 与 ProfileService。"""

    def __init__(
        self,
        document_repo: DocumentRepository,
        profile_service: ProfileService,
    ) -> None:
        self._document_repo = document_repo
        self._profile_service = profile_service

    def save_resume(
        self, title: str, content: str, doc_type: str = "resume"
    ) -> int:
        if not content.strip():
            raise ValueError("文档内容不能为空")
        return self._document_repo.add(title.strip() or "个人简历", content, doc_type)

    def list_documents(self) -> list[dict[str, Any]]:
        return self._document_repo.list_all()

    def get_document(self, document_id: int) -> Optional[dict[str, Any]]:
        return self._document_repo.get(document_id)

    def delete_document(self, document_id: int) -> None:
        self._document_repo.delete(document_id)

    def resume_materials_text(self) -> str:
        """汇总分散在各处的档案素材，作为 LLM 撰写简历的输入文本。"""

        return str(self._profile_service.collect_resume_materials()["text"])
