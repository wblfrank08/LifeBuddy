"""简历中心视图：按目标职位生成简历、查看 Markdown 源码与导出。

点击「生成简历」弹出中等大小对话框，填写「职位名称」与「职位 JD」后，
由简历撰写 Agent 结合档案素材，严格按《个人简历-模板》的样式组织内容；
标题下方的「求职意向：xxx」由代码层确定性注入，不依赖模型是否记得输出。

简历正文经嵌套 ``ResumeData(BaseModel)`` 结构化校验后由 ``format_resume``
确定性排版为固定结构（姓名 / 求职意向 / 个人评价 / 技能与证书 /
工作项目经历），避免模型自由发挥导致样式漂移。工作项目经历全量列出素材中的
每段任职经历与每个项目，并带经历概述 / 项目概述，不再按篇幅删减条目。

生成成功后不单独展示结果预览——下方「已生成文档」已有记录，且记录内以
Markdown 源码（st.text）呈现，不调用 st.markdown 渲染富文本。

本页交互（生成入口、生成动作、文档列表与删除）封装在 ``st.fragment`` 中；
筛选 / 上传等局部交互由 Streamlit 自动触发 fragment 级重跑，避免整页重绘。
"""

from __future__ import annotations

import asyncio

import streamlit as st

import config
from agent.agent_core import Deps, generate_resume
from services.document_service import inject_job_intention
from ui.notify import toast_error, toast_success, toast_warning

_PENDING_KEY = "resume_generate_pending"
_TITLE_KEY = "resume_job_title"
_JD_KEY = "resume_job_jd"

RESUME_TASK_PROMPT = """请为下面的《目标职位》生成一份简历，输出一个 JSON 对象
（不要任何解释、前言、分析过程、代码块围栏或客套话）。

《目标职位》
职位名称：{job_title}
目标职位 JD：
{jd}

《生成要求（按优先级从高到低严格执行）》
1. 先对照 JD 提炼 3~5 项核心要求与高频专业术语，再到《档案素材》里逐条找证据；
   把素材中每一段任职经历、每一个项目全部列出，一项都不能漏，不再按相关度
   淘汰条目，只调整排序与详略。每条都写成「动作 + 结果 / 产出」的书面语，
   绝不按时间顺序把档案原样铺成流水账。
2. 「个人评价」是全篇重点，只写「结合目标岗位我具备哪些能力」，不写项目技术：
   总述 80~120 字，第一句直接给出匹配结论（资历年限 + 最对口方向 + 最亮眼
   量化成果），随后用 2~3 个能力词概括核心能力，不要用技术栈、框架或项目名
   堆砌；再写最多 3 条 ● 优势要点，标签取 JD 的核心能力词，冒号后写
   「具备 / 擅长 / 能够 + 能力描述 + 能承担什么职责或解决什么问题」，只用
   一句量化结果佐证，按「与岗位相关度 × 说服力」降序，最强的放第一条。
   严禁把架构设计、实现方案、技术组件、参数配置、开发过程写进本章——这些
   属于「工作项目经历」；与岗位无关的能力再亮眼也不写，严禁把 JD 要求改写
   成已有经历。
3. 全文与 JD 的贴合只通过选词、排序和详略体现：经历概述、项目概述与每条要点
   都用 JD 的高频专业术语组织，把最能证明核心要求、带数据的证据放在最前；
   与岗位弱相关的条目仍完整列出，只是排在后面、精简带过。匹配分析是内部推理，
   严禁把「对应 JD」「JD 要求」「岗位要求」「满足 JD」等面向 JD 的说明、
   映射关系或推理过程写进任何字段（完整禁用清单见系统提示《简历样式规范》
   第六节）——简历是给 HR 看的成品，不应让读者看出 JD 的存在。
4. 有数据支持：每条要点写成「动作 + 量化结果」，优先保留百分比、倍数、规模、
   时长、用户量等真实数字；「成果：」必须含数字或具体产出物，禁止「效果良好」
   「显著提升」这类空话，也禁止编造任何数字、公司名、时间与业绩。
   无数据、无产出的条目精简为一行并排在最后，但不得删除。
5. 全量列出、不设上限：工作项目经历有几段经历写几段、几个项目写几个、几条
   证据写几条要点；个人评价优势要点最多 3 条、技能行最多 4 行保持不变。
   同一成果只出现一次，不重复堆砌，不为凑篇幅删条目。
6. 只输出姓名、个人评价、技能与证书、工作项目经历四部分对应的字段；
   章节、符号、标签体系与字段结构严格按系统提示中的《简历样式规范》
   与《简历内容质量规范》执行，字段类型与顺序以系统提示为准；jobs / projects
   内部按系统提示的字段表填写，每段经历与每个项目都要写 overview（经历概述 /
   项目概述）与要点；空部分用空数组或空字符串，不要写空标题或占位说明，
   也不要输出顶层清单之外的字段。
7. 「求职意向」由系统在标题下方自动注入，你不要输出该字段，也不要输出任何以
   「求职意向」「应聘职位」开头的文本。

《档案素材》
{materials}
"""


@st.dialog("生成简历", width="medium")
def _resume_dialog() -> None:
    """收集职位名称与 JD，交由主页面执行生成（弹窗随即关闭）。"""

    st.caption("填写目标职位，AI 会结合你的档案素材组织语言并按简历格式规范输出。")
    job_title = st.text_input(
        "职位名称",
        placeholder="例如：高级 Python 工程师",
        max_chars=60,
        key="resume_job_title_input",
    )
    job_jd = st.text_area(
        "职位 JD",
        placeholder="粘贴招聘信息中的岗位职责与任职要求（2000 字以内）…",
        max_chars=2000,
        height=240,
        key="resume_jd_input",
    )
    # 不使用 st.form：弹窗（fragment）内 form 的提交状态在重跑时容易丢失，
    # 直接读取上方控件的当前值最稳，交互也更直观。
    submitted = st.button("立即生成", key="resume_submit", width="stretch", type="secondary")

    if not submitted:
        return

    title = (job_title or "").strip()
    if not title:
        toast_warning("请先填写职位名称，生成的简历中会自动加入「求职意向：{职位名称}」。")
        return

    st.session_state[_TITLE_KEY] = title
    st.session_state[_JD_KEY] = (job_jd or "").strip()
    st.session_state[_PENDING_KEY] = True
    # 弹窗由片段内按钮打开：关闭弹窗后由主页面消费 _PENDING_KEY 触发生成
    st.rerun()


@st.fragment
def _resume_section() -> None:
    """简历中心的局部交互区：生成入口、生成动作、文档列表与删除。"""

    deps: Deps = st.session_state["runtime"]["deps"]
    document_service = deps.document_service

    if st.button("生成简历", key="generate_resume", width="stretch", type="secondary"):
        _resume_dialog()

    # 弹窗提交后由本 fragment 执行真正的生成：此时弹窗已关闭，下方列表在本轮
    # 渲染时自然包含新简历，避免 dialog 内长耗时阻塞与渲染结构不一致。
    if st.session_state.pop(_PENDING_KEY, False):
        job_title = str(st.session_state.pop(_TITLE_KEY, ""))
        job_jd = str(st.session_state.pop(_JD_KEY, ""))
        jd = job_jd or "（未提供职位 JD，请生成一份通用但专业、结构完整的简历。）"
        prompt = RESUME_TASK_PROMPT.format(
            job_title=job_title,
            jd=jd,
            materials=document_service.resume_materials_text(),
        )
        with st.spinner("AI 正在结合职位 JD 汇总素材、组织语言并规范简历格式…"):
            try:
                raw = asyncio.run(generate_resume(prompt, deps))
                content = inject_job_intention(raw, job_title)
                if not content:
                    raise RuntimeError("模型未返回任何简历内容")
                document_service.save_resume("个人简历", content)
                toast_success("简历生成完毕")
            except Exception as exc:  # noqa: BLE001 - 连接失败/模型断片必须明确报错
                toast_error(
                    f"简历生成失败：{exc}。请确认网络可访问 "
                    f"{config.LLM.base_url}、config.py 中 LLM.api_key 已填写，"
                    "或修改模型地址。"
                )

    st.subheader("已生成文档")
    documents = document_service.list_documents()
    if not documents:
        st.caption("还没有生成过文档，点击上方按钮即可创建一份简历。")
        return

    for document in documents:
        document_id = document["id"]
        with st.expander(f"{document['title']} · {document['created_at']}"):
            full = document_service.get_document(document_id)
            if full is None:
                toast_error("文档内容读取失败。")
                continue
            st.caption("以下为 Markdown 源码，可通过下方按钮导出 .md 文件。")
            st.text(full["content"])
            st.download_button(
                "导出 Markdown",
                data=full["content"],
                file_name=f"{document['title']}.md",
                mime="text/markdown",
                key=f"export_document_{document_id}",
                width="stretch",
            )
            if st.button("删除文档", key=f"delete_document_{document_id}", width="stretch"):
                document_service.delete_document(document_id)
                # 列表在删除前已查询（见上方 list_documents），本帧渲染的仍是旧列表，
                # 故用 st.rerun(scope="fragment") 让片段局部重跑并重拉列表；
                # 该调用不会取消紧随其后的彩色 toast。
                toast_success("文档已删除")
                st.rerun(scope="fragment")


def render_documents() -> None:
    """渲染简历中心（无状态函数）。"""

    st.header("简历中心")
    st.caption(
        "由 AI 汇总档案素材、结合目标职位 JD 组织语言并规范格式生成简历，"
        "可查看源码并导出到本地。"
    )
    _resume_section()
