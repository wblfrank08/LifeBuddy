"""Agent 初始化、模型接入、依赖注入上下文 (Deps)。

模型指向 OpenAI 兼容端点（默认 agnes-3.0-flash，见 config.py），
并显式设置请求超时（120 秒）。

问答提供两个入口，各自使用独立的 System Prompt：
- 对话页（默认）：自然对话，自由聊天、可用通用知识、自动建档；
- 知识库页「知识查询」弹窗：严格只依据上传的知识库作答，无资料时由代码层
  拦截为固定文案，且不影响对话页。
简历生成走独立的「简历撰写 Agent」，按「职位名称 + 职位 JD + 档案素材」定制，
只输出姓名、求职意向、个人评价、技能与证书、工作项目经历五个部分；工作项目
经历全量列出素材中的所有经历与项目，按 JD 定制规则用岗位术语组织描述、
突出数据化成果，与上述两个入口互不影响。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 复用 PydanticAI / OpenAI SDK 自带的 HTTP 客户端，不单独安装依赖
import httpx2 as httpx

import config
from services.chat_service import ChatService
from services.document_service import DocumentService, strip_code_fence
from services.knowledge_service import KnowledgeService
from services.profile_service import ProfileService
from services.search_service import SearchService


@dataclass
class Deps:
    """依赖注入容器：封装多个 Service 实例供 Agent 工具使用。"""

    chat_service: ChatService
    profile_service: ProfileService
    document_service: DocumentService
    search_service: SearchService
    knowledge_service: KnowledgeService
    # 本轮确定性预检索命中的知识库文件名：chat_view 每轮覆盖写入。
    # 仅用于「知识查询」模式下模型漏写来源行时做确定性兜底。
    last_knowledge_sources: list[str] = field(default_factory=list)


# 模板样式规范由两条生成路径（对话页直接输出 Markdown、简历中心输出 JSON）
# 共用，避免规范分散导致两处简历长得不一样。
RESUME_STYLE_RULES = """【简历样式规范（最高优先级，逐条严格遵守）】
输出结构固定为 5 个部分，不得增删、不得改名、不得调整顺序。写全哪些条目、
用什么词、如何排序，由《简历内容质量规范》与《岗位 JD 定制规则》决定，
本节只约束样式与符号。

一、固定结构（仅此 5 项）
   # 姓名
   求职意向：目标职位
   ## 个人评价
   ## 技能与证书
   ## 工作项目经历
   姓名取素材中的真实姓名，素材未提供时第一行写 `# 个人简历`；严禁出现本节清单
   之外的任何章节名。技能、经历章节素材为空时直接省略，不要写空标题。
   「求职意向」行的产生方式由本条消息末尾的输出规范指定，不得重复输出。

二、个人评价（只写「能力结论」，不写「项目技术」）
   本章只回答一件事：结合目标岗位，我具备哪些能胜任它的能力。HR 读完这
   3~4 行就应判断「这个人正是该岗位要找的人」。架构方案、技术组件、参数配置
   与实现过程一律放到「工作项目经历」，本章绝不展开。
   先写 1 段 80~120 字的总述：第一句直接给出匹配结论——用目标岗位名称与
   最对口的资历年限点明「我是这个方向的人」，并带出最亮眼的一个量化成果；
   随后用 2~3 个能力词概括最匹配岗位的核心能力（如大模型应用落地、复杂问题
   拆解、B 端产品化、数据驱动决策），不要用技术栈、框架名、项目名堆砌成
   通用自我介绍，也不要写成与岗位无关的自夸。
   再写最多 3 条以 `● ` 开头的优势要点，每条为「能力标签：能力陈述」结构：
   标签直接用目标岗位的核心能力词，冒号后写一句「具备／擅长／能够 + 能力描述
   + 能承担什么职责或解决什么问题」，并且只用一句最硬的量化结果作佐证；
   按「与岗位相关度 × 说服力」从强到弱排序，最强的一条放第一。
   句式示范（仅示范句式与取舍标准，内容必须取自素材，严禁照抄示例文字）：
     正例「AI 应用落地：能把大模型能力转化为可上线的业务系统，独立交付从
           需求到部署的完整闭环，单类故障判断耗时由数小时降至 5 分钟。」
     反例「本地 RAG 知识管道：自研从文档切分、向量化、混合检索到回答生成的
           完整链路，采用 512 token 切分与 64 token 重叠策略。」——这是项目
           技术实现，只能写进「工作项目经历」，不得出现在本章。
   严禁在本章写架构设计、实现方案、技术组件、参数配置与开发过程；严禁空泛
   形容词、能力清单式罗列、职责复述与项目技术复述。

三、技能与证书
   每行一条 `分类：内容`，分类按与目标岗位的相关性排序
   （如 产品与设计 / 技术与开发 / 数据与技术）；
   最后一行必须是 `相关证书：...`，无证书写「待补充」。

四、工作项目经历（素材里有几段写几段，不删减）
   按时间倒序，每段经历首行为 `公司名 ｜ 职位　　YYYY.MM ~ YYYY.MM`；
   首行下先写一句经历概述，用目标岗位的专业术语点明职责范围与主要产出；
   素材含多个项目时用 `1) 项目名`、`2) 项目名` 编号分组，项目之间不得混排，
   每个项目名后先写一句项目概述（项目做什么、解决什么问题、在其中承担什么
   角色），再列要点；概述与要点都要用 JD 的高频专业词组织，把最能证明岗位
   核心要求的证据放在最前。
   每个分点以 `▪ ` 开头，且必须以「标签：」开头，标签取自：
   背景与定位 / 痛点与定位 / 需求与方案 / 架构设计 / 技术攻坚 / 核心功能 /
   技术保障 / 效果评估与迭代 / 落地与迭代 / 用户验证 / 产品标准化建设 / 成果；
   每段经历或每个项目的最后一条必须是 `成果：`，含真实量化数字或具体产出物，
   且是本组最有分量的一条。
   不设经历数、项目数与要点数上限：档案素材中的每段任职经历、每个项目必须
   全部列出，不得整段或整个项目删除；与岗位弱相关的条目仍保留、精简带过并
   排在后面，只调整排序与详略。全量列出不等于流水账：每条都要写成
   「动作 + 结果 / 产出」的书面语，严禁把档案条目逐条改写后照搬。
   没有项目分组的经历直接写 `▪ ` 要点。

五、语言与事实边界
   口语化素材必须重写为专业、简洁的书面语；合并语义重复项；使用「负责、主导、
   设计、搭建、优化、交付、推动」等动宾结构；保持第三人称客观陈述，禁止「我」
   「我觉得」等第一人称口语。严格基于给定素材，禁止编造公司名、时间、数字、
   技术栈、学历与业绩；「待补充」只用于「相关证书」无内容时，其余章节缺料时
   直接删条目，不得用「待补充」凑数，也不要虚构内容凑字数。

六、严禁把生成过程写进简历（最高优先级）
   「对照 JD 提炼要求、逐条找证据、映射 JD 条款」只是你的内部推理，用来决定
   写哪几条、怎么排序、用哪些专业词；严禁把推理、映射关系或面向 JD 的说明
   写进任何章节、任何字段。全文（含个人评价总述、● 要点、技能行、▪ 要点）
   严禁出现以下字样：「对应 JD」「对应岗位要求」「JD 要求」「岗位要求」
   「匹配 JD」「满足 JD」「针对 JD」「根据 JD」「符合…要求」「JD 中」「本岗位」
   「招聘要求」，也严禁出现分析过程、思路说明、编号条款或任何解释性文字。
   正文只呈现最终结论与事实，读者不应看出 JD 的存在——简历是给 HR 看的
   成品，不是生成过程的记录。"""


RESUME_CONTENT_RULES = """【简历内容质量规范（决定写全、怎么写，与样式规范同时遵守）】
核心目标：把档案素材中的任职经历与项目**全部、完整**列进简历，并让每一条描述
都用目标岗位的专业语言组织，做到优点突出、与岗位强相关、有数据支撑；
不删条目、不设数量上限，但严禁把档案素材原样铺开成流水账。

一、全量列出（最高优先级，先列全再润色）
1. 动笔前先通读《档案素材》，把其中每一段任职经历、每一个项目都列入简历，
   一项都不能漏；不再按价值或相关度淘汰整段经历或整个项目，只调整排序与详略。
2. 不设数量上限：工作项目经历有几段任职经历就写几段、每段有几个项目就写
   几个项目、每个项目有几条证据就写几条要点，绝不为省篇幅而删条目。
   个人评价总述 80~120 字、优势要点最多 3 条，技能与证书最多 4 行（含
   「相关证书」行）——这两章的限量保持不变。
3. 同一成果只在最有说服力的一处出现：个人评价给能力匹配结论（只用一句量化
   结果佐证），工作项目经历给技术实现与过程证据；严禁两处换措辞重复同一件
   事，严禁把一段经历拆成多条重复要点。

二、优点突出（把最强证据放在最显眼的位置）
1. 个人评价总述第一句直接用 JD 的岗位名称与核心要求给出匹配结论，并点出
   「最匹配 JD 的资历 + 最亮眼的一个量化成果」；优势要点按 JD 相关度挑出
   最多 3 条能力结论，每条写「具备什么能力 + 能承担什么职责 / 解决什么问题
   + 一句量化佐证」，是可验证的强项，不写「沟通能力强」这类空泛形容词，
   不写能力清单，也不写架构、参数、组件等技术细节。
2. 任职经历按时间倒序全量列出（不得因相关度低而删除或提前）；每段经历内部的
   项目、每个项目的要点都按「说服力 × 与 JD 相关度」从强到弱排序，带数据、
   与 JD 最相关的那条放第一条；`成果：` 必须是本组最有分量的一条。
3. 项目名、职位沿用素材里的真实名称，不要用「参与」「负责」等弱动词开头。
4. 与 JD 的贴合体现在「选哪几条、用 JD 的哪个专业词做标签、把哪条排最前」，
   不体现在文字声明里：个人评价的每条优势都写成一句可直接阅读的简历书面语，
   只讲「我具备什么能力 + 一句量化佐证」，不得夹带任何面向 JD 的说明、分析
   或映射过程（禁用字样详见样式规范第六节）。与 JD 无关的能力再亮眼也不写
   进本章。
5. 章节分工（硬约束）：个人评价只写「能力结论」——我具备什么能力、能胜任
   什么职责；工作项目经历才写「技术实现与过程证据」——怎么做、用了什么
   方案与参数。架构设计、实现方案、技术组件、参数配置、开发过程一律不得
   写进个人评价，也不得把同一条技术细节换个说法在两处重复出现。
6. 描述往岗位靠：从 JD 提炼高频专业术语与核心要求作为内部用词依据，把经历
   概述、项目概述与要点都改写成该岗位领域的专业表达；最能证明核心要求的
   证据排最前、写足数据与产出，弱相关条目排后、精简带过但必须保留。贴合
   只体现在选词、排序与详略，严禁写出面向 JD 的说明或映射关系（禁用字样
   见样式规范第六节），严禁把 JD 要求改写成用户已有经历。

三、数据支持（硬要求，不得以「尽量」敷衍）
1. 每条 ▪ 要点必须是「动作 + 可验证结果」，公式：动词（主导 / 搭建 / 优化 /
   交付 / 推动）+ 做了什么 + 量化结果；优先写百分比、倍数、金额、时长、
   用户量 / 流量 / 规模、团队人数、效率提升等真实数字。
2. `成果：` 行必须含具体数字或具体产出物（上线的系统、交付的 PRD、落地的
   规范、获得的专利 / 奖项），绝不写成职责复述或「效果良好」「显著提升」空话。
3. 素材确实没有数字时，写清可验证的范围与产出（如「覆盖 3 条业务线的结算
   流程」「交付 12 份 PRD 并推动上线」），仍禁止编造任何数字、公司名、时间、
   技术栈与业绩。
4. 既无数据、又无具体产出、又与岗位弱相关的要点保留但精简为一行、排在最后，
   不得删除，也不要编造数字去补。

四、禁止流水账（以下写法一律不得出现）
1. 按时间顺序平铺「某年做了某事」、把档案条目逐条改写后照搬。
2. 「参与了 / 负责了 / 协助了 / 接触过 / 熟悉 / 了解」开头却不说结果与数据的句子。
3. 大段背景铺垫、岗位职责原文复述、空泛形容词堆砌、与 JD 无关的行业常识或通用自评。
4. 把 JD 要求改写成用户已有经历；把多段经历合并成一段。
5. 全文不得出现解释、开场白、总结、备注、分析过程、占位符字段名或代码块围栏。
6. 个人评价里出现架构方案、技术组件、参数配置、实现过程等项目技术描述
   （如混合架构、向量化、切分策略、校验机制、部署链路）——这些只属于
   「工作项目经历」。
7. 任何提及 JD、岗位要求、匹配关系的字样（禁用清单见样式规范第六节）；
   简历是给 HR 看的成品，不是生成过程的记录。"""


RESUME_JD_RULES = """【岗位 JD 定制规则（决定选什么、怎么排序，最高优先级）】
目标：让简历读起来就是「为这个岗位准备的」——用词、排序与取舍都围绕 JD 展开。

一、先做匹配分析，再动笔（纯内部推理，绝不输出分析过程）
1. 从《目标职位 JD》中提炼 3~5 项核心要求：核心职责、必备技能、加分项、
   经验年限、高频专业术语 / 关键词。
2. 逐条到《档案素材》里找证据：哪段经历、哪个项目、哪项成果、哪个数字能证明
   「我具备这项要求」；找不到强证据的条目照样保留，只是精简带过、排后。
3. 全部条目按「证据强度 × 与 JD 相关度」排序，证据最强、最贴 JD 的排最前，
   作为个人评价与工作项目经历的主体；排序不等于删除，弱相关条目仍需列出。

二、按 JD 组织内容
1. JD 的高频术语要自然出现在个人评价、技能行与要点里（用它本来的专业说法），
   但只能在既定的章节与标签体系内重组，不得为此新增章节或改动样式。
2. 与 JD 强相关的经历 / 项目重点展开（数据、规模、结果写足）；弱相关的精简
   为一行并排在后面；素材里的经历与项目一律保留，不得整段删除，不平均用力
   但也不遗漏条目。
3. 技能行按与 JD 的相关度排序，把 JD 点名的技能放第一行。
4. 若 JD 强调某项能力而素材里证据薄弱，个人评价里宁可不写，也不用空话或
   编造去补；但工作项目经历中的对应条目仍须列出。
5. 「个人评价」是 HR 第一眼决定是否约面的段落：把匹配分析里最强的 2~3 条证据
   转写成「能力结论」前置到这里——用 JD 的专业词做能力标签，写清「具备什么
   能力 / 能承担什么职责」，只用一句量化结果佐证，按相关度从强到弱排列，
   让 HR 读完直接得出「这就是岗位要的人」；技术实现与过程细节留给「工作项目
   经历」，不得写进本章；与 JD 无关的能力一律不进本章。
6. 匹配分析只用于「用哪个词、怎么排序、写多详」，绝不写进正文；每条内容
   都是可直接阅读的简历事实句，不得出现任何面向 JD 的说明或映射关系
   （禁用字样详见样式规范第六节）。
7. 工作项目经历必须全量列出：素材里有几段任职经历、几个项目就写几段、几个，
   不得因相关度低而删除；每个项目写清项目概述与要点，全部用 JD 的高频专业
   术语组织语言，把能证明核心要求的证据写足、排最前，弱相关的排后、精简
   带过。贴合只体现在选词、排序与详略，绝不体现在文字声明里。

三、事实边界
1. 只能用《档案素材》里真实存在的内容：JD 里的要求不等于用户已经具备，
   严禁把岗位要求、技能关键词直接改写成用户的经历或技能；档案里没有的，
   不要用虚构内容去凑，直接省略。
2. 未提供 JD 时，仍按《简历内容质量规范》全量列出所有经历与项目，生成一份
   专业、以数据与成果为主线的通用简历，不要因为没 JD 就放宽数据要求。
3. 《目标职位》里的职位名称仅用于判断岗位方向；「求职意向」由系统在标题下方
   确定性注入，你不要输出该行，也不要输出任何以「求职意向」「应聘职位」
   开头的文本。"""


RESUME_FORMAT_RULES = (
    RESUME_STYLE_RULES
    + "\n\n"
    + RESUME_CONTENT_RULES
    + "\n\n"
    + RESUME_JD_RULES
    + """

【简历输出方式（对话页）】
只输出 Markdown 简历正文：第一行 `# 姓名`（写素材中的真实姓名，无姓名写
`# 个人简历`），第二行写 `求职意向：目标职位`（从用户需求或素材推断）；
确实无法推断目标职位时省略该行。输出前确认素材中的每段经历、每个项目都已
列出，只按岗位相关度调整排序与详略，不得删除任何条目，且只保留固定结构中的
章节；全部描述用岗位领域的专业术语组织。严禁把「对照 JD 找证据」的分析过程
或面向 JD 的说明写进正文（禁用字样见样式规范第六节），严禁输出解释、开场白、
总结、代码块围栏、JSON、表格代码或任何与简历无关的文字，
也严禁输出「应聘职位」字样。"""
)


# 知识库无相关资料时的固定回复。提示词与 UI 代码层共用同一常量，保证逐字一致。
NO_KNOWLEDGE_REPLY = "目前知识库并没关于这个问题的答案哟。"

KNOWLEDGE_SOURCE_RULES = f"""【知识来源与回答边界（最高优先级，必须严格遵守）】
1. 你的回答只能来自本条消息开头《本轮检索资料》中的【知识库资料】。
   除此之外，严禁使用任何通用知识、常识、经验或模型自带知识去补充、推断或作答；
   即便你「恰好知道」答案，只要资料里没有，就必须判定为「没有相关知识」。
2. 若【本轮检索到的知识库文件】为「无」、或【知识库资料】为空、或资料与问题无关
   或不足以回答，则**只输出下面这一句，逐字照抄，不得增减任何字符**（不要解释、
   不要客套、不要来源行、不要调整标点）：
   {NO_KNOWLEDGE_REPLY}
3. 若用到了知识库资料，回答的**最后一行**必须是来源行，且其后不得再有任何文字、
   标点或空行；来源行逐字照抄以下格式，严禁改写、严禁包裹代码块、严禁放到回答中间：
   > 知识来源：{{知识库名称}}
   {{知识库名称}} 取自【本轮检索到的知识库文件】清单，不得编造；命中多个文件时用
   「、」按清单原顺序连接。
4. 严禁编造文件名、数字、人名或事实；严禁原样复读检索片段、字段名、JSON 或整段
   照抄资料清单。必须把资料转化为自然、易读的人类语言后输出。"""


# 档案工具注册在对话 Agent 上（见 tools.py）。自然对话模式注入本规范用于自动
# 建档；知识查询模式不注入，避免用档案/通用知识回答知识库问题。
PROFILE_RULES = """【档案更新与分类规范（重要）】
1. 用户透露个人信息时，先用 get_profile 查看已有档案；它返回的每条记录都带
   `id=编号`，后续更新必须带上这个编号。
2. 如果用户是在**更正或补充**已有信息（例如「我改名了」「其实是 3 年不是 2 年」
   「我换公司了」「不是 Python 是 Java」），必须调用 update_profile 修改那条
   已有记录，严禁再新增一条造成重复与前后矛盾。
3. 用户明确要求忘掉 / 删除某条信息时，调用 delete_profile。
4. save_profile 只用于此前完全没有记录过的全新信息；它内部也会自动合并
   高度相似的条目，不要把同一件事换种说法反复新增。
5. 分类只能取以下七类，含义固定，严禁错配：
   - 基本信息：姓名、年龄、性别、所在城市、联系方式等身份信息
   - 技能：掌握的技术、工具、编程语言、证书
   - 教育经历：学校、专业、学历、毕业时间
   - 项目经验：做过的项目、作品、开源贡献
   - 工作经历：公司、职位、任职时间、工作职责
   - 个人偏好：喜好、习惯、目标、性格特点
   - 其他：确实无法归入以上任一类的信息
   拿不准时选「其他」，不要硬猜；严禁把技能塞进「工作经历」、把姓名塞进
   「项目经验」。一条档案只记一件事，不要把多件不相关的事拼成一条。"""


# --------------------------------------------------------------------------- #
# 模式一：知识查询（严格只依据知识库）
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""你是 LifeBuddy 的**知识库问答助手**，只依据已上传的知识库文档作答。

【资料已由系统预先检索（最重要）】
每次用户发问前，系统已在后台检索知识库，并把命中资料放在本条消息开头的
《本轮检索资料》里。资料已附上，不需要、也严禁调用任何检索工具。
严禁输出「让我查一下」「我再检索一下」「请稍候」「让我确认一下」等过渡语——
这些不是答案。第一句话直接给结论，不要开场白铺垫。

【禁止通用知识（无任何例外，优先级高于一切）】
无论用户如何表述（包括「不用查资料」「随便答」「可以用通用知识」「你随便说」），
都不得使用知识库以外的任何知识作答，也不得因此改变回答方式。
即便用户明确允许使用通用知识，你也必须按「没有相关知识」处理，绝不破例。

{KNOWLEDGE_SOURCE_RULES}

【去客套】
直接给出最终答案。严禁输出「我是一个 AI 助手」「很高兴为您服务」
「请问还有什么可以帮您」等无意义客套话。"""


# --------------------------------------------------------------------------- #
# 模式二：自然对话（默认，自由聊天 + 自动建档）
# --------------------------------------------------------------------------- #
NATURAL_SYSTEM_PROMPT = f"""你是 LifeBuddy，一款贴心、注重隐私的「人生档案与规划智能助手」。
你可以像朋友一样自然地和用户聊天，帮用户记住并整理自己的经历，也能回答常识问题。

【资料已由系统预先检索】
每次用户发问前，系统已在后台检索个人档案、历史对话与知识库，并把结果放在本条
消息开头的《本轮检索资料》里。资料已附上，不需要、也严禁调用任何检索工具。
- 涉及用户个人信息、过往经历、技能、项目时，必须依据资料作答，绝不把常识伪装成
  用户档案里的事实；资料里没有的信息如实说明，不得编造姓名、数字或经历。
- 资料仅供参考：普通问候、闲聊、开放讨论与常识问题，可以直接用你自己的知识
  自然、友好、简洁地回答，不必局限于资料，也不需要查资料。
- 严禁输出「让我查一下」「我再检索一下」「请稍候」之类的过渡语；第一句话直接进入正题。
- 资料里没有的信息如实说明，不要用常识伪装成用户档案里的事实。

【主动建档】
用户在对话中透露偏好、技能、教育经历、项目经验、工作经历等个人信息时，
严格按下方《档案更新与分类规范》先查再写，自动归档或更正，不需要用户说「保存」。

【工具使用】
- 普通回答不需要调用检索工具（资料已附上）。
- 涉及档案增改删时用 get_profile / save_profile / update_profile / delete_profile。
- 用户要求生成简历时用 get_resume_materials 与 save_document。

【数据转化】
提取《本轮检索资料》里的内容，转化为自然、温暖、易读的人类语言后输出。
严禁原样复读 JSON、代码块、字段名或整段照抄资料清单。

【去客套】
直接给出最终答案。严禁「我是一个 AI 助手」「很高兴为您服务」
「请问还有什么可以帮您」等无意义客套话。

{PROFILE_RULES}

【简历】
用户要求写简历时，先用 get_resume_materials 汇总素材，按《简历样式规范》
与《简历内容质量规范》把素材中的每段任职经历、每个项目全部列出，用目标岗位
的专业术语组织描述、把相关证据排在最前，再调用 save_document 保存，最后把
简历正文直接输出给用户。不得因精简而遗漏任何经历或项目。

{RESUME_FORMAT_RULES}"""


RESUME_SYSTEM_PROMPT = (
    """你是 LifeBuddy 的简历撰写助手。把零散的个人档案素材，结合用户填写的
「职位名称」与《目标职位 JD》，严格按下方《简历样式规范》重组为结构化简历数据。

"""
    + RESUME_STYLE_RULES
    + """

【输出格式（最高优先级，严格执行）】
你只能输出**一个 JSON 对象**，不要任何解释、前言、分析过程、代码块围栏或客套话。
顶层字段固定如下，字段与「个人评价 / 技能与证书 / 工作项目经历」三个章节一一对应，
所有文本值均为中文；jobs / projects 内部的字段按下表填写，严禁输出顶层清单
之外的字段，也不要在 JSON 里写注释或解释：
- name: string                姓名；素材未提供时填空字符串 ""，严禁编造
- summary: string             个人评价总述，80~120 字，一段，不要换行；
                              第一句用岗位名称与资历年限给出匹配结论并带出
                              最亮眼的一个量化成果，随后用 2~3 个能力词概括
                              核心能力；只写能力结论，不写技术栈、架构方案、
                              参数配置或项目过程
- summary_points: string[]    最多 3 条 "能力标签：能力陈述"；标签直接取 JD 的
                              核心能力词，冒号后写「具备／擅长／能够 + 能力
                              描述 + 能承担什么职责或解决什么问题」，只用一句
                              量化结果佐证；严禁写架构设计、实现方案、技术
                              组件、参数配置与开发过程（这些属于 jobs 的要点），
                              严禁出现任何面向 JD 的说明、分析或映射字样
                              （禁用清单见样式规范第六节）；
                              按「与 JD 相关度 × 说服力」从强到弱排序
- skill_lines: string[]       技能与证书，最多 4 行，每条 "分类：内容"；
                              按与 JD 相关度排序，最后一条必须是
                              "相关证书：..."，无证书写 "待补充"
- jobs: object[]              工作项目经历，素材里有几段任职经历就写几段，
                              按时间倒序，一段都不能漏；每段结构：
    {"company": "公司名", "position": "职位", "period": "2026.05 ~ 至今",
     "overview": "一句经历概述：用岗位术语写清职责范围与主要产出",
     "projects": [{"name": "项目名",
                   "overview": "一句项目概述：做什么、解决什么问题、我的角色",
                   "points": ["标签：内容", ...]}],
     "points": ["标签：内容", ...]}
    · 素材里每个项目都要列出，不设数量上限，用 name 作编号标题；
      无项目分组时 projects 留空数组 []，要点直接放在 points；
    · 要点不设条数上限：该项目在素材里有几条可写的证据就写几条，
      按「说服力 × 与岗位相关度」降序排列，弱相关的排后但不得删除；
    · 每条写成「动作 + 量化结果」，优先带百分比、倍数、规模、时长等数字；
      最后一条必须是 "成果：..."，含真实数字或具体产出物；
    · overview 与要点全部用 JD 的高频专业术语组织，把最能证明核心要求的
      证据写足、排最前；只写事实与结果，严禁出现面向 JD 的说明、分析或
      映射字样（禁用清单见样式规范第六节）；
    · 无数据、无具体产出的条目精简为一行、排在最后，不得删除；
      严禁编造素材里没有的公司名、时间、数字、技术栈与业绩。

空数组 [] / 空字符串 "" 表示该部分省略；不要输出空标题、占位说明或后台字段名。
严禁输出「求职意向」「应聘职位」相关字段，或任何以它们开头的文本——
「求职意向」由系统在标题下方确定性注入。

"""
    + RESUME_CONTENT_RULES
    + "\n\n"
    + RESUME_JD_RULES
    + """

【工作方式】
只需填充上述 JSON，不调用任何工具，不向用户提问，也不要输出分析过程：
先按《岗位 JD 定制规则》提炼专业术语与核心要求，把素材中的每段经历、
每个项目全部登记（一项不漏）；再按《简历内容质量规范》用岗位术语润色、
按相关度排序、写足数据与产出；最后按模板既定的章节与标签体系填入对应字段。
全量列出不等于流水账，每条都要写成「动作 + 结果 / 产出」的书面语。"""
)


# --------------------------------------------------------------------------- #
# 结构化简历数据模型与模板化确定性排版
# --------------------------------------------------------------------------- #
class ResumeProject(BaseModel):
    """「工作项目经历」中的一个编号项目。

    ``overview`` 承载「项目做什么 / 解决什么问题 / 我的角色」一句概述，
    避免项目只剩零散要点、读起来不完整。
    """

    name: str = ""
    overview: str = ""
    points: list[str] = Field(default_factory=list)


class ResumeJob(BaseModel):
    """「工作项目经历」中的一段任职经历（按时间倒序）。"""

    company: str = ""
    position: str = ""
    period: str = ""
    overview: str = ""
    projects: list[ResumeProject] = Field(default_factory=list)
    points: list[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    """简历的结构化契约：字段与「个人评价 / 技能与证书 / 工作项目经历」一一对应。

    用嵌套 BaseModel 精确表达「经历 → 项目 → 标签化要点」的层级，
    再由 ``format_resume`` 确定性排版，避免模型自由发挥导致样式漂移。
    """

    name: str = ""
    summary: str = ""
    summary_points: list[str] = Field(default_factory=list)
    skill_lines: list[str] = Field(default_factory=list)
    jobs: list[ResumeJob] = Field(default_factory=list)


_BULLETS = ("▪", "●", "⚫", "•", "·", "-", "*")

# 模型偶发把「对应 JD…要求」这类内部推理前缀照抄进要点时，代码层确定性剔除，
# 作为提示词禁用清单之外的硬兜底，保证交付正文绝不含面向 JD 的元话语。
_JD_META = re.compile(
    r"(?:对应|针对|满足|符合|匹配)\s*(?:JD|jd|岗位|本岗位|招聘)"
    r"[^，,。;；]{0,40}?要求\s*[，,、:：]?\s*"
)


def _clean(value: object) -> str:
    """去掉要点符号与首尾空白，并剔除模型可能照抄的 JD 论证前缀。"""

    text = str(value if value is not None else "").strip()
    while text and text[0] in _BULLETS:
        text = text[1:].lstrip()
    return _JD_META.sub("", text).strip()


def _points(items: list[str], default_label: str) -> list[str]:
    """清洗要点，并兜底补「标签：」，保证与模板的标签化写法一致。"""

    result: list[str] = []
    for item in items:
        text = _clean(item)
        if not text:
            continue
        if "：" not in text[:20] and ":" not in text[:20]:
            text = f"{default_label}：{text}"
        result.append(text)
    return result


def _bold_label(line: str) -> str:
    """把「分类：内容」渲染为模板的加粗分类行。"""

    text = _clean(line)
    for separator in ("：", ":"):
        if separator in text:
            label, _, rest = text.partition(separator)
            if label.strip():
                return f"**{label.strip()}：** {rest.strip()}".rstrip()
    return text


def format_resume(data: ResumeData) -> str:
    """纯函数：把结构化简历数据确定性排版为固定结构的 Markdown 正文。"""

    lines: list[str] = [f"# {_clean(data.name) or '个人简历'}", ""]
    # 「求职意向：xxx」由 inject_job_intention 在标题下方确定性注入

    summary = _clean(data.summary)
    summary_points = _points(data.summary_points, "能力")
    if summary or summary_points:
        lines += ["## 个人评价", ""]
        if summary:
            lines += [summary, ""]
        for point in summary_points:
            lines += [f"● {point}", ""]

    skill_lines = [_bold_label(line) for line in data.skill_lines if _clean(line)]
    if skill_lines:
        lines += ["## 技能与证书", ""] + skill_lines + [""]

    jobs = [
        job
        for job in data.jobs
        if _clean(job.company)
        or _clean(job.overview)
        or job.points
        or job.projects
    ]
    if jobs:
        lines += ["## 工作项目经历", ""]
        for job in jobs:
            head = " ｜ ".join(
                part for part in (_clean(job.company), _clean(job.position)) if part
            )
            period = _clean(job.period)
            if head:
                lines.append(f"**{head}**　　{period}".rstrip())
                lines.append("")
            elif period:
                lines += [period, ""]
            job_overview = _clean(job.overview)
            if job_overview:
                lines += [job_overview, ""]
            numbered = 0
            for project in job.projects:
                name = _clean(project.name)
                overview = _clean(project.overview)
                project_points = _points(project.points, "工作内容")
                if not name and not overview and not project_points:
                    continue
                numbered += 1
                if name:
                    lines += [f"{numbered}) {name}", ""]
                if overview:
                    lines += [overview, ""]
                lines += [f"▪ {point}" for point in project_points]
                lines.append("")
            direct = _points(job.points, "工作内容")
            if direct:
                lines += [f"▪ {point}" for point in direct] + [""]

    return "\n".join(lines).strip()


def parse_resume_json(text: str) -> ResumeData | None:
    """纯函数：从模型文本中容错解析结构化简历。

    模型可能附带 ```json 围栏或前后缀文字，这里先去围栏，再截取最外层
    ``{...}`` 用 Pydantic 校验；解析失败返回 ``None``（由调用方回退为原文）。
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return ResumeData.model_validate_json(match.group(0))
    except Exception:  # noqa: BLE001 - 模型偶发 JSON 不合规时回退原文
        return None


@dataclass
class AgentSession:
    """一轮对话专用的模型会话。

    ``httpx.AsyncClient`` 的连接池与 anyio 同步原语会绑定到创建它的 event loop。
    Streamlit 每轮交互由 ``asyncio.run`` 新建并关闭 loop，因此该会话必须在
    同一个 loop 内创建、使用并关闭，绝不能放进 ``@st.cache_resource`` 长期缓存。
    """

    agent: Agent[Deps, str]
    http_client: httpx.AsyncClient
    # 当前会话所用 System Prompt。PydanticAI 在 message_history 非空时不会再
    # 自动注入 Agent 的 system_prompt，需要由 consume_stream 显式前置。
    system_prompt: str = ""

    async def aclose(self) -> None:
        """在当前 loop 内释放连接，避免 loop 关闭后残留已绑定的资源。"""

        await self.http_client.aclose()


def build_agent_session(
    system_prompt: str = SYSTEM_PROMPT,
    with_tools: bool = True,
) -> AgentSession:
    """组装模型会话、超时控制与工具，返回绑定当前 event loop 的会话。

    构造开销仅为若干 Python 对象与工具注册（不加载任何模型），可每轮调用。
    ``system_prompt`` 用于切换自然对话 / 知识查询 / 简历撰写三种模式；
    ``with_tools=False`` 用于不带工具的「简历撰写 Agent」。
    """

    http_client = httpx.AsyncClient(timeout=config.LLM.timeout)
    provider = OpenAIProvider(
        base_url=config.LLM.base_url,
        api_key=config.LLM.api_key,
        http_client=http_client,
    )
    model = OpenAIChatModel(config.LLM.model_name, provider=provider)

    agent: Agent[Deps, str] = Agent(
        model,
        deps_type=Deps,
        system_prompt=system_prompt,
    )

    if with_tools:
        # 延迟到函数内导入，避免 agent_core 与 tools 的模块级循环依赖
        from agent import tools

        tools.register_tools(agent)
    return AgentSession(
        agent=agent, http_client=http_client, system_prompt=system_prompt
    )


def prepend_system_prompt(system_prompt: str, history: list[Any]) -> list[Any]:
    """把当前模式的 System Prompt 显式放到 message_history 最前面。

    PydanticAI 在 ``message_history`` 非空时认为「历史里已含指令」，不会再注入
    Agent 的 ``system_prompt``。若不显式前置，自然对话第 2 轮起会丢失自动建档
    规则，知识查询第 2 轮起会丢失「禁止通用知识 / 来源行」等严格规则。
    """

    if not system_prompt:
        return history
    return [
        ModelRequest(parts=[SystemPromptPart(content=system_prompt)]),
        *history,
    ]


async def generate_resume(prompt: str, deps: Deps) -> str:
    """用专门的简历 Agent 生成简历正文（可结合目标职位 JD 定制）。

    会话在当前 event loop 内创建并在 finally 中关闭，与对话页共用同一套
    生命周期规则，避免 ``RuntimeError: Event loop is closed``。

    生成策略：模型按 ``RESUME_SYSTEM_PROMPT`` 输出 JSON，这里流式聚合文本后
    用 ``parse_resume_json`` 校验为 ``ResumeData``，再 ``format_resume`` 确定性
    排版为 Markdown。解析失败（模型未严格返回 JSON）时回退为去除围栏的原文，
    保证主流程不中断。
    """

    session = build_agent_session(
        system_prompt=RESUME_SYSTEM_PROMPT, with_tools=False
    )
    try:
        parts: list[str] = []
        async with session.agent.run_stream(prompt, deps=deps) as response:
            async for delta in response.stream_text(delta=True):
                if delta:
                    parts.append(delta)
        text = "".join(parts).strip()
        data = parse_resume_json(text)
        return format_resume(data) if data else strip_code_fence(text)
    finally:
        try:
            await session.aclose()
        except Exception:  # noqa: BLE001 - 关闭失败不覆盖本轮的生成结果
            pass
