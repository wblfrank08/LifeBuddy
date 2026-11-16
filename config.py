"""LifeBuddy 全局配置。

所有与外部服务相关的地址、模型名、密钥与超时时间集中在本文件。

- LLM 默认指向 agnes-3.0-flash 的 OpenAI 兼容端点。请把 ``LLM.api_key``
  替换为你自己的 API Key（该请求会发往外部服务，请自行评估数据外发风险）。
- Embedding 仍为本地占位配置（bge 服务，默认 8081 端口）；未启动时会自动
  降级为纯关键词检索，不影响问答与知识库导入。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

# --------------------------------------------------------------------------- #
# 路径配置
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "lifebuddy.db"
UPLOAD_DIR = DATA_DIR / "uploads"
CONFIG_PATH = BASE_DIR / "config.json"

# --------------------------------------------------------------------------- #
# 读取json配置
# --------------------------------------------------------------------------- #
def load_json_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# 获取 JSON 数据
_json_data = load_json_config()
_llm_config = _json_data.get("llm_config",{})
_embedding_config = _json_data.get("embedding_config",{})


# --------------------------------------------------------------------------- #
# 本地大模型（llama.cpp，OpenAI 兼容接口）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LLMConfig:
    """模型后端配置。base_url 需指向 OpenAI 兼容的 /v1 端点。"""

    base_url: str = _llm_config.get("base_url", "http://127.0.0.1:8080/v1")
    # 请替换为你自己的 API Key。留空字符串会导致 OpenAI SDK 抛 Missing credentials
    api_key: str = _llm_config.get("api_key", "sk-")
    model_name: str = _llm_config.get("model_name", "agnes-3.0-flash")
    # 在线生成可能较慢，显式设置较长超时，防止长耗时任务断开连接
    timeout: float = 120.0


# --------------------------------------------------------------------------- #
# 本地向量化模型（bge-small-q8-zh-v1.5.gguf）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 服务配置。默认对应 llama.cpp server 的 /embedding 端点。"""

    base_url: str = _embedding_config.get("base_url", "http://127.0.0.1:8081/v1")
    embedding_path: str = "/embedding"
    # 首次请求可能触发模型加载，单独设置较短但足够的超时
    timeout: float = 30.0
    dimension: int = _embedding_config.get("dimension", 512)  # bge-small-zh-v1.5 输出维度


# --------------------------------------------------------------------------- #
# 检索与分块参数
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 320          # 每个文本块的最大字符数
CHUNK_OVERLAP = 64        # 滑动窗口重叠字符数
SEARCH_TOP_K = _json_data.get("top_k",5)          # 单路检索返回的候选数量
SHORT_QUERY_LENGTH = 3    # 查询词长度小于该值时强制走 LIKE 兜底
TOOL_CONTEXT_MAX_CHARS = 1200  # 工具返回给模型的上下文裁剪长度，避免挤爆小模型窗口
TURN_CONTEXT_MAX_CHARS = 2400  # 每轮确定性预检索注入模型的资料长度上限
# 简历素材需要容纳全部任职经历与项目，不能用上面的通用裁剪长度截断；
# 否则后半段项目根本进不了模型输入，简历必然缺项。
RESUME_MATERIALS_MAX_CHARS = 12000
RESUME_RELATED_MESSAGES_LIMIT = 50  # 简历素材附带的近期对话线索条数


# --------------------------------------------------------------------------- #
# 全局配置实例
# --------------------------------------------------------------------------- #
LLM = LLMConfig()
EMBEDDING = EmbeddingConfig()
