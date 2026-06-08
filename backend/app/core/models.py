"""共享 LLM 模型实例 — 所有 ChatOpenAI 统一管理

从 config.settings 显式读取 api_key / base_url，
不依赖模块导入时的环境变量顺序。
"""

import os
import dotenv
from langchain_openai import ChatOpenAI
from app.config import settings

dotenv.load_dotenv()


def _get_api_params() -> dict:
    """从 settings 获取 API 参数，确保不为空"""
    key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    base = settings.OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL", "")
    params = {}
    if key:
        params["api_key"] = key
        os.environ.setdefault("OPENAI_API_KEY", key)
    if base:
        params["base_url"] = base
        os.environ.setdefault("OPENAI_BASE_URL", base)
    return params


_api_params = _get_api_params()
_json_mode = {"response_format": {"type": "json_object"}}

# ReAct Agent 模型（自由文本输出，不加 JSON mode）
agent_model = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    **_api_params,
)

# 元数据提取 / 摘要模型（JSON 结构化输出）
extract_model = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    model_kwargs=_json_mode,
    **_api_params,
)

# 向后兼容别名
back_agent_model = agent_model
summarize_model = extract_model
