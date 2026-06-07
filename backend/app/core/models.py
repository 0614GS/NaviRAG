"""共享 LLM 模型实例 — 供 middleware、indexing 等模块复用"""

import os

import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "")

_model_kwargs = {"response_format": {"type": "json_object"}}

# Agent 中间件使用的模型
back_agent_model = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    model_kwargs=_model_kwargs,
)

summarize_model = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    model_kwargs=_model_kwargs,
)
