"""ReAct Agent — 自主使用文档检索工具来探索和回答问题"""

from typing import Optional

from langchain.agents import create_agent

from app.agent.tools import list_documents, get_doc_tree, get_node_content
from app.agent.middleware import get_middlewares
from app.core.models import agent_model

SYS_PROMPT = """你是一个技术文档助手，帮助用户理解和查找本地文档中的信息。

## 可用工具

- **list_documents**: 列出所有可用的文档及其摘要，了解有哪些文档
- **get_doc_tree**: 获取指定文档的目录树（只有章节标题，无正文），用来浏览文档结构
- **get_node_content**: 获取指定节点的完整正文，阅读具体内容

## 检索策略

1. 先用 list_documents 了解有哪些文档可用
2. 用 get_doc_tree 浏览相关文档的结构，找到最相关的章节
3. 用 get_node_content 读取那些章节的正文（可以一次传入多个 node_id）
4. 如果信息不足，继续探索其他文档或章节
5. 根据检索到的内容如实回答，不要编造

## 行为准则

- 必须通过工具检索信息，不要凭空猜测
- 回答时标注信息来源（文档名称、章节路径）
- 如果找不到相关内容，如实告知用户
- 回答要简洁准确，直接回应用户的问题"""

_agent: Optional = None


async def get_agent():
    global _agent
    if _agent is None:
        tools = [list_documents, get_doc_tree, get_node_content]
        _agent = create_agent(
            agent_model,
            system_prompt=SYS_PROMPT,
            tools=tools,
            middleware=get_middlewares(),
        )
    return _agent
