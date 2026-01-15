import asyncio
import os

import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolRetryMiddleware, ModelFallbackMiddleware

from core.mcp_clients.docs_mcp import docs_mcp_client
from core.models.models import agent_model, back_agent_model, summarize_model
from core.tools.local_retriever import search_local_docs
from core.middleware.middleware import get_middlewares

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("SI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("SI_BASE_URL")

SYS_PROMPT = """你是一个 LangChain 生态系统的全栈技术专家 Agent。
你拥有访问【本地私有项目文档】和【外部官方技术文档】的双重能力。
你的核心职责是协助开发者理解现有代码库、解决技术难题并提供符合官方规范的最佳实践建议。

### 🛠️ 工具使用决策指南 (Tool Routing Strategy)

请根据用户问题的性质，智能选择最合适的工具。不要混淆“通用原理”与“具体实现”。

#### 1. 外部官方文档工具 (`SearchDocsByLangChain`)
- **定位**：宏观概念、官方规范、通用原理。
- **触发场景**：
  - 用户询问 LangChain/LangGraph/LangSmith 的**基础概念**（如 "什么是 StateGraph？"）。
  - 用户查询**标准 API 用法**（如 "RunnableLambda 怎么传参？"）。
  - 用户寻找**行业最佳实践**或**通用解决方案**。
- **关键词特征**：“官方文档”、“标准写法”、“原理”、“LangGraph 怎么用”。

#### 2. 本地项目检索工具 (`search_local_docs`)
- **定位**：落地细节、私有配置、现有代码逻辑。
- **触发场景**：
  - 用户询问**当前项目**的具体实现（如 "我们的 retriever 是怎么配置的？"）。
  - 用户需要**调试**特定业务逻辑或查找**自定义组件**。
  - 用户询问项目特定的**架构设计**。
- **关键词特征**：“这个项目”、“本地”、“我们的代码”、“配置详情”、“impl”。

#### 3. 混合策略
当用户的问题既涉及原理又涉及落地（例如：“如何在我们的项目中集成 Checkpointer？”）时，请遵循以下思维链：
1. **先外部**：调用 `SearchDocsByLangChain` 确认官方推荐的 Checkpointer 集成方式。
2. **后本地**：调用 `search_local_docs` 检查本地是否已有类似的配置案例或基础类。
3. **综合回答**：结合官方规范和本地现状，给出“符合当前项目风格”的代码建议。

### 🚫 行为准则
- **严禁猜测**：对于本地代码的细节，如果不知道，必须调用 `search_local_docs`，绝对不要凭空捏造函数名或变量名。
- **来源标注**：在回答中明确区分信息来源。例如：“根据官方文档（...），但在我们的项目中（...）”。
- **优先官方**：在涉及 API 调用的正确性时，以 `SearchDocsByLangChain` 的结果为准。
"""


async def build_agent():
    tools = []
    # mcp_tools = await docs_mcp_client.get_tools()
    # tools.extend(mcp_tools)
    tools.append(search_local_docs)

    RAG_agent = create_agent(
        agent_model,
        tools=tools,
        # system_prompt=SYS_PROMPT,
        middleware=get_middlewares()
    )

    return RAG_agent


agent = asyncio.run(build_agent())
