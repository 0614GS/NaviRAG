import os
from typing import List, Literal

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from core.workflow.prompts import global_index
from core.workflow.states import State
from data.storage import doc_tree_store, node_content_store

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("SI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("SI_BASE_URL")

search_model = ChatOpenAI(
    model="MiniMaxAI/MiniMax-M2",
    temperature=0,
)

grade_model = ChatOpenAI(
    model="MiniMaxAI/MiniMax-M2",
    temperature=0
)


def select_docs(state: State):
    class output(BaseModel):
        doc_ids: List[str] = Field(description="选中的相关文档 ID 列表")
        reasoning: str = Field(description="简要说明为什么选择这些文档，以及它们如何覆盖用户的问题")

    # 1. 提取 Query
    query = state["query"]

    # 2. 优化后的系统提示词
    system_prompt = f"""你是一个专业的技术文档路由专家。
    你的任务是阅读【文档索引库】，并根据用户的【搜索意图】，挑选出所有可能包含答案的文档 `doc_id`。
    
    ### 角色逻辑：
    - 你不仅看关键词匹配，更要理解技术组件之间的依赖关系。
    - 采用“宁可稍微扩大范围，也不漏掉关键文档”的策略（High Recall）。
    
    ### 文档索引库 (Global Index):
    {global_index}
    
    ### 筛选准则：
    1. **直接关联**：文档的 `keywords` 或 `summary` 直接提到了问题中的技术名词。
    2. **场景关联**：用户描述的是一个场景（如“部署”），你需要关联到相关的 `cli`、`backends` 或 `quickstart` 等文档。
    3. **概念覆盖**：如果问题涉及底层原理，挑选 `overview` 或 `concepts` 类文档。
    
    ### 输出要求：
    - 必须返回 `doc_ids` 列表。
    - 如果用户的问题完全属于闲聊或与本技术栈毫无关系，请返回空列表 []。
    """

    # 3. 执行调用
    response = search_model.with_structured_output(schema=output).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户的搜索意图：'{query}'\n请给出最相关的文档列表。")
    ])

    print(f"--- [Router] 选定的文档: {response.doc_ids} ---")
    print(f"--- [Router] 理由: {response.reasoning} ---")

    return {"doc_ids": response.doc_ids}


def select_nodes(state: State):
    class output(BaseModel):
        node_ids: List[str] = Field(description="相关节点的node_id列表")

    query = state["query"]
    doc_ids = state["doc_ids"]
    if len(doc_ids) == 0:
        return Command(goto=END)
    # 拿到所有的tree
    trees = doc_tree_store.mget(doc_ids)
    # print(trees)
    system_prompt_list = [
        f"""你是一个文档导航助手。你的任务是从给定的【文档层级树】中，识别出与用户问题最相关的节点 ID（node_id）。
        检索规则：
        1. 层级理解：文档采用树状结构。如果父节点的主题相关，请深入查看其子节点（nodes）。
        2. 多点检索：如果你不能确定某一个node一定包含相关主题，返回多个相关的 node_id。
        3. 排除无关：如果某些节点显然不相关，请忽略它们。
        4. 如果你确定没有相关主题，返回空列表
    
        ### 当前文档层级树：
        {tree_structure}
    
        ### 注意事项：
        - 请仅从上方提供的树结构中选择存在的 `node_id`。
        - 如果树中没有相关内容，请返回空列表。
        """
        for tree_structure in trees]
    msg_list = [
        [SystemMessage(content=system_prompt),
         HumanMessage(content=f"用户的问题是：'{query}'。请给出最相关的 node_id列表。")]
        for system_prompt in system_prompt_list]
    # 调用模型
    response_list = search_model.with_structured_output(schema=output).batch(msg_list)
    full_node_ids = []
    for response in response_list:
        full_node_ids.extend(response.node_ids)
    # print(response)
    return {"node_ids": full_node_ids}


# LLM并行评价文档content批处理
def grade_node_content(state: State):
    class output(BaseModel):
        ans: Literal["yes", "no"] = Field(description="判断内容是否能回答问题或与问题高度相关")
        reason: str = Field(description="简要解释判断理由（可选，用于调试）")

    node_ids = state["node_ids"]
    if not node_ids:
        return Command(goto=END)

    query = state["query"]
    # node_list 现在是从 store 拿到的字典列表，包含 content, path, summary 等
    node_list = node_content_store.mget(node_ids)

    # 构造批量请求
    msg_list = []
    for node in node_list:
        if not node: continue

        # 核心优化：提供多维度的上下文
        system_prompt = f"""你是一个专业的技术文档审查专家。
        你的任务是判断下方提供的【文档片段】是否包含足够的信息来回答、或者与用户的【问题】直接相关。
        
        【评分标准】：
        1. 能够直接回答用户的问题或提供操作步骤 -> yes
        2. 虽然不能完全回答，但提供了关键背景、定义或相关的配置参数 -> yes
        3. 内容仅提及关键词但属于无关主题，或者内容完全为空 -> no
        4. 内容是无法理解的代码片段或目录列表且无解释说明 -> no
        
        【用户的原始问题】：
        {query}
        """

        user_content = f"""
        ### 待评估文档信息：
        - 章节路径: {node.get('path', '未知')}
        - 核心内容: 
        ---
        {node.get('content', '')}
        ---
        """

        msg_list.append([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ])

    # 批量调用模型
    response_list = grade_model.with_structured_output(schema=output).batch(msg_list)

    final_nodes = []
    for i, response in enumerate(response_list):
        print(response.ans)
        print(response.reason)

        if response.ans == "yes":
            # 返回完整的节点对象，方便后续 Agent 引用 path 和 content
            final_nodes.append(node_list[i])

    return {"final_nodes": final_nodes}
