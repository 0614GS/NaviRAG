import asyncio
import uuid
import glob
import json
import os
import re
import time
from typing import List, Dict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from data.storage import doc_tree_store
from data.storage import node_content_store

# 加载环境变量
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("SI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("SI_BASE_URL")
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENROUTER_API_KEY")
# os.environ['OPENAI_BASE_URL'] = os.getenv("OPENROUTER_API_KEY")

# 关键词和摘要的提取模型
extract_model = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.2", temperature=0)


# extract_model = ChatOpenAI(model="Qwen/Qwen3-Omni-30B-A3B-Instruct", temperature=0)


# extract_model = ChatOpenAI(model="gpt-oss-120b", temperature=0)

# 获取扁平化的node列表
def extract_nodes_from_markdown(markdown_content: str) -> List[Dict]:
    """
    解析 Markdown 内容，将每个标题及其下方的文本提取为一个节点列表。

    Args:
        markdown_content: Markdown 文件内容字符串

    Returns:
        包含 'level', 'title', 'text' 的扁平节点列表
    """
    # 匹配 Markdown 标题 (例如: ## Title)
    header_pattern = r'^(#{1,6})\s+(.+)$'
    # 匹配代码块标记，用于避免在代码块内部匹配标题
    code_block_pattern = r'^```'

    lines = markdown_content.split('\n')
    node_list = []
    current_node = None
    in_code_block = False

    # 虚拟根节点，用于捕获文件开头没有标题的内容（如果有）
    # 但通常 markdown 都是从 # Title 开始

    for line in lines:
        stripped_line = line.strip()

        # 1. 状态检查：是否在代码块中
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block

        # 2. 检查是否是标题行 (且不在代码块中)
        header_match = re.match(header_pattern, stripped_line)
        if header_match and not in_code_block:
            # 如果之前有正在处理的节点，先保存其文本内容
            if current_node:
                current_node['text'] = '\n'.join(current_node['text_lines']).strip()
                del current_node['text_lines']  # 移除临时列表
                node_list.append(current_node)
                # print(current_node)

            # 创建新节点
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_node = {
                'level': level,
                'title': title,
                'text_lines': []
            }
        else:
            # 3. 如果是普通行，归属于当前节点
            if current_node:
                current_node['text_lines'].append(line)

    # 处理最后一个节点
    if current_node:
        current_node['text'] = '\n'.join(current_node['text_lines']).strip()
        del current_node['text_lines']
        node_list.append(current_node)

    return node_list


# 从扁平化列表中构建带有text的树
def build_tree_from_flat_nodes(node_list: List[Dict]) -> List[Dict]:
    """
    使用栈算法将扁平的节点列表转换为嵌套的树结构。
    逻辑源自 page_index_md.py 的 build_tree_from_nodes。
    """
    if not node_list:
        return []

    stack = []  # 用于追踪父节点路径 [(node, level), ...]
    root_nodes = []  # 最终的树根列表

    for node in node_list:
        current_level = node['level']

        # 基础树节点结构
        tree_node = {
            'title': node['title'],
            'text': node['text'],  # 暂时保留 text 用于给 LLM 分析
            'nodes': []  # 子节点列表
        }

        # 栈逻辑：如果栈顶节点的层级 >= 当前层级，说明栈顶节点不是当前节点的父级
        # 需要弹出，直到找到一个层级比当前小的节点（即父节点）
        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            # 如果栈空了，说明当前节点是顶层节点（Root）
            root_nodes.append(tree_node)
        else:
            # 栈顶元素即为父节点，将当前节点加入其 nodes 列表
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)

        # 将当前节点压入栈，作为潜在的下一级父节点
        stack.append((tree_node, current_level))

    return root_nodes


# 使用大模型提取关键词与摘要
async def generate_metadata_with_llm(title: str, path: str, content: str, children_summary: str = "") -> Dict:
    """
    调用大模型生成 Summary 和 Keywords。
    为了稳定性，使用了简单的重试机制。
    """

    class outputSchema(BaseModel):
        keywords: List[str] = Field(description="一个字符串列表，包含5-10个关键技术名词（API名称、特定概念等），in English")
        summary: str = Field(description="50字以内的极简摘要，需涵盖子章节的核心主题，in English")

    system_prompt = """
    你是一个专业的技术文档分析助手。请根据提供的文档节点信息提取元数据。
    请返回指定格式，包含以下字段：
    1. "summary": 50字以内的内容极简摘要。如果是父节点，需涵盖子节点的核心主题，in English。
    2. "keywords": 一个字符串列表，包含5-10个关键技术名词（API名称、特定概念等），in English。
    """

    user_prompt = f"""
    文档路径: {path}
    章节标题: {title}
    本章节内容:
    {content if content else "（无直接正文）"}
    
    子章节摘要内容：
    {children_summary if children_summary else "（无子章节）"}
    """

    print("正在提取title:", title, "的关键词与总结")
    # print("子章节内容：", children_summary, "\n")
    print("*" * 100)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await extract_model.with_structured_output(schema=outputSchema).ainvoke([
                SystemMessage(system_prompt),
                HumanMessage(user_prompt)
            ])
            return response.model_dump()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error generating metadata for {path}: {e}")
                return {"summary": "生成失败", "keywords": []}
            time.sleep(5)  # 等待后重试


# 全局并发限制
sem = asyncio.Semaphore(15)


# 树的递归处理与 ID 生成
async def process_tree_recursive(nodes: List[Dict], parent_path: str) -> List[Dict]:
    """
    异步递归遍历树。
    特性：
    1. 同级节点并发执行 (Parallel Siblings)。
    2. 自底向上汇总 (Bottom-Up): 父节点会等待所有子节点完成后，拿到汇总信息才开始生成自己的元数据。
    """

    # 内部函数, 处理单个节点的逻辑
    async def _process_single_node(node: Dict) -> Dict:
        # 1. 构建基础信息
        current_title = node['title']
        current_path = f"{parent_path} > {current_title}" if parent_path else current_title
        content = node.get('text', '')

        # 2. 【递归关键】：先处理子节点 (递归调用主函数)
        # 这里会等待该节点下的所有子节点并发处理完成
        children = []
        children_info_for_parent = ""

        if node.get('nodes'):
            # await 这一层，意味着当前节点会挂起，直到它的子树全部构建完毕
            children = await process_tree_recursive(node['nodes'], current_path)

            # 汇总子节点信息给 LLM 参考
            summary_list = [
                (f"- {child['title']}: {child['summary']} "
                 f"(Keywords: {', '.join(child['keywords'])})") for child in children
            ]
            children_info_for_parent = "\n".join(summary_list)

        # 3. 生成当前节点的元数据 (LLM)
        # 使用 Semaphore 控制并发，避免 Rate Limit
        async with sem:
            metadata = await generate_metadata_with_llm(
                title=current_title,
                path=current_path,
                content=content,
                children_summary=children_info_for_parent
            )
        node_id = str(uuid.uuid4())[:8]
        # 4. 构建完整存储对象 (Metadata + Content) 并存入 DB
        full_content_obj = {
            "node_id": node_id,
            "title": current_title,
            "path": current_path,
            "content": content,
            "summary": metadata.get("summary", ""),
            "keywords": metadata.get("keywords", [])
        }

        # 使用 asyncio.to_thread 在独立的线程中执行同步写入，避免阻塞事件循环
        await asyncio.to_thread(
            node_content_store.mset, [(node_id, full_content_obj)]
        )
        print(f"  [Storage] Saved node {node_id}: {current_title}")

        # 5. 返回给上一层的轻量级节点结构
        return {
            "node_id": node_id,
            "path": current_path,
            "title": current_title,
            "keywords": metadata.get("keywords", []),
            "summary": metadata.get("summary", ""),
            "nodes": children
        }

    # --- 主逻辑：并发调度 ---

    # 1. 为当前层级的每一个节点创建一个 Task
    tasks = [_process_single_node(node) for node in nodes]

    # 2. 并发执行所有 Task，并按顺序收集结果
    # 这里实现了同级并发：List 中的 Node 1, Node 2... 会同时开始跑
    results = await asyncio.gather(*tasks)

    return list(results)


# 主流程
async def analyze_markdown_file(file_path: str):
    """
    主函数：读取文件 -> 解析 -> 处理 -> 保存
    """
    print(f"正在处理文件: {file_path}")

    # 读取 Markdown 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取扁平节点
    flat_nodes = extract_nodes_from_markdown(content)
    print("扁平节点", flat_nodes)

    # 构建树状结构
    tree_structure = build_tree_from_flat_nodes(flat_nodes)
    print("树形结构", tree_structure)

    # 递归增强节点信息 (ID, Path, Summary, Keywords)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    processed_tree = await process_tree_recursive(
        nodes=tree_structure,
        parent_path=file_name  # 将文件名作为 Path 的第一级
    )

    # 获取文件的metadata
    doc_overview = await generate_doc_global_summary(file_name, processed_tree)

    # 为整个文档生成 doc_id 并存入 doc_tree_store
    doc_id = str(uuid.uuid4())[:8]
    doc_data = {
        "doc_id": doc_id,
        "doc_name": file_name,
        "summary": doc_overview["summary"],
        "keywords": doc_overview["keywords"],
        "structure": processed_tree
    }

    await asyncio.to_thread(doc_tree_store.mset, [(doc_id, doc_data)])

    print(f"文档 {file_name} 处理完成。DocID: {doc_id}, 导航树已存入 doc_tree_store")

    return doc_data


async def generate_doc_global_summary(doc_name: str, level1_nodes: List[Dict]) -> Dict:
    """基于所有一级标题的信息，生成整篇文档的总览摘要和关键词"""

    class outputSchema(BaseModel):
        keywords: List[str] = Field(description="一个字符串列表，包含5-10个关键技术名词（API名称、特定概念等）")
        summary: str = Field(description="50字以内的中文内容极简摘要。如果是父节点，需涵盖子节点的核心主题")

    # 汇总一级标题的信息作为上下文
    context_list = [f"标题: {n['title']}\n摘要: {n['summary']}\n关键词: {', '.join(n['keywords'])}" for n in
                    level1_nodes]
    context_text = "\n\n".join(context_list)

    system_prompt = "你是一个文档索引专家。请根据文档各章节的摘要，为整篇文档生成一份总览元数据。"
    user_prompt = f"""
    文档名称: {doc_name}
    各章节核心内容汇总:
    {context_text}

    你是一个专业的技术文档分析助手。请根据提供的文档节点信息提取元数据。
    请返回指定格式，包含以下字段：
    1. "summary": 50字以内的中文内容极简摘要。如果是父节点，需涵盖子节点的核心主题，in English。
    2. "keywords": 一个字符串列表，包含5-10个关键技术名词（API名称、特定概念等），in English。
    """

    print("各章节汇总信息\n", context_text, "\n\n")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await extract_model.with_structured_output(schema=outputSchema).ainvoke([
                SystemMessage(system_prompt),
                HumanMessage(user_prompt)
            ])
            return response.model_dump()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error generating metadata for {doc_name}: {e}")
                return {"summary": "生成失败", "keywords": []}
            time.sleep(1)  # 等待后重试


# 批量处理主入口
async def batch_process_markdowns(input_dir: str, output_dir: str):
    """批量处理入口"""
    md_files = glob.glob(os.path.join(input_dir, "*.md"))
    if not md_files:
        print("未找到任何 .md 文件")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 用于存储所有文档的元数据，最后生成全局索引
    global_index_list = []

    tasks = [analyze_markdown_file(md_file) for md_file in md_files]

    results = await asyncio.gather(*tasks)

    for result in results:
        with open(os.path.join(output_dir, result["doc_id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 收集元数据用于全局索引
        global_index_list.append({
            "doc_id": result["doc_id"],
            "doc_name": result["doc_name"],
            "keywords": result["keywords"],
            "summary": result["summary"]
        })

        print(f"收集文档: {result['doc_name']} (ID: {result['doc_id']})")

    # 生成顶层目录索引 json
    global_index_path = os.path.join(output_dir, "global_index.json")
    with open(global_index_path, "w", encoding="utf-8") as f:
        json.dump(global_index_list, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    INPUT_DIR = "../data/input/langgraph"
    OUTPUT_DIR = "../data/output"

    asyncio.run(batch_process_markdowns(INPUT_DIR, OUTPUT_DIR))
