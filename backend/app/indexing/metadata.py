"""LLM 元数据生成 — 摘要提取与关键词生成

从原 data/md2tree.py 提取的 LLM 调用逻辑，适配异步环境
"""

import asyncio
import json
import time
import uuid
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.models import extract_model

# 全局并发控制（控制 LLM 并发调用数量）
_sem = asyncio.Semaphore(15)


class NodeMetadata(BaseModel):
    keywords: List[str] = Field(description="5-10个关键技术名词，in English")
    summary: str = Field(description="50字以内的极简摘要，in English")


class DocOverview(BaseModel):
    keywords: List[str] = Field(description="5-10个关键技术名词")
    summary: str = Field(description="50字以内的中文内容极简摘要")


async def generate_metadata_with_llm(
    title: str,
    path: str,
    content: str,
    children_summary: str = "",
    model: ChatOpenAI | None = None,
) -> Dict:
    """调用 LLM 生成 Summary 和 Keywords（使用 JSON mode）"""
    if model is None:
        model = extract_model

    system_prompt = """
    你是一个专业的技术文档分析助手。请根据提供的文档节点信息提取元数据。
    你必须以 JSON 格式返回结果，格式如下：
    {"summary": "50字以内的内容极简摘要（英文）", "keywords": ["关键词1", "关键词2", ...]}
    如果是父节点，summary 需涵盖子节点的核心主题。keywords 应包含5-10个关键技术名词。
    """

    user_prompt = f"""
    文档路径: {path}
    章节标题: {title}
    本章节内容:
    {content if content else "（无直接正文）"}

    子章节摘要内容：
    {children_summary if children_summary else "（无子章节）"}
    """

    print(f"正在提取title: {title} 的关键词与总结")
    print("*" * 100)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await model.ainvoke([
                SystemMessage(system_prompt),
                HumanMessage(user_prompt)
            ])
            result = json.loads(response.content)
            # 验证必要字段
            return {
                "summary": str(result.get("summary", "生成失败")),
                "keywords": list(result.get("keywords", [])),
            }
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error generating metadata for {path}: {e}")
                return {"summary": "生成失败", "keywords": []}
            time.sleep(5)


async def process_tree_recursive(
    nodes: List[Dict],
    parent_path: str,
    doc_id: str,
    level_offset: int = 1,
) -> tuple[List[Dict], List[Dict]]:
    """
    异步递归遍历树，自底向上生成元数据。

    返回 (tree_nodes, flat_nodes)：
    - tree_nodes: 嵌套树结构（供 global_index 和 Document 使用）
    - flat_nodes: 扁平节点列表（供批量 DB 写入）

    特性：
    1. 同级节点并发执行（仅 LLM 调用并发，不涉及 DB 写入）
    2. 父节点等待所有子节点完成后才开始生成自己的元数据
    3. 节点数据收集后由调用方统一批量写入 DB
    """

    async def _process_single_node(node: Dict, level: int) -> tuple[Dict, List[Dict]]:
        current_title = node['title']
        current_path = f"{parent_path} > {current_title}" if parent_path else current_title
        content = node.get('text', '')

        # 先处理子节点
        children = []
        children_info_for_parent = ""
        flat_children = []

        if node.get('nodes'):
            child_trees, child_flats = await process_tree_recursive(
                node['nodes'],
                current_path,
                doc_id,
                level_offset=level + 1,
            )
            children = child_trees
            flat_children = child_flats

            summary_list = [
                (f"- {child['title']}: {child['summary']} "
                 f"(Keywords: {', '.join(child['keywords'])})") for child in children
            ]
            children_info_for_parent = "\n".join(summary_list)

        # 生成当前节点的元数据
        async with _sem:
            metadata = await generate_metadata_with_llm(
                title=current_title,
                path=current_path,
                content=content,
                children_summary=children_info_for_parent,
            )

        node_id = str(uuid.uuid4())[:8]

        # 构建扁平节点对象（供批量 DB 写入）
        flat_node = {
            "node_id": node_id,
            "doc_id": doc_id,
            "title": current_title,
            "path": current_path,
            "content": content,
            "summary": metadata.get("summary", ""),
            "keywords": metadata.get("keywords", []),
            "level": level,
            "sort_order": 0,
            "parent_node_id": None,
        }

        print(f"  [Collect] Node {node_id}: {current_title}")

        # 构建树节点（不含完整 content，供 global_index）
        tree_node = {
            "node_id": node_id,
            "path": current_path,
            "title": current_title,
            "keywords": metadata.get("keywords", []),
            "summary": metadata.get("summary", ""),
            "nodes": children,
        }

        # 返回 (tree_node, flat_nodes_including_self_and_children)
        return tree_node, [flat_node] + flat_children

    # 并发处理当前层级的所有节点
    tasks = [_process_single_node(node, level_offset) for node in nodes]
    results = await asyncio.gather(*tasks)

    trees = [r[0] for r in results]
    flats = []
    for r in results:
        flats.extend(r[1])

    return trees, flats


async def generate_doc_overview(doc_name: str, top_level_nodes: List[Dict]) -> Dict:
    """基于顶层节点信息，生成整篇文档的总览摘要和关键词（使用 JSON mode）"""
    context_list = [
        f"标题: {n['title']}\n摘要: {n['summary']}\n关键词: {', '.join(n['keywords'])}"
        for n in top_level_nodes
    ]
    context_text = "\n\n".join(context_list)

    system_prompt = """你是一个文档索引专家。请根据文档各章节的摘要，为整篇文档生成一份总览元数据。
    你必须以 JSON 格式返回结果，格式如下：
    {"summary": "50字以内的内容极简摘要（英文）", "keywords": ["关键词1", "关键词2", ...]}
    summary 应涵盖各章节的核心主题，keywords 应包含5-10个关键技术名词。"""

    user_prompt = f"""
    文档名称: {doc_name}
    各章节核心内容汇总:
    {context_text}
    """

    print(f"各章节汇总信息\n{context_text}\n\n")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await extract_model.ainvoke([
                SystemMessage(system_prompt),
                HumanMessage(user_prompt)
            ])
            result = json.loads(response.content)
            return {
                "summary": str(result.get("summary", "生成失败")),
                "keywords": list(result.get("keywords", [])),
            }
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error generating metadata for {doc_name}: {e}")
                return {"summary": "生成失败", "keywords": []}
            time.sleep(1)
