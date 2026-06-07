"""文档检索工具 — Agent 用这些工具自主探索文档

- list_documents: 列出所有已索引文档
- get_doc_tree: 获取文档的章节树（无正文）
- get_node_content: 获取节点的完整正文
"""

from langchain_core.tools import tool

from app.db.session import get_db_session
from app.db.queries import get_global_index, get_doc_trees, get_nodes


@tool
async def list_documents() -> list[dict]:
    """列出所有已索引的文档，返回每个文档的 ID、名称、摘要和关键词。
    在开始检索之前，先用此工具了解有哪些文档可用。
    """
    async with get_db_session() as db:
        docs = await get_global_index(db)
    return docs


@tool
async def get_doc_tree(doc_id: str) -> dict:
    """获取指定文档的完整目录树，包含所有章节标题但不含正文。
    用它来快速浏览文档结构，定位相关章节。

    Args:
        doc_id: 文档 ID（从 list_documents 获取）
    """
    async with get_db_session() as db:
        trees = await get_doc_trees(db, [doc_id])
    if not trees:
        return {"error": f"未找到文档 {doc_id}"}
    return trees[0]


@tool
async def get_node_content(node_ids: list[str]) -> list[dict]:
    """获取指定节点的完整正文内容。每个节点包含标题、路径、正文文本。
    批量传入多个 node_id 以减少调用次数。

    Args:
        node_ids: 节点 ID 列表（从 get_doc_tree 返回的树结构中获取）
    """
    async with get_db_session() as db:
        nodes = await get_nodes(db, node_ids)
    return nodes
