"""数据库查询函数 — 替代原 data/storage.py 的文件系统 KV 存储"""

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Document, Node


async def get_doc_trees(db: AsyncSession, doc_ids: List[str]) -> List[dict]:
    """获取文档的导航树结构（替代 doc_tree_store.mget）"""
    result = await db.execute(
        select(Document).where(Document.id.in_(doc_ids))
    )
    docs = result.scalars().all()
    return [
        {
            "doc_id": doc.id,
            "doc_name": doc.name,
            "summary": doc.summary,
            "keywords": doc.keywords,
            "tree_structure": doc.tree_structure,
        }
        for doc in docs
    ]


async def get_nodes(db: AsyncSession, node_ids: List[str]) -> List[dict]:
    """获取节点完整内容（替代 node_content_store.mget）"""
    result = await db.execute(
        select(Node).where(Node.id.in_(node_ids))
    )
    nodes = result.scalars().all()
    return [
        {
            "node_id": node.id,
            "doc_id": node.doc_id,
            "title": node.title,
            "path": node.path,
            "content": node.content,
            "summary": node.summary,
            "keywords": node.keywords,
            "level": node.level,
            "parent_node_id": node.parent_node_id,
        }
        for node in nodes
    ]


async def get_global_index(db: AsyncSession) -> List[dict]:
    """获取全局文档索引（替代 prompts.py 中的硬编码列表）"""
    result = await db.execute(
        select(Document)
        .where(Document.status == "indexed")
        .order_by(Document.name)
    )
    docs = result.scalars().all()
    return [
        {
            "doc_id": doc.id,
            "doc_name": doc.name,
            "keywords": doc.keywords,
            "summary": doc.summary,
        }
        for doc in docs
    ]


async def save_doc_tree(db: AsyncSession, doc_id: str, doc_data: dict) -> Document:
    """保存文档及其导航树"""
    doc = Document(
        id=doc_id,
        name=doc_data.get("doc_name", ""),
        filename=doc_data.get("filename", ""),
        summary=doc_data.get("summary"),
        keywords=doc_data.get("keywords", []),
        tree_structure=doc_data.get("structure"),
        doc_metadata={"structure": doc_data.get("structure")},
        status="indexed",
    )
    db.add(doc)
    await db.flush()
    return doc


async def save_node(db: AsyncSession, node_data: dict) -> Node:
    """保存单个节点"""
    node = Node(
        id=node_data["node_id"],
        doc_id=node_data["doc_id"],
        title=node_data["title"],
        path=node_data["path"],
        content=node_data.get("content", ""),
        summary=node_data.get("summary"),
        keywords=node_data.get("keywords", []),
        level=node_data.get("level", 0),
        parent_node_id=node_data.get("parent_node_id"),
        sort_order=node_data.get("sort_order", 0),
    )
    db.add(node)
    await db.flush()
    return node


async def save_nodes_bulk(db: AsyncSession, nodes_data: List[dict]) -> List[Node]:
    """批量保存节点"""
    nodes = [
        Node(
            id=nd["node_id"],
            doc_id=nd["doc_id"],
            title=nd["title"],
            path=nd["path"],
            content=nd.get("content", ""),
            summary=nd.get("summary"),
            keywords=nd.get("keywords", []),
            level=nd.get("level", 0),
            parent_node_id=nd.get("parent_node_id"),
            sort_order=nd.get("sort_order", 0),
        )
        for nd in nodes_data
    ]
    db.add_all(nodes)
    await db.flush()
    return nodes


async def delete_document(db: AsyncSession, doc_id: str) -> bool:
    """删除文档及其所有节点（级联删除）"""
    result = await db.execute(
        select(Document).where(Document.id == doc_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        return False
    await db.delete(doc)
    await db.flush()
    return True


async def list_documents(
    db: AsyncSession,
    page: int = 1,
    size: int = 20,
    status: Optional[str] = None,
) -> tuple[List[Document], int]:
    """分页获取文档列表"""
    query = select(Document)
    count_query = select(Document)

    if status:
        query = query.where(Document.status == status)
        count_query = count_query.where(Document.status == status)

    # 总数
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    # 分页
    query = query.order_by(Document.created_at.desc()).offset((page - 1) * size).limit(size)
    result = await db.execute(query)
    docs = result.scalars().all()

    return list(docs), total


async def get_document(db: AsyncSession, doc_id: str) -> Optional[Document]:
    """获取单个文档"""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def update_document_status(
    db: AsyncSession, doc_id: str, status: str, error_msg: Optional[str] = None
) -> Optional[Document]:
    """更新文档索引状态"""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        return None
    doc.status = status
    if error_msg:
        doc.error_msg = error_msg
    await db.flush()
    return doc
