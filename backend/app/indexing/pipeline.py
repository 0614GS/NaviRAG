"""索引编排管道 — 整合解析、元数据生成、DB 写入

作为 FastAPI BackgroundTask 调用，将 Markdown 文档完整索引到数据库
"""

from pathlib import Path

from app.db.session import get_db_session
from app.db.models import Document
from app.indexing.md_parser import parse_document
from app.indexing.metadata import process_tree_recursive, generate_doc_overview


async def index_document(doc_id: str, content: str, filename: str) -> str:
    """
    对文档内容执行完整索引流程。

    Args:
        doc_id: 已创建的文档 ID（由上层传入）
        content: Markdown 文件内容
        filename: 原始文件名

    Returns:
        doc_id: 索引完成的文档 ID
    """
    doc_name = Path(filename).stem

    print(f"\n{'='*60}")
    print(f"[Pipeline] 开始索引文档: {filename} (ID: {doc_id})")
    print(f"{'='*60}")

    # 1. 解析 Markdown → 树结构
    tree_structure = parse_document(content)
    print(f"[Pipeline] 解析完成，共 {len(tree_structure)} 个顶层节点")

    # 2. 递归处理 → 生成元数据（LLM 调用，无 DB）
    processed_tree, flat_nodes = await process_tree_recursive(
        nodes=tree_structure,
        parent_path=doc_name,
        doc_id=doc_id,
    )
    print(f"[Pipeline] 元数据生成完成，共 {len(flat_nodes)} 个节点")

    # 3. 生成文档总览（LLM 调用，无 DB）
    doc_overview = await generate_doc_overview(doc_name, processed_tree)

    # 4. 更新数据库记录并写入节点（单会话，顺序写入）
    async with get_db_session() as db:
        try:
            from sqlalchemy import select
            from app.db.models import Document, Node

            # 查询已有记录并更新
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one()
            doc.raw_content = content
            doc.summary = doc_overview["summary"]
            doc.keywords = doc_overview["keywords"]
            doc.tree_structure = {"nodes": processed_tree}
            doc.status = "indexed"

            # 批量写入所有节点
            for nd in flat_nodes:
                node = Node(
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
                db.add(node)

            await db.commit()
            print(f"[Pipeline] 文档 {doc_name} 索引完成 (ID: {doc_id}), 共 {len(flat_nodes)} 个节点")

        except Exception as e:
            await db.rollback()
            print(f"[Pipeline] 索引失败: {e}")
            raise

    return doc_id
