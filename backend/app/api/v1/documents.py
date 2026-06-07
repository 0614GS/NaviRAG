"""文档管理 API — 上传、列表、详情、删除、重索引"""

import uuid
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy import select

from app.db.session import get_db_session
from app.db.models import Node
from app.db.queries import (
    list_documents,
    get_document,
    delete_document,
    update_document_status,
)
from app.schemas.document import DocumentSummary, DocumentDetail, DocumentListResponse

router = APIRouter()


async def _run_indexing(doc_id: str, filename: str):
    """后台执行文档索引"""

    async with get_db_session() as db:
        try:
            await update_document_status(db, doc_id, "indexing")
            await db.commit()
        except Exception:
            pass

    try:
        # 从 DB 读取原始内容
        async with get_db_session() as db:
            doc = await get_document(db, doc_id)
            if not doc or not doc.raw_content:
                raise ValueError("文档原始内容不存在")
            content = doc.raw_content

        from app.indexing.pipeline import index_document
        await index_document(doc_id, content, filename)

        async with get_db_session() as db:
            await update_document_status(db, doc_id, "indexed")
            await db.commit()

    except Exception as e:
        error_msg = str(e)
        print(f"[Indexing] 索引失败 for {doc_id}: {error_msg}")
        async with get_db_session() as db:
            await update_document_status(db, doc_id, "failed", error_msg)
            await db.commit()


@router.post("/documents/upload", response_model=DocumentSummary)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """上传 Markdown 文档并触发后台索引"""
    if not file.filename or not file.filename.endswith(('.md', '.markdown', '.txt')):
        raise HTTPException(status_code=400, detail="仅支持 .md / .markdown / .txt 文件")

    # 1. 读取文件内容
    raw_content = (await file.read()).decode("utf-8")
    doc_id = str(uuid.uuid4())[:8]
    filename = file.filename
    doc_name = filename.rsplit(".", 1)[0] if "." in filename else filename

    # 2. 存入数据库（原始内容 + 元数据）
    from app.db.models import Document
    async with get_db_session() as db:
        doc = Document(
            id=doc_id,
            name=doc_name,
            filename=filename,
            raw_content=raw_content,
            status="pending",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

    # 3. 触发后台索引
    background_tasks.add_task(_run_indexing, doc_id, filename)

    return DocumentSummary(
        doc_id=doc.id,
        doc_name=doc.name,
        summary=None,
        keywords=[],
        status=doc.status,
        created_at=doc.created_at,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents_api(
    page: int = 1,
    size: int = 20,
    status: Optional[str] = None,
):
    """获取文档列表（分页）"""
    async with get_db_session() as db:
        docs, total = await list_documents(db, page, size, status)

    return DocumentListResponse(
        documents=[
            DocumentSummary(
                doc_id=d.id,
                doc_name=d.name,
                summary=d.summary,
                keywords=d.keywords or [],
                status=d.status,
                created_at=d.created_at,
            )
            for d in docs
        ],
        total=total,
        page=page,
        size=size,
    )


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
async def get_document_api(doc_id: str):
    """获取文档详情（含导航树结构）"""
    async with get_db_session() as db:
        doc = await get_document(db, doc_id)

    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    return DocumentDetail(
        doc_id=doc.id,
        doc_name=doc.name,
        filename=doc.filename,
        summary=doc.summary,
        keywords=doc.keywords or [],
        tree_structure=doc.tree_structure,
        status=doc.status,
        error_msg=doc.error_msg,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.delete("/documents/{doc_id}")
async def delete_document_api(doc_id: str):
    """删除文档及其所有节点（级联删除）"""
    async with get_db_session() as db:
        doc = await get_document(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        await delete_document(db, doc_id)
        await db.commit()

    return {"detail": "文档已删除"}


@router.post("/documents/{doc_id}/reindex", response_model=DocumentSummary)
async def reindex_document(
    doc_id: str,
    background_tasks: BackgroundTasks = None,
):
    """重新索引文档"""
    async with get_db_session() as db:
        doc = await get_document(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        if not doc.raw_content:
            raise HTTPException(status_code=400, detail="文档原始内容不存在，无法重索引")

        # 删除旧节点（保留文档记录，后面重建）
        await delete_document(db, doc_id)
        # 重建文档记录（保留原始内容）
        from app.db.models import Document as DocModel
        new_doc = DocModel(
            id=doc_id,
            name=doc.name,
            filename=doc.filename,
            raw_content=doc.raw_content,
            status="pending",
        )
        db.add(new_doc)
        await db.commit()
        await db.refresh(new_doc)

        # 触发后台索引
        background_tasks.add_task(_run_indexing, doc_id, doc.filename)

    return DocumentSummary(
        doc_id=new_doc.id,
        doc_name=new_doc.name,
        summary=None,
        keywords=[],
        status=new_doc.status,
        created_at=new_doc.created_at,
    )


@router.get("/nodes/{node_id}")
async def get_node_api(node_id: str):
    """获取单个节点的完整内容（用于文档树点击查看原文和引用溯源）"""
    async with get_db_session() as db:
        result = await db.execute(select(Node).where(Node.id == node_id))
        node = result.scalar_one_or_none()

    if not node:
        raise HTTPException(status_code=404, detail="节点不存在")

    return {
        "node_id": node.id,
        "doc_id": node.doc_id,
        "title": node.title,
        "path": node.path,
        "content": node.content,
        "summary": node.summary,
        "keywords": node.keywords,
        "level": node.level,
    }
