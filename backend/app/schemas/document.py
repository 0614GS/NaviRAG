"""文档相关 API Schema"""

from datetime import datetime
from pydantic import BaseModel


class DocumentSummary(BaseModel):
    doc_id: str
    doc_name: str
    summary: str | None = None
    keywords: list[str] = []
    status: str = "pending"
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentDetail(BaseModel):
    doc_id: str
    doc_name: str
    filename: str
    summary: str | None = None
    keywords: list[str] = []
    tree_structure: dict | None = None
    status: str = "pending"
    error_msg: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int
    page: int
    size: int
