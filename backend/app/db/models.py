"""SQLAlchemy ORM 模型定义"""

import uuid
from datetime import datetime

from sqlalchemy import String, Text, Integer, Float, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(8), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, comment="文档标识名称")
    filename: Mapped[str] = mapped_column(String(255), nullable=False, comment="原始文件名")
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True, comment="原始 Markdown 全文")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSONB, default=list)
    tree_structure: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    doc_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending", comment="pending|indexing|indexed|failed")
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    nodes: Mapped[list["Node"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(String(8), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(8), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSONB, default=list)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_node_id: Mapped[str | None] = mapped_column(String(8), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="nodes")


class SourceReference(Base):
    """LLM 回答与源文档段落的引用映射"""
    __tablename__ = "source_references"

    id: Mapped[str] = mapped_column(String(8), primary_key=True)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False)
    node_id: Mapped[str] = mapped_column(String(8), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    doc_id: Mapped[str] = mapped_column(String(8), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    cited_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255), default="新对话")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan",
                                                         order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"),
                                                  nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, comment="user | assistant")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta_info: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
