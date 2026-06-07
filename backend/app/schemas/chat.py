"""聊天相关 API Schema"""

from datetime import datetime
import uuid
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: uuid.UUID | None = None
    query: str


class ChatSessionCreate(BaseModel):
    title: str = "新对话"


class ChatSessionSummary(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    metadata: dict = {}
    created_at: datetime | None = None

    model_config = {"from_attributes": True}
