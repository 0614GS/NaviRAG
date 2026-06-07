"""通用 API Schema"""

from pydantic import BaseModel


class PaginationParams(BaseModel):
    page: int = 1
    size: int = 20


class ErrorResponse(BaseModel):
    detail: str
