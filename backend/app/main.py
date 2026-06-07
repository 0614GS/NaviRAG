"""NaviRAG FastAPI 应用入口"""

import os
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db, close_db
from app.api.v1 import health, documents, chat

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", settings.LLM_API_KEY)
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", settings.LLM_BASE_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：初始化数据库
    await init_db()
    yield
    # 关闭时：清理连接
    await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
