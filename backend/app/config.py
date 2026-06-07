"""应用配置管理"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL 连接（组件模式，兼容 Docker 和本地开发）
    POSTGRES_SERVER: str = ""
    POSTGRES_PORT: str = ""
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""

    # 兼容旧配置：直接指定完整 URL
    DATABASE_URL: str = ""

    # LLM API
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""

    # 应用
    APP_NAME: str = "NaviRAG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }

    @property
    def database_url(self) -> str:
        """优先用完整 DATABASE_URL，否则从组件拼接"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        host = self.POSTGRES_SERVER or "localhost"
        port = self.POSTGRES_PORT or "5432"
        user = self.POSTGRES_USER or "navirag"
        password = self.POSTGRES_PASSWORD or "navirag"
        db = self.POSTGRES_DB or "navirag"
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


settings = Settings()
