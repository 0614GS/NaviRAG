# NaviRAG — 基于树状结构的智能文档检索系统

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)

NaviRAG 是一个**层级推理检索增强生成（RAG）系统**，用文档树结构替代传统的向量嵌入检索。系统解析 Markdown 文档的 H1–H6 标题层级，由 LLM 自底向上生成摘要和关键词，构建导航树；在对话时，AI Agent 自主调用工具沿树探索，精准定位并阅读原文内容，最终给出有据可循的回答。

---

## 1. 功能特性

### 文档管理
- **Markdown 上传与索引**：上传 `.md`/`.markdown`/`.txt` 文件，后台自动解析 H1–H6 标题树，LLM 生成逐层摘要和关键词，存入 PostgreSQL
- **文档列表与状态跟踪**：实时查看索引状态（等待中 / 索引中 / 已索引 / 失败），支持删除和重新索引
- **层级树浏览**：点击文档展开章节树，再点击任意节点即可查看原文内容

### 智能对话
- **Agentic 自主检索**：AI Agent 自主决定检索策略 — 先列出所有文档，再浏览相关文档的目录树，最后读取具体章节内容
- **流式 SSE 输出**：实时展示 LLM token 生成过程，检索工具调用内联显示在回答中
- **溯源引用**：回答末尾列出引用来源 `[1]` `[2]` …，点击即可查看原始文档内容
- **多轮对话**：会话历史自动持久化，切换对话不丢失上下文

### 技术栈
| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI (Python 3.11+) |
| AI 框架 | LangChain (ReAct Agent + 工具调用) |
| ORM | SQLAlchemy 2.0 (异步 PostgreSQL) |
| 数据库 | PostgreSQL 16 |
| 前端 | React 19 + TypeScript + TailwindCSS 3 |
| 部署 | Docker Compose (Nginx 反向代理 + SSE 支持) |

---

## 2. 核心原理

### 2.1 索引管道

```
Markdown 文件 → md_parser (正则 H1–H6 解析，代码块过滤)
              → 栈算法构建嵌套树 {title, content, children}
              → process_tree_recursive (自底向上 LLM 元数据生成)
              → 写入 PostgreSQL (documents + nodes)
```

**自底向上合成**：叶子节点先生成摘要和关键词，父节点汇总所有子节点信息后再生成自己的元数据。同级节点 LLM 调用并发执行（Semaphore(15) 控制并发）。

**存储解耦**：
- `documents` 表：存原始全文（`raw_content`）、文档级摘要、导航树结构（`tree_structure` JSONB，含每个节点的 `node_id`、`title`、`path`，不含正文）
- `nodes` 表：存每个节点的完整正文（`content`）、摘要、关键词
- 导航树与正文分离，LLM 浏览目录时 Token 消耗极低

### 2.2 Agentic 检索

```
用户提问 → Agent 接收查询 + 对话历史
         → Agent 自主规划：list_documents → get_doc_tree → get_node_content
         → 流式返回 tokens + 工具调用状态 + 溯源引用
         → 保存回答到 chat_messages + source_references
```

**三个检索工具**：
- `list_documents()` — 列出所有已索引文档的 ID、名称、摘要
- `get_doc_tree(doc_id)` — 获取指定文档的完整目录树（无正文）
- `get_node_content(node_ids)` — 批量获取节点的完整正文

**中间件**：SummarizationMiddleware（压缩超长历史）、ToolRetryMiddleware（工具调用 3 次重试 + 指数退避）、ModelFallbackMiddleware（模型故障自动切换）

### 2.3 与向量检索的本质区别

| | 传统向量 RAG | NaviRAG |
|---|---|---|
| 索引方式 | Embedding → 向量相似度 | 文档树 → LLM 推理导航 |
| 检索粒度 | 固定 chunk（语义断裂） | 原生章节（保留文档结构） |
| 可解释性 | 黑盒分数排序 | 树路径 + 溯源引用 |
| 跨文档 | 需额外路由逻辑 | 全局索引 + 逐文档探索 |

---

## 3. Docker 部署

### 3.1 前期准备

```bash
git clone <repo-url>
cd NaviRAG
```

在 `backend/` 目录下创建 `.env` 文件（参考 `.env.example`）：

```env
# 数据库（Docker 模式使用组件方式配置）
POSTGRES_SERVER=postgres
POSTGRES_PORT=5432
POSTGRES_USER=navirag
POSTGRES_PASSWORD=navirag
POSTGRES_DB=navirag

# LLM API（OpenAI 兼容接口）
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 应用
DEBUG=false
```

### 3.2 一键启动

```bash
docker compose -f docker/docker-compose.yml up -d
```

启动三个服务：
- **postgres** — PostgreSQL 16，数据持久化在 `pgdata` 卷
- **backend** — FastAPI 后端，端口 8000，等待 postgres 健康检查通过后启动
- **frontend** — Nginx + React 前端，端口 80，反向代理 `/api/` 到后端

### 3.3 数据库迁移

```bash
# 进入 backend 目录
cd backend

# 执行迁移
uv run alembic upgrade head
```

### 3.4 验证部署

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 打开前端
# 浏览器访问 http://localhost
```

### 3.5 本地开发

```bash
# 1. 启动 PostgreSQL
docker compose -f docker/docker-compose.yml up postgres -d

# 2. 安装后端依赖 + 启动
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000

# 3. 安装前端依赖 + 启动
cd frontend
npm install
npm run dev        # http://localhost:5173，API 自动代理到 localhost:8000
```

本地开发时 `.env` 中的数据库配置使用：
```env
POSTGRES_SERVER=localhost   # 非 Docker 环境下用 localhost
POSTGRES_PORT=5432
POSTGRES_USER=navirag
POSTGRES_PASSWORD=navirag
POSTGRES_DB=navirag
```

### 3.6 项目结构

```
NaviRAG/
├── backend/
│   ├── app/
│   │   ├── agent/          # ReAct Agent + 检索工具 + 中间件
│   │   ├── api/v1/          # REST API: chat, documents, health
│   │   ├── core/            # 共享模型定义
│   │   ├── db/              # SQLAlchemy ORM + 查询函数
│   │   ├── indexing/        # Markdown 解析 + LLM 元数据生成
│   │   └── schemas/         # Pydantic 请求/响应模型
│   ├── migrations/          # Alembic 数据库迁移
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/      # React 组件
│   │   ├── hooks/           # useChat 等自定义 Hook
│   │   ├── api/             # API 客户端 + SSE 流解析
│   │   └── types/           # TypeScript 类型定义
│   ├── Dockerfile
│   └── nginx.conf
├── docker/
│   └── docker-compose.yml
└── README.md
```
