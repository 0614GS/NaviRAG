"""聊天 API — SSE 流式对话 + 会话管理

Agentic 模式: Agent 自主使用 list_documents / get_doc_tree / get_node_content
工具检索文档，不再经过固定的 workflow 管道。
"""

import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.db.session import get_db_session
from app.db.models import ChatSession, ChatMessage
from app.schemas.chat import ChatRequest, ChatSessionCreate, ChatSessionSummary, ChatMessageResponse

router = APIRouter()

HISTORY_LIMIT = 20


async def sse_event(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def chat_stream(query: str, session_id: str) -> AsyncGenerator[str, None]:
    """Agentic 聊天流：Agent 自主检索 + 流式回答"""
    from app.agent import get_agent

    # 1. 加载对话历史
    async with get_db_session() as db:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == uuid.UUID(session_id))
            .order_by(ChatMessage.created_at.desc())
            .limit(HISTORY_LIMIT)
        )
        history = result.scalars().all()[::-1]  # 最早的消息在前

    # 2. 构建消息列表（系统提示词在 agent 创建时已设置，这里只传对话）
    messages = [{"role": msg.role, "content": msg.content} for msg in history]

    # 3. 流式调用 agent
    agent = await get_agent()
    full_response = ""
    retrieved_sources: list[dict] = []  # 收集检索到的节点作为溯源引用

    yield await sse_event("message", {"type": "status", "content": "正在检索文档..."})

    try:
        async for event in agent.astream_events(
            {"messages": messages},
            version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_tool_start":
                name = event.get("name", "")
                yield await sse_event("message", {
                    "type": "status",
                    "stage": name,
                    "content": f"检索中: {name}..."
                })

            elif kind == "on_tool_end":
                name = event.get("name", "")
                output = event.get("data", {}).get("output")
                # 收集 get_node_content 返回的节点作为溯源引用
                raw_nodes = _extract_raw_nodes(output)
                for node in raw_nodes:
                    if not any(s["node_id"] == node["node_id"] for s in retrieved_sources):
                        retrieved_sources.append(node)
                content = _serialize_tool_output(output)
                yield await sse_event("message", {
                    "type": "retrieval",
                    "stage": name,
                    "content": content,
                })

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", "")
                if hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    full_response += token
                    yield await sse_event("message", {
                        "type": "token",
                        "content": token,
                    })

    except Exception as e:
        yield await sse_event("message", {
            "type": "error",
            "content": f"处理出错: {str(e)}",
        })
        if full_response:
            await _save_assistant_message(session_id, full_response)
        return

    # 4. 保存助手回答并记录溯源引用
    if full_response:
        await _save_assistant_message(session_id, full_response, retrieved_sources)

    yield await sse_event("message", {
        "type": "done",
        "content": {
            "full_response": full_response,
            "sources": [
                {
                    "index": i + 1,
                    "node_id": s["node_id"],
                    "doc_id": s.get("doc_id", ""),
                    "title": s.get("title", ""),
                    "path": s.get("path", ""),
                }
                for i, s in enumerate(retrieved_sources)
            ],
        },
    })


async def _save_assistant_message(session_id: str, content: str, sources: list[dict] | None = None):
    """保存助手回答到数据库，同时写入溯源引用"""
    try:
        async with get_db_session() as db:
            msg = ChatMessage(
                session_id=uuid.UUID(session_id),
                role="assistant",
                content=content,
            )
            db.add(msg)
            await db.flush()

            if sources:
                from app.db.models import SourceReference
                for i, src in enumerate(sources):
                    ref_id = str(uuid.uuid4())[:8]
                    ref = SourceReference(
                        id=ref_id,
                        message_id=msg.id,
                        node_id=src["node_id"],
                        doc_id=src.get("doc_id", ""),
                        relevance_score=1.0 - i * 0.1,
                    )
                    db.add(ref)

            await db.commit()
    except Exception as e:
        print(f"[Chat] 保存助手消息失败: {e}")


def _extract_raw_nodes(output) -> list[dict]:
    """从 tool output 中提取原始节点数据"""
    import json as _json

    # LangChain ToolMessage
    if hasattr(output, "content") and hasattr(output, "tool_call_id"):
        raw = output.content
        try:
            parsed = _json.loads(raw)
            return _extract_raw_nodes(parsed)
        except (_json.JSONDecodeError, TypeError):
            return []

    if isinstance(output, list):
        nodes = []
        for item in output:
            if isinstance(item, dict) and "node_id" in item:
                nodes.append(item)
        return nodes

    return []


def _serialize_tool_output(output) -> dict:
    """将工具输出序列化为前端可展示的格式"""
    import json as _json

    # LangChain ToolMessage: 提取 .content 字符串
    if hasattr(output, "content") and hasattr(output, "tool_call_id"):
        raw = output.content
        try:
            parsed = _json.loads(raw)
            return _serialize_tool_output(parsed)
        except (_json.JSONDecodeError, TypeError):
            return {"raw": str(raw)[:500]}
    if output is None:
        return {"note": "无结果"}
    if isinstance(output, list):
        return {"count": len(output), "preview": _summarize_list(output)}
    if isinstance(output, dict):
        if "error" in output:
            return output
        if "tree_structure" in output:
            ts = output["tree_structure"]
            node_count = _count_tree_nodes(ts.get("nodes", []))
            return {
                "doc_name": output.get("doc_name", ""),
                "summary": output.get("summary", ""),
                "node_count": node_count,
            }
        return {k: str(v)[:200] for k, v in output.items()}
    return {"raw": str(output)[:500]}


def _summarize_list(items: list) -> list:
    """摘要列表内容"""
    result = []
    for item in items[:10]:
        if isinstance(item, dict):
            result.append({
                k: str(v)[:100] if not isinstance(v, (dict, list)) else "..."
                for k, v in item.items()
                if k != "content"  # 不展示正文
            })
        else:
            result.append(str(item)[:100])
    return result


def _count_tree_nodes(nodes: list) -> int:
    """递归统计树节点数"""
    count = len(nodes)
    for node in nodes:
        if "nodes" in node:
            count += _count_tree_nodes(node["nodes"])
    return count


@router.post("/chat/completions")
async def chat_completions(request: ChatRequest, http_request: Request = None):
    """聊天补全 (SSE 流式输出) — Agentic 检索模式"""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询不能为空")

    # 如果没有 session_id，创建新会话
    session_id = request.session_id
    if session_id is None:
        async with get_db_session() as db:
            title = query[:30] + ("..." if len(query) > 30 else "")
            session = ChatSession(title=title)
            db.add(session)
            await db.commit()
            await db.refresh(session)
            session_id = session.id

    # 保存用户消息
    async with get_db_session() as db:
        user_msg = ChatMessage(
            session_id=session_id,
            role="user",
            content=query,
        )
        db.add(user_msg)
        await db.commit()

    return StreamingResponse(
        chat_stream(query, str(session_id)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- 会话管理 API ---

@router.post("/chat/sessions", response_model=ChatSessionSummary)
async def create_session(body: ChatSessionCreate = None):
    """创建新会话"""
    async with get_db_session() as db:
        session = ChatSession(title=body.title if body else "新对话")
        db.add(session)
        await db.commit()
        await db.refresh(session)

    return ChatSessionSummary(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/chat/sessions", response_model=list[ChatSessionSummary])
async def list_sessions():
    """获取会话列表"""
    async with get_db_session() as db:
        result = await db.execute(
            select(ChatSession).order_by(ChatSession.updated_at.desc())
        )
        sessions = result.scalars().all()

    return [
        ChatSessionSummary(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sessions
    ]


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: uuid.UUID):
    """删除会话及其消息"""
    async with get_db_session() as db:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        await db.delete(session)
        await db.commit()

    return {"detail": "会话已删除"}


@router.get("/chat/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def list_messages(session_id: uuid.UUID):
    """获取会话历史消息"""
    async with get_db_session() as db:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        messages = result.scalars().all()

    return [
        ChatMessageResponse(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            metadata=m.meta_info or {},
            created_at=m.created_at,
        )
        for m in messages
    ]
