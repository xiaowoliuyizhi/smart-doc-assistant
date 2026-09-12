"""
会话管理路由 ——
  GET    /api/v1/chat/sessions       列出所有会话
  GET    /api/v1/chat/history/{sessionId}  查看对话历史
  DELETE /api/v1/chat/session/{sessionId}  清除会话
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ChatHistoryResponse,
    ChatMessage,
    SessionDeleteResponse,
    SessionItem,
    SessionListResponse,
)
from app.services.session_service import SessionService

router = APIRouter(prefix="/api/v1/chat", tags=["会话管理"])


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="会话列表",
    description="列出所有会话及其最近活动摘要。",
)
async def list_sessions():
    svc = SessionService()
    raw = svc.list_sessions()
    sessions = [SessionItem(**s) for s in raw]
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="查看会话历史",
    description="获取指定会话的完整对话记录（用户提问 + AI 回答），按时间正序排列。",
)
async def get_chat_history(session_id: str):
    svc = SessionService()
    messages = svc.get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatMessage(**m) for m in messages],
        message_count=len(messages),
    )


@router.delete(
    "/session/{session_id}",
    response_model=SessionDeleteResponse,
    summary="清除会话",
    description="重置/删除指定会话的所有对话历史记录。",
)
async def delete_session(session_id: str):
    svc = SessionService()
    deleted = svc.clear_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"会话 {session_id} 不存在或已为空"
        )
    return SessionDeleteResponse(
        session_id=session_id, deleted=True, message="会话历史已清除"
    )
