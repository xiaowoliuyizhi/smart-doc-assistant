"""
对话路由 —— POST /api/v1/chat/completions

支持流式（SSE）与非流式两种模式。
"""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["对话问答"])


@router.post(
    "/completions",
    summary="发起对话",
    description="""
基于 RAG 的智能问答接口。

- **stream=true**（默认）：以 SSE 格式流式返回，Content-Type: text/event-stream。首个事件包含检索到的文档片段，后续事件逐字推送 AI 回答。
- **stream=false**：返回完整 JSON 响应，包含回答和检索片段。

支持多轮对话，同一个 session_id 下 AI 会理解对话历史上下文。
    """,
)
async def chat_completions(req: ChatCompletionRequest):
    chat_service = ChatService()

    if req.stream:
        return StreamingResponse(
            chat_service.stream_chat(
                session_id=req.session_id,
                question=req.question,
                top_k=req.top_k,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            result = await chat_service.chat(
                session_id=req.session_id,
                question=req.question,
                top_k=req.top_k,
            )
            return ChatCompletionResponse(**result)
        except Exception as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=502,
                content={"detail": f"LLM 调用失败: {str(e)}"},
            )
