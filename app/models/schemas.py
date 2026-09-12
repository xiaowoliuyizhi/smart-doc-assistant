"""
Pydantic 数据模型 —— 请求 / 响应统一定义。
"""
from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# 1. POST /api/v1/document/import & /upload
# ============================================================

class DocumentImportRequest(BaseModel):
    """文档导入请求"""
    content: str = Field(..., description="文档正文，支持 Markdown 或纯文本")
    title: Optional[str] = Field(None, description="文档标题（方便后续溯源）")
    doc_id: Optional[str] = Field(None, description="自定义文档编号，留空则自动生成")
    content_type: Literal["markdown", "text"] = Field(
        "text", description="内容格式：markdown 会按标题层级切分，text 按段落切分"
    )


class DocumentImportResponse(BaseModel):
    """文档导入响应"""
    doc_id: str = Field(..., description="文档唯一标识")
    title: Optional[str] = Field(None, description="文档标题")
    chunk_count: int = Field(..., description="切分后的片段数量")
    total_chars: int = Field(..., description="文档总字符数")
    imported_at: float = Field(default_factory=time.time, description="导入时间戳")


class DocListItem(BaseModel):
    """知识库文档摘要"""
    doc_id: str = Field(..., description="文档编号")
    title: str = Field("", description="文档标题")
    chunk_count: int = Field(0, description="切片数量")
    total_chars: int = Field(0, description="字符总数")
    content_type: str = Field("text", description="格式类型")
    imported_at: float = Field(0, description="导入时间戳")


class DocListResponse(BaseModel):
    """知识库文档列表"""
    documents: list[DocListItem] = Field(default_factory=list, description="文档列表")
    total: int = Field(0, description="文档总数")


# ============================================================
# 2. POST /api/v1/chat/completions
# ============================================================

class ChatCompletionRequest(BaseModel):
    """对话请求"""
    session_id: str = Field(..., description="会话编号（同一会话下 AI 会记住上下文）")
    question: str = Field(..., description="你的提问")
    stream: bool = Field(True, description="是否流式输出（默认开启，逐字返回回答）")
    top_k: Optional[int] = Field(None, description="检索片段数，留空使用默认值 4")


class RetrievedChunk(BaseModel):
    """单个检索片段"""
    doc_id: str = Field(..., description="来源文档编号")
    doc_title: Optional[str] = Field(None, description="来源文档标题或文件名")
    chunk_index: int = Field(..., description="片段序号")
    content: str = Field(..., description="片段文本内容")
    score: float = Field(..., description="语义相似度（0~1，越接近 1 越相关）")
    retrieval_mode: Optional[Literal["vector", "hybrid"]] = Field(
        None, description="检索模式"
    )
    source: Optional[Literal["vector", "bm25", "both"]] = Field(
        None, description="命中的检索来源"
    )
    rank: Optional[int] = Field(None, description="最终结果排名")


class ChatCompletionResponse(BaseModel):
    """非流式对话响应（stream=false 时返回）"""
    session_id: str = Field(..., description="会话编号")
    answer: str = Field(..., description="AI 完整回答")
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list, description="检索到的相关文档片段")
    created_at: float = Field(default_factory=time.time, description="回答生成时间戳")


# ============================================================
# 3. GET /api/v1/chat/history/{sessionId}
# ============================================================

class ChatMessage(BaseModel):
    """单条对话消息"""
    role: Literal["user", "assistant"] = Field(..., description="发言角色：user（你）/ assistant（AI）")
    content: str = Field(..., description="消息正文")
    timestamp: float = Field(..., description="发送时间戳")
    retrieved_chunks: Optional[list[RetrievedChunk]] = Field(
        None, description="该助手回答使用的参考资料"
    )


class ChatHistoryResponse(BaseModel):
    """会话历史响应"""
    session_id: str = Field(..., description="会话编号")
    messages: list[ChatMessage] = Field(..., description="对话记录列表（按时间顺序）")
    message_count: int = Field(..., description="消息总数")


# ============================================================
# 4. DELETE /api/v1/chat/session/{sessionId}
# ============================================================

class SessionDeleteResponse(BaseModel):
    """会话清除响应"""
    session_id: str = Field(..., description="已清除的会话编号")
    deleted: bool = Field(..., description="是否成功删除")
    message: str = Field("会话历史已清除", description="操作结果说明")


# ============================================================
# SSE 流式事件（stream=true 时逐条推送）
# ============================================================

class SSEChunk(BaseModel):
    """SSE 流式推送的数据块"""
    session_id: str = Field(..., description="会话编号")
    delta: str = Field("", description="本次增量文本（逐字返回）")
    retrieved_chunks: Optional[list[RetrievedChunk]] = Field(
        None, description="仅在首个事件中携带：检索到的相关文档片段"
    )
    finish: bool = Field(False, description="是否已结束（最后一个事件为 true）")


# ============================================================
# 会话列表
# ============================================================

class SessionItem(BaseModel):
    """会话摘要"""
    session_id: str = Field(..., description="会话编号")
    message_count: int = Field(0, description="消息数")
    last_message: str = Field("", description="最后一条消息预览")
    updated_at: float = Field(0, description="最后活动时间戳")


class SessionListResponse(BaseModel):
    """会话列表"""
    sessions: list[SessionItem] = Field(default_factory=list, description="会话列表")
    total: int = Field(0, description="会话总数")
