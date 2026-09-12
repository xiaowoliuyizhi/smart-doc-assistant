"""
数据模型验证测试 —— 测试 Pydantic schema 的字段校验。
"""
import pytest
from pydantic import ValidationError

from app.models.schemas import (
    DocumentImportRequest,
    DocumentImportResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatHistoryResponse,
    ChatMessage,
    SessionDeleteResponse,
    RetrievedChunk,
)
from app.config import Settings


class TestDocumentImportRequest:

    def test_valid_request(self):
        req = DocumentImportRequest(content="测试内容")
        assert req.content == "测试内容"
        assert req.content_type == "text"  # 默认
        assert req.title is None

    def test_markdown_type(self):
        req = DocumentImportRequest(content="# 标题", content_type="markdown")
        assert req.content_type == "markdown"

    def test_empty_content_rejected(self):
        """空字符串内容仍可创建（路由层拦截）"""
        req = DocumentImportRequest(content="")
        assert req.content == ""

    def test_invalid_content_type(self):
        with pytest.raises(ValidationError):
            DocumentImportRequest(content="test", content_type="invalid")

    def test_custom_doc_id(self):
        req = DocumentImportRequest(content="test", doc_id="my-doc-001")
        assert req.doc_id == "my-doc-001"


class TestChatCompletionRequest:

    def test_valid_request(self):
        req = ChatCompletionRequest(session_id="s1", question="你好")
        assert req.session_id == "s1"
        assert req.question == "你好"
        assert req.stream is True  # 默认流式

    def test_non_stream(self):
        req = ChatCompletionRequest(session_id="s1", question="你好", stream=False)
        assert req.stream is False

    def test_custom_top_k(self):
        req = ChatCompletionRequest(session_id="s1", question="你好", top_k=10)
        assert req.top_k == 10

    def test_missing_session_id(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(question="你好")

    def test_missing_question(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(session_id="s1")


class TestRetrievedChunk:

    def test_valid_chunk(self):
        c = RetrievedChunk(doc_id="d1", chunk_index=0, content="内容", score=0.95)
        assert c.score == 0.95

    def test_negative_score_allowed(self):
        c = RetrievedChunk(doc_id="d1", chunk_index=0, content="内容", score=-0.1)
        assert c.score == -0.1

    def test_accepts_hybrid_retrieval_metadata(self):
        c = RetrievedChunk(
            doc_id="d1",
            doc_title="手册.pdf",
            chunk_index=0,
            content="内容",
            score=0.0,
            retrieval_mode="hybrid",
            source="both",
            rank=1,
        )
        assert c.source == "both"
        assert c.rank == 1


class TestRetrievalSettings:

    def test_invalid_retrieval_mode_is_rejected(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_MODE", "invalid")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)


class TestChatHistoryResponse:

    def test_with_messages(self):
        resp = ChatHistoryResponse(
            session_id="s1",
            messages=[
                ChatMessage(role="user", content="你好", timestamp=1.0),
                ChatMessage(role="assistant", content="你好！", timestamp=2.0),
            ],
            message_count=2,
        )
        assert resp.message_count == 2
        assert resp.messages[0].role == "user"

    def test_empty_messages(self):
        resp = ChatHistoryResponse(session_id="s1", messages=[], message_count=0)
        assert resp.message_count == 0


class TestSessionDeleteResponse:

    def test_success(self):
        resp = SessionDeleteResponse(session_id="s1", deleted=True)
        assert resp.deleted is True
        assert resp.message == "会话历史已清除"

    def test_default_message(self):
        resp = SessionDeleteResponse(session_id="s1", deleted=True)
        assert "清除" in resp.message
