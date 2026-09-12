"""ChatService 检索结果字段传递测试。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.chat_service import ChatService


def test_non_stream_chat_returns_document_title_with_reference():
    """参考资料应保留检索到的文档标题，供前端展示来源。"""
    retrieval = MagicMock()
    retrieval.retrieve.return_value = [
        {
            "doc_id": "doc-1",
            "doc_title": "CBAM模块论文.pdf",
            "chunk_index": 2,
            "content": "CBAM 通过通道和空间注意力增强特征。",
            "heading_path": "模块说明",
            "score": 0.91,
        }
    ]
    session = MagicMock()
    session.get_history.return_value = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="测试回答"))]
    )

    with patch("app.services.chat_service.AsyncOpenAI"):
        service = ChatService(retrieval_service=retrieval, session_service=session)
    service._client.chat.completions.create = AsyncMock(return_value=response)

    result = asyncio.run(service.chat("session-1", "CBAM 是什么？"))

    assert result["retrieved_chunks"][0]["doc_title"] == "CBAM模块论文.pdf"
