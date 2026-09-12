"""
Pytest 公共 fixtures —— 模拟外部依赖 (Redis / LLM) 使测试可独立运行。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def fake_embedding_dim():
    return 8  # 测试用低维向量


@pytest.fixture
def fake_vector():
    return [0.1] * 8


@pytest.fixture
def mock_embedding_service(fake_vector):
    """模拟 EmbeddingService，返回固定向量"""
    mock = MagicMock()
    mock.embed.return_value = fake_vector
    mock.embed_batch.return_value = [fake_vector] * 100
    mock.dim = 8
    return mock


@pytest.fixture
def mock_vector_store():
    """模拟 VectorStore"""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.ensure_index.return_value = None
    mock.store_chunks.return_value = 3
    mock.count_chunks.return_value = 3
    mock.search.return_value = [
        {
            "doc_id": "doc_test_001",
            "chunk_index": 0,
            "content": "Redis 是一个高性能的键值数据库，支持多种数据结构。",
            "heading_path": "Redis 简介",
            "score": 0.95,
        },
        {
            "doc_id": "doc_test_001",
            "chunk_index": 1,
            "content": "RediSearch 是 Redis 的搜索模块，支持全文检索和向量相似度查询。",
            "heading_path": "RediSearch",
            "score": 0.88,
        },
    ]
    mock.delete_doc.return_value = 2
    return mock


@pytest.fixture
def mock_session_service():
    """模拟 SessionService"""
    mock = MagicMock()
    mock.add_message.return_value = None
    mock.get_history.return_value = []
    mock.clear_session.return_value = True
    mock.exists.return_value = True
    mock.build_context_messages.return_value = []
    return mock


@pytest.fixture
def mock_chat_service():
    """模拟 ChatService 的流式输出"""
    mock = MagicMock()

    async def fake_stream(*args, **kwargs):
        events = [
            'data: {"session_id": "test-session", "delta": "", "retrieved_chunks": [{"doc_id": "doc_1", "chunk_index": 0, "content": "test", "score": 0.9}], "finish": false}\n\n',
            'data: {"session_id": "test-session", "delta": "这是", "finish": false}\n\n',
            'data: {"session_id": "test-session", "delta": "一个测试回答", "finish": false}\n\n',
            'data: {"session_id": "test-session", "delta": "", "finish": true}\n\n',
        ]
        for e in events:
            yield e

    mock.stream_chat = fake_stream
    mock.chat = AsyncMock(return_value={
        "session_id": "test-session",
        "answer": "这是一个测试回答",
        "retrieved_chunks": [
            {"doc_id": "doc_1", "chunk_index": 0, "content": "test", "score": 0.9}
        ],
        "created_at": 1700000000.0,
    })
    return mock
