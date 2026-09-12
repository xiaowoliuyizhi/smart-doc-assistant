"""
API 接口测试 —— 使用 FastAPI TestClient + Mock 外部服务。

覆盖 4 个核心接口：
  1. POST   /api/v1/document/import
  2. POST   /api/v1/chat/completions     (流式 + 非流式)
  3. GET    /api/v1/chat/history/{sessionId}
  4. DELETE /api/v1/chat/session/{sessionId}
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """创建测试客户端，mock 掉所有外部依赖"""
    with patch("app.services.vector_store.VectorStore") as mock_vs_class, \
         patch("app.services.embedding_service.OpenAI") as mock_openai:
        mock_vs = MagicMock()
        mock_vs.ping.return_value = True
        mock_vs.ensure_index.return_value = None
        mock_vs.count_chunks.return_value = 0
        mock_vs_class.return_value = mock_vs

        from app.main import app
        with TestClient(app) as c:
            yield c


# ======================================================================
# 1. POST /api/v1/document/import
# ======================================================================

class TestDocumentImport:

    def test_import_success(self, client, mock_embedding_service, mock_vector_store):
        """测试文档导入成功"""
        with patch(
            "app.routers.document.DocumentChunker"
        ) as mock_chunker_class, \
             patch(
            "app.routers.document.EmbeddingService"
        ) as mock_emb_class, \
             patch(
            "app.routers.document.VectorStore"
        ) as mock_vs_class:

            # 配置 mock
            mock_chunker = MagicMock()
            mock_chunker.split.return_value = [
                MagicMock(doc_id="doc_1", chunk_index=0, content="内容A", heading_path=""),
                MagicMock(doc_id="doc_1", chunk_index=1, content="内容B", heading_path=""),
                MagicMock(doc_id="doc_1", chunk_index=2, content="内容C", heading_path=""),
            ]
            mock_chunker_class.return_value = mock_chunker

            mock_emb = mock_embedding_service
            mock_emb_class.return_value = mock_emb

            mock_vs = mock_vector_store
            mock_vs_class.return_value = mock_vs

            resp = client.post("/api/v1/document/import", json={
                "content": "这是一段测试文档内容。" * 20,
                "title": "测试文档",
                "content_type": "text",
            })

            assert resp.status_code == 200
            data = resp.json()
            assert "doc_id" in data
            assert isinstance(data["doc_id"], str)
            assert len(data["doc_id"]) > 0
            assert data["chunk_count"] == 3
            assert data["title"] == "测试文档"
            assert data["total_chars"] > 0

    def test_import_empty_content(self, client):
        """空内容应返回 400"""
        resp = client.post("/api/v1/document/import", json={
            "content": "",
        })
        assert resp.status_code == 400

    def test_import_markdown(self, client, mock_embedding_service, mock_vector_store):
        """测试 Markdown 文档导入"""
        with patch("app.routers.document.DocumentChunker") as mock_chunker_class, \
             patch("app.routers.document.EmbeddingService") as mock_emb_class, \
             patch("app.routers.document.VectorStore") as mock_vs_class:

            mock_chunker = MagicMock()
            mock_chunker.split.return_value = [
                MagicMock(doc_id="doc_md", chunk_index=0, content="# 标题\n内容", heading_path="标题"),
            ]
            mock_chunker_class.return_value = mock_chunker
            mock_emb_class.return_value = mock_embedding_service
            mock_vs_class.return_value = mock_vector_store

            resp = client.post("/api/v1/document/import", json={
                "content": "# 标题\n\n内容",
                "content_type": "markdown",
            })

            assert resp.status_code == 200
            assert resp.json()["chunk_count"] == 1
            # 验证调用了 markdown 切片
            call_kwargs = mock_chunker.split.call_args[1]
            assert call_kwargs["content_type"] == "markdown"


# ======================================================================
# 2. POST /api/v1/chat/completions
# ======================================================================

class TestChatCompletions:

    def test_non_stream_chat(self, client, mock_chat_service):
        """测试非流式对话"""
        with patch("app.routers.chat.ChatService") as mock_cs_class:
            mock_cs_class.return_value = mock_chat_service

            resp = client.post("/api/v1/chat/completions", json={
                "session_id": "test-session",
                "question": "Redis 是什么？",
                "stream": False,
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == "test-session"
            assert data["answer"] == "这是一个测试回答"
            assert len(data["retrieved_chunks"]) == 1
            assert data["retrieved_chunks"][0]["doc_id"] == "doc_1"

    def test_stream_chat(self, client, mock_chat_service):
        """测试流式对话（SSE）"""
        with patch("app.routers.chat.ChatService") as mock_cs_class:
            mock_cs_class.return_value = mock_chat_service

            resp = client.post("/api/v1/chat/completions", json={
                "session_id": "test-session",
                "question": "Redis 是什么？",
                "stream": True,
            })

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # 解析 SSE 事件
            body = resp.text
            events = [
                line.replace("data: ", "")
                for line in body.strip().split("\n")
                if line.startswith("data: ")
            ]
            assert len(events) >= 3  # 至少：首条 + delta + finish

            # 第一个事件应包含 retrieved_chunks
            first = json.loads(events[0])
            assert "retrieved_chunks" in first
            assert first["finish"] is False

            # 最后一个事件应 finish=true
            last = json.loads(events[-1])
            assert last["finish"] is True

    def test_chat_default_stream(self, client, mock_chat_service):
        """不传 stream 时默认为流式"""
        with patch("app.routers.chat.ChatService") as mock_cs_class:
            mock_cs_class.return_value = mock_chat_service

            resp = client.post("/api/v1/chat/completions", json={
                "session_id": "test-session",
                "question": "测试",
            })

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_chat_missing_session_id(self, client):
        """缺少 session_id 应返回 422"""
        resp = client.post("/api/v1/chat/completions", json={
            "question": "测试",
        })
        assert resp.status_code == 422

    def test_chat_missing_question(self, client):
        """缺少 question 应返回 422"""
        resp = client.post("/api/v1/chat/completions", json={
            "session_id": "test",
        })
        assert resp.status_code == 422


# ======================================================================
# 3. GET /api/v1/chat/history/{sessionId}
# ======================================================================

class TestChatHistory:

    def test_get_history_with_messages(self, client):
        """测试获取有消息的会话历史"""
        with patch("app.routers.session.SessionService") as mock_ss_class:
            mock_ss = MagicMock()
            mock_ss.get_history.return_value = [
                {"role": "user", "content": "你好", "timestamp": 1700000000.0},
                {"role": "assistant", "content": "你好！有什么可以帮你的？", "timestamp": 1700000001.0},
            ]
            mock_ss_class.return_value = mock_ss

            resp = client.get("/api/v1/chat/history/my-session")

            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == "my-session"
            assert data["message_count"] == 2
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][1]["role"] == "assistant"

    def test_get_history_returns_assistant_references(self, client):
        """重新加载会话时，每条助手回答都应携带自己的参考资料。"""
        sources = [
            {
                "doc_id": "doc-1",
                "doc_title": "参考文档.pdf",
                "chunk_index": 2,
                "content": "参考内容",
                "score": 0.91,
            }
        ]
        with patch("app.routers.session.SessionService") as mock_ss_class:
            mock_ss = MagicMock()
            mock_ss.get_history.return_value = [
                {"role": "user", "content": "问题", "timestamp": 1700000000.0},
                {
                    "role": "assistant",
                    "content": "回答",
                    "timestamp": 1700000001.0,
                    "retrieved_chunks": sources,
                },
            ]
            mock_ss_class.return_value = mock_ss

            resp = client.get("/api/v1/chat/history/my-session")

            assert resp.status_code == 200
            source = resp.json()["messages"][1]["retrieved_chunks"][0]
            assert source["doc_title"] == "参考文档.pdf"
            assert source["retrieval_mode"] is None

    def test_get_history_empty(self, client):
        """测试获取空会话历史"""
        with patch("app.routers.session.SessionService") as mock_ss_class:
            mock_ss = MagicMock()
            mock_ss.get_history.return_value = []
            mock_ss_class.return_value = mock_ss

            resp = client.get("/api/v1/chat/history/empty-session")

            assert resp.status_code == 200
            data = resp.json()
            assert data["message_count"] == 0
            assert data["messages"] == []


# ======================================================================
# 4. DELETE /api/v1/chat/session/{sessionId}
# ======================================================================

class TestSessionDelete:

    def test_delete_session_success(self, client):
        """测试成功删除会话"""
        with patch("app.routers.session.SessionService") as mock_ss_class:
            mock_ss = MagicMock()
            mock_ss.clear_session.return_value = True
            mock_ss_class.return_value = mock_ss

            resp = client.delete("/api/v1/chat/session/my-session")

            assert resp.status_code == 200
            data = resp.json()
            assert data["deleted"] is True
            assert data["session_id"] == "my-session"

    def test_delete_session_not_found(self, client):
        """测试删除不存在的会话"""
        with patch("app.routers.session.SessionService") as mock_ss_class:
            mock_ss = MagicMock()
            mock_ss.clear_session.return_value = False
            mock_ss_class.return_value = mock_ss

            resp = client.delete("/api/v1/chat/session/nonexistent")

            assert resp.status_code == 404


# ======================================================================
# 健康检查
# ======================================================================

class TestHealth:

    def test_health_check(self, client):
        """测试健康检查端点"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "redis" in data

    def test_root(self, client):
        """测试根路由"""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "智能文档助手"
