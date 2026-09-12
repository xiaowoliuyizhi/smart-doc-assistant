"""
会话管理服务测试 —— 使用 mock Redis。
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.session_service import SessionService


class TestSessionService:
    """测试会话管理逻辑"""

    @pytest.fixture
    def session_service(self):
        """创建带 mock Redis 的 SessionService"""
        with patch("app.services.session_service.redis") as mock_redis_mod:
            mock_redis = MagicMock()
            mock_redis_mod.from_url.return_value = mock_redis
            svc = SessionService()
            svc._redis = mock_redis
            return svc, mock_redis

    def test_add_message(self, session_service):
        svc, mock_redis = session_service
        pipe = MagicMock()
        mock_redis.pipeline.return_value = pipe

        svc.add_message("sess-1", "user", "你好")

        pipe.lpush.assert_called_once()
        pipe.ltrim.assert_called_once()
        pipe.execute.assert_called_once()
        # 验证 lpush 的 key
        lpush_args = pipe.lpush.call_args
        assert "session:sess-1:history" in str(lpush_args)

    def test_get_history_empty(self, session_service):
        svc, mock_redis = session_service
        mock_redis.lrange.return_value = []

        history = svc.get_history("sess-empty")
        assert history == []

    def test_get_history_with_messages(self, session_service):
        svc, mock_redis = session_service
        # Redis List: 最新在前，所以 assistant 在前
        mock_redis.lrange.return_value = [
            json.dumps({"role": "assistant", "content": "回答", "timestamp": 2.0}),
            json.dumps({"role": "user", "content": "问题", "timestamp": 1.0}),
        ]

        history = svc.get_history("sess-1")
        # 应翻转为正序：user 在前
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "问题"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "回答"

    def test_add_assistant_message_persists_retrieved_chunks(self, session_service):
        """助手回答携带的参考片段应随会话历史保存。"""
        svc, mock_redis = session_service
        pipe = MagicMock()
        mock_redis.pipeline.return_value = pipe
        sources = [
            {"doc_id": "doc-1", "chunk_index": 2, "content": "参考内容", "score": 0.91}
        ]

        svc.add_message("sess-1", "assistant", "回答", retrieved_chunks=sources)

        stored_message = json.loads(pipe.lpush.call_args.args[1])
        assert stored_message["retrieved_chunks"] == sources

    def test_clear_session_success(self, session_service):
        svc, mock_redis = session_service
        mock_redis.delete.return_value = 1

        result = svc.clear_session("sess-1")
        assert result is True
        mock_redis.delete.assert_called_once_with("session:sess-1:history")

    def test_clear_session_not_found(self, session_service):
        svc, mock_redis = session_service
        mock_redis.delete.return_value = 0

        result = svc.clear_session("nonexistent")
        assert result is False

    def test_exists_true(self, session_service):
        svc, mock_redis = session_service
        mock_redis.exists.return_value = 1

        assert svc.exists("sess-1") is True

    def test_exists_false(self, session_service):
        svc, mock_redis = session_service
        mock_redis.exists.return_value = 0

        assert svc.exists("nonexistent") is False

    def test_max_messages_config(self, session_service):
        """验证 max_history_turns * 2 的消息上限"""
        svc, mock_redis = session_service
        # 默认 max_history_turns=6 → max_messages=12
        assert svc._max_messages == 12

    def test_build_context_messages(self, session_service):
        svc, mock_redis = session_service
        mock_redis.lrange.return_value = [
            json.dumps({"role": "assistant", "content": "历史回答", "timestamp": 2.0}),
            json.dumps({"role": "user", "content": "历史问题", "timestamp": 1.0}),
        ]

        messages = svc.build_context_messages("sess-1", "当前问题")
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "历史问题"}
        assert messages[1] == {"role": "assistant", "content": "历史回答"}
