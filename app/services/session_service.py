"""
会话管理服务 —— 使用 Redis List 存储多轮对话历史。

存储结构：
  - Key:  session:{session_id}:history
  - Type: Redis List（LPUSH 追加，最新在前）
  - 元素: JSON 序列化的 {role, content, timestamp}

长对话处理策略：
  - 只保留最近 max_history_turns 轮（一问一答为一轮 = 2 条消息）
  - 超出时自动 LTRIM 截断旧消息
  - 检索结果不存入历史，仅当前轮使用
"""
from __future__ import annotations

import json
import time

import redis

from app.config import get_settings


class SessionService:
    """会话历史管理"""

    def __init__(self):
        settings = get_settings()
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        self._max_messages = settings.max_history_turns * 2  # 每轮 = user + assistant

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:history"

    # ------------------------------------------------------------------
    # 追加消息
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        retrieved_chunks: list[dict] | None = None,
    ) -> None:
        """向会话历史追加一条消息，并自动截断旧消息。"""
        message = {"role": role, "content": content, "timestamp": time.time()}
        if retrieved_chunks is not None:
            message["retrieved_chunks"] = retrieved_chunks
        msg = json.dumps(
            message,
            ensure_ascii=False,
        )
        key = self._key(session_id)
        pipe = self._redis.pipeline()
        pipe.lpush(key, msg)          # 最新在前
        pipe.ltrim(key, 0, self._max_messages - 1)  # 保留最近 N 条
        pipe.execute()

    # ------------------------------------------------------------------
    # 获取历史
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict]:
        """获取会话历史（按时间正序，旧 → 新）。"""
        key = self._key(session_id)
        raw = self._redis.lrange(key, 0, -1)  # 最新在前
        messages = [json.loads(item) for item in raw]
        messages.reverse()  # 翻转为正序
        return messages

    # ------------------------------------------------------------------
    # 清除会话
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> bool:
        """删除指定会话的全部历史记录。"""
        deleted = self._redis.delete(self._key(session_id))
        return deleted > 0

    # ------------------------------------------------------------------
    # 构建对话上下文（供 LLM 使用）
    # ------------------------------------------------------------------

    def build_context_messages(
        self,
        session_id: str,
        current_question: str,
    ) -> list[dict[str, str]]:
        """构建发送给 LLM 的 messages 数组。

        格式遵循 OpenAI Chat Completions:
          [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "历史问题1"},
            {"role": "assistant", "content": "历史回答1"},
            {"role": "user",   "content": "当前问题（含检索上下文）"},
          ]
        """
        history = self.get_history(session_id)
        messages: list[dict[str, str]] = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # 当前问题由调用方组装后替换最后一条 user 消息
        return messages

    # ------------------------------------------------------------------
    # 列出所有会话
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        """扫描所有会话，返回摘要列表（按最近活动时间倒序）。"""
        sessions = []
        for key in self._redis.scan_iter("session:*:history"):
            sid = key.removeprefix("session:").removesuffix(":history")
            raw = self._redis.lrange(key, 0, 0)  # 最新一条
            last_msg = ""
            updated_at = 0.0
            if raw:
                msg = json.loads(raw[0])
                last_msg = msg.get("content", "")[:100]
                updated_at = msg.get("timestamp", 0)
            length = self._redis.llen(key)
            sessions.append({
                "session_id": sid,
                "message_count": length,
                "last_message": last_msg,
                "updated_at": updated_at,
            })
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    # ------------------------------------------------------------------
    # 检查会话是否存在
    # ------------------------------------------------------------------

    def exists(self, session_id: str) -> bool:
        return self._redis.exists(self._key(session_id)) > 0
