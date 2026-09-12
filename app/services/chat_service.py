"""
RAG 问答服务 —— 检索增强生成的核心链路。

流程：
  1. 将用户问题向量化，检索 Top-K 相关片段
  2. 组装 System Prompt（强调严格基于文档）+ 检索上下文 + 对话历史 + 当前问题
  3. 调用 LLM，流式产出回答
  4. 将本轮问答存入会话历史
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.retrieval_service import RetrievalService
from app.services.session_service import SessionService

# ---- System Prompt ----
SYSTEM_PROMPT = """你是一个智能文档助手。请严格根据下方「参考资料」回答用户的问题。

规则：
1. 回答必须基于参考资料中的内容，不得编造、猜测或脱离资料发挥。
2. 如果参考资料中没有相关信息，请明确回答"根据现有文档，我无法回答这个问题"。
3. 回答时可以适当组织语言使其通顺，但不得改变原文含义。
4. 如果多个片段涉及同一问题，请综合归纳后回答。
5. 在回答末尾，可注明引用了哪些文档片段（如"参考：doc_xxx 片段 0,2"）。"""

CONTEXT_TEMPLATE = """以下是从知识库中检索到的参考资料：

{context}

请基于以上资料回答用户的问题。如果资料不足，请如实说明。"""


class ChatService:
    """RAG 问答服务"""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        session_service: SessionService | None = None,
    ):
        settings = get_settings()
        self._client = AsyncOpenAI(
            base_url=settings.llm_api_base,
            api_key=settings.llm_api_key,
        )
        self._model = settings.llm_model
        self._top_k = settings.top_k
        self._retrieval = retrieval_service or RetrievalService()
        self._session = session_service or SessionService()

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def _retrieve_context(self, question: str, top_k: int) -> list[dict]:
        """检索相关片段并格式化为上下文文本。"""
        chunks = self._retrieval.retrieve(question, top_k=top_k)
        return chunks

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        """将检索片段格式化为 prompt 中的上下文块。"""
        if not chunks:
            return "（未检索到相关文档片段）"
        parts: list[str] = []
        for i, c in enumerate(chunks):
            heading = c.get("heading_path", "")
            prefix = f"[{heading}] " if heading else ""
            parts.append(
                f"--- 片段 {i + 1}（文档: {c['doc_id']}, 序号: {c['chunk_index']}, "
                f"相似度: {c['score']}）---\n{prefix}{c['content']}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # 组装 messages
    # RAG 流程&#x4E2D;__&#x7EC4;装发给大语言模型（LLM）的完整消息列表核心方法。它把"系统指令 + 对话历史 + 当前问题 + 检索到的上下文"拼装成 OpenAI 兼容的 `messages` 格式
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        chunks: list[dict],
        session_id: str,
    ) -> list[dict[str, str]]:
        """组装发送给 LLM 的完整 messages。"""
        context_text = self._format_context(chunks)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # 加入对话历史（不包含当前问题）
        history = self._session.get_history(session_id)
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 当前问题 + 检索上下文
        user_content = f"{CONTEXT_TEMPLATE.format(context=context_text)}\n\n用户问题：{question}"
        messages.append({"role": "user", "content": user_content})

        return messages

    # ------------------------------------------------------------------
    # 流式问答
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
    ) -> AsyncIterator[str]:
        """流式 RAG 问答，逐条 yield SSE 格式的 JSON 字符串。

        每条格式: data: {json}\n\n
        - 首条: 携带 retrieved_chunks
        - 中间: 携带 delta（增量文本）
        - 末条: finish=true
        """
        k = top_k or self._top_k

        # 1. 检索
        chunks = self._retrieve_context(question, k)

        # 2. 组装 messages
        messages = self._build_messages(question, chunks, session_id)

        # 3. 先保存用户问题到历史
        self._session.add_message(session_id, "user", question)

        # 4. 首条 SSE：携带检索结果
        first_event = {
            "session_id": session_id,
            "delta": "",
            "retrieved_chunks": [
                {
                    "doc_id": c["doc_id"],
                    "doc_title": c.get("doc_title"),
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "score": c["score"],
                    "retrieval_mode": c.get("retrieval_mode"),
                    "source": c.get("source"),
                    "rank": c.get("rank"),
                }
                for c in chunks
            ],
            "finish": False,
        }
        yield f"data: {json.dumps(first_event, ensure_ascii=False)}\n\n"

        # 5. 流式调用 LLM
        full_answer: list[str] = []
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                temperature=0.3,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer.append(delta)
                    event = {
                        "session_id": session_id,
                        "delta": delta,
                        "finish": False,
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            import re
            err_msg = str(e)
            # 清理特殊字符，防止 JSON 序列化问题
            err_msg = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', err_msg)
            error_event = {
                "session_id": session_id,
                "delta": f"\n\n[生成错误: {err_msg[:300]}]",
                "finish": False,
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

        # 6. 保存助手回答到历史
        answer_text = "".join(full_answer)
        if answer_text:
            self._session.add_message(
                session_id,
                "assistant",
                answer_text,
                retrieved_chunks=first_event["retrieved_chunks"],
            )

        # 7. 结束事件
        done_event = {
            "session_id": session_id,
            "delta": "",
            "finish": True,
        }
        yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

    # ------------------------------------------------------------------
    # 非流式问答
    # ------------------------------------------------------------------

    async def chat(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
    ) -> dict:
        """非流式 RAG 问答，返回完整结果。"""
        k = top_k or self._top_k
        chunks = self._retrieve_context(question, k)
        messages = self._build_messages(question, chunks, session_id)

        # 保存用户问题
        self._session.add_message(session_id, "user", question)

        # 调用 LLM（非流式）
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content

        # 保存助手回答
        self._session.add_message(
            session_id,
            "assistant",
            answer,
            retrieved_chunks=[
                {
                    "doc_id": c["doc_id"],
                    "doc_title": c.get("doc_title"),
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "score": c["score"],
                    "retrieval_mode": c.get("retrieval_mode"),
                    "source": c.get("source"),
                    "rank": c.get("rank"),
                }
                for c in chunks
            ],
        )

        return {
            "session_id": session_id,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "doc_id": c["doc_id"],
                    "doc_title": c.get("doc_title"),
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "score": c["score"],
                    "retrieval_mode": c.get("retrieval_mode"),
                    "source": c.get("source"),
                    "rank": c.get("rank"),
                }
                for c in chunks
            ],
            "created_at": time.time(),
        }
