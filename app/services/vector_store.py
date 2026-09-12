"""
Redis 向量存储服务 —— 使用 RediSearch (FT) 模块实现向量持久化与相似度检索。

存储结构：
  - Key:  doc:chunk:{doc_id}:{chunk_index}
  - Type: Redis Hash
  - Fields:
      content       — TEXT，片段原文
      doc_id        — TAG，所属文档 ID
      chunk_index   — NUMERIC，片段序号
      heading_path  — TEXT，Markdown 标题路径
      embedding     — VECTOR (FLAT, DIM=1536, DISTANCE_METRIC=COSINE)

索引：
  - Name:  idx:doc_chunks
  - 使用 FT.CREATE 创建，FT.SEARCH 做 KNN 查询
"""
from __future__ import annotations

import json
import re
import struct
import time
from typing import Any

import redis

from app.config import get_settings


def _vector_to_bytes(vec: list[float]) -> bytes:
    """float32 列表 → little-endian bytes（RediSearch VECTOR 字段要求）"""
    return struct.pack(f"<{len(vec)}f", *vec)


def _bytes_to_vector(b: bytes) -> list[float]:
    n = len(b) // 4
    return list(struct.unpack(f"<{n}f", b))


def _escape_redis_text_query(query: str) -> str:
    """将用户输入作为 RediSearch TEXT 查询文本，而不是查询语法。"""
    normalized = re.sub(r"[\x00-\x1f\x7f]", " ", query).strip()
    return re.sub(r'([@{}\[\]()|+\-~*"\\\\:])', r"\\\1", normalized)


class VectorStore:
    """Redis 向量存储"""

    def __init__(self):
        settings = get_settings()
        self._redis = redis.from_url(
            settings.redis_url, decode_responses=False
        )
        self._index_name = settings.redis_index_name
        self._dim = settings.embedding_dim
        self._prefix = "doc:chunk:"

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def ensure_index(self) -> None:
        """创建 RediSearch 索引（若不存在）。"""
        try:
            self._redis.ft(self._index_name).info()
            return  # 已存在
        except Exception:
            pass  # 不存在，继续创建

        from redis.commands.search.field import (
            TextField,
            NumericField,
            TagField,
            VectorField,
        )
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        schema = (
            TextField("content"),
            TagField("doc_id"),
            NumericField("chunk_index"),
            TextField("heading_path"),
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self._dim,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(
            prefix=[self._prefix], index_type=IndexType.HASH
        )
        self._redis.ft(self._index_name).create_index(
            fields=schema, definition=definition
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def store_chunks(
        self,
        doc_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """批量写入片段及其向量。

        Args:
            doc_id: 文档 ID
            chunks: [{content, chunk_index, heading_path}, ...]
            embeddings: 与 chunks 等长的向量列表

        Returns:
            写入的片段数
        """
        self.ensure_index()
        pipe = self._redis.pipeline(transaction=False)
        for chunk, emb in zip(chunks, embeddings):
            key = f"{self._prefix}{doc_id}:{chunk['chunk_index']}"
            mapping = {
                "content": chunk["content"],
                "doc_id": doc_id,
                "chunk_index": chunk["chunk_index"],
                "heading_path": chunk.get("heading_path", ""),
                "embedding": _vector_to_bytes(emb),
            }
            pipe.hset(key, mapping=mapping)
        pipe.execute()
        return len(chunks)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """向量相似度检索（KNN）。

        Returns:
            [{doc_id, chunk_index, content, heading_path, score}, ...]
            score 越高越相关（已将 distance 转换为 similarity）。
        """
        from redis.commands.search.query import Query

        query_bytes = _vector_to_bytes(query_vector)

        q = (
            Query(
                f"(*)=>[KNN {top_k} @embedding $query_vec AS distance]"
            )
            .return_fields(
                "content", "doc_id", "chunk_index", "heading_path", "distance"
            )
            .dialect(2)
        )

        results = self._redis.ft(self._index_name).search(
            q,
            query_params={"query_vec": query_bytes},
        )

        return self._decode_search_results(results, score_field="distance")

    def search_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """使用 RediSearch 的 BM25 全文检索返回候选片段。"""
        from redis.commands.search.query import Query

        safe_query = _escape_redis_text_query(query)
        if not safe_query:
            return []

        q = (
            Query(f"@content:({safe_query})")
            .return_fields("content", "doc_id", "chunk_index", "heading_path")
            .paging(0, top_k)
            .with_scores()
            .dialect(2)
        )
        results = self._redis.ft(self._index_name).search(q)
        return self._decode_search_results(results, score_field="score")

    def _decode_search_results(
        self,
        results: Any,
        score_field: str,
    ) -> list[dict[str, Any]]:
        """将 RediSearch 结果统一转换为检索片段字典。"""
        items: list[dict[str, Any]] = []
        titles_by_doc_id: dict[str, str] = {}
        for doc in results.docs:
            raw_score = float(getattr(doc, score_field))
            score = 1.0 - (raw_score / 2.0) if score_field == "distance" else raw_score
            content = doc.content
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            heading = doc.heading_path
            if isinstance(heading, bytes):
                heading = heading.decode("utf-8")
            doc_id = doc.doc_id.decode() if isinstance(doc.doc_id, bytes) else doc.doc_id
            if doc_id not in titles_by_doc_id:
                title = self._redis.hget(f"doc:meta:{doc_id}", "title")
                if isinstance(title, bytes):
                    title = title.decode("utf-8")
                titles_by_doc_id[doc_id] = title or doc_id
            items.append(
                {
                    "doc_id": doc_id,
                    "doc_title": titles_by_doc_id[doc_id],
                    "chunk_index": int(doc.chunk_index),
                    "content": content,
                    "heading_path": heading,
                    "score": round(score, 4),
                }
            )
        return items

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------

    def set_doc_meta(
        self,
        doc_id: str,
        title: str,
        chunk_count: int,
        total_chars: int,
        content_type: str,
    ) -> None:
        """记录文档元信息"""
        self._redis.hset(
            f"doc:meta:{doc_id}",
            mapping={
                "title": title or "",
                "chunk_count": chunk_count,
                "total_chars": total_chars,
                "content_type": content_type,
                "imported_at": time.time(),
            },
        )

    def list_docs(self) -> list[dict]:
        """列出所有已导入的文档"""
        import time as _time
        docs = []
        for key in self._redis.scan_iter("doc:meta:*"):
            doc_id = key.decode() if isinstance(key, bytes) else key
            doc_id = doc_id.removeprefix("doc:meta:")
            meta = self._redis.hgetall(key)
            dec = {}
            for k, v in meta.items():
                kk = k.decode() if isinstance(k, bytes) else k
                vv = v.decode() if isinstance(v, bytes) else v
                dec[kk] = vv
            docs.append({
                "doc_id": doc_id,
                "title": dec.get("title", ""),
                "chunk_count": int(dec.get("chunk_count", 0)),
                "total_chars": int(dec.get("total_chars", 0)),
                "content_type": dec.get("content_type", "text"),
                "imported_at": float(dec.get("imported_at", 0)),
            })
        docs.sort(key=lambda d: d["imported_at"], reverse=True)
        return docs

    def delete_doc(self, doc_id: str) -> int:
        """删除指定文档的所有片段及元信息"""
        self._redis.delete(f"doc:meta:{doc_id}")
        keys = self._redis.keys(f"{self._prefix}{doc_id}:*")
        if keys:
            return self._redis.delete(*keys)
        return 0

    def count_chunks(self) -> int:
        """统计索引中的片段总数"""
        try:
            info = self._redis.ft(self._index_name).info()
            return int(info.get("num_docs", 0))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # 连接检查
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            return self._redis.ping()
        except Exception:
            return False
