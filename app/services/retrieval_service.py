"""
检索服务 —— 将 Embedding + VectorSearch 组合为一步：文本 → 相关片段。
"""
from __future__ import annotations

from typing import Any
import redis

from app.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class RetrievalService:
    """语义检索服务"""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ):
        self._embedder = embedding_service or EmbeddingService()
        self._store = vector_store or VectorStore()
        self._settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """根据用户问题检索最相关的 N 个片段。

        Args:
            query: 用户问题
            top_k: 返回的片段数

        Returns:
            [{doc_id, chunk_index, content, heading_path, score}, ...]
        """
        if self._settings.retrieval_mode == "hybrid":
            return self.retrieve_hybrid(query, self._settings.vector_candidate_k)[:top_k]
        return self.retrieve_vector(query, top_k)

    def retrieve_vector(self, query: str, candidate_k: int | None = None) -> list[dict[str, Any]]:
        results = self._store.search(self._embedder.embed(query), candidate_k or self._settings.vector_candidate_k)
        return [dict(item, retrieval_mode="vector", source="vector", rank=index) for index, item in enumerate(results, 1)]

    def retrieve_hybrid(self, query: str, candidate_k: int | None = None) -> list[dict[str, Any]]:
        limit = candidate_k or self._settings.vector_candidate_k
        vector = self._store.search(self._embedder.embed(query), limit)
        try:
            bm25 = self._store.search_text(query, self._settings.bm25_candidate_k)
        except redis.RedisError:
            bm25 = []
        return self._rrf_merge(vector, bm25)

    def _rrf_merge(self, vector: list[dict], bm25: list[dict]) -> list[dict[str, Any]]:
        merged: dict[str, dict] = {}
        for source, items in (("vector", vector), ("bm25", bm25)):
            for rank, item in enumerate(items, 1):
                key = f"{item['doc_id']}:{item['chunk_index']}"
                entry = merged.setdefault(key, {"item": item.copy(), "score": 0.0, "vector_rank": float("inf"), "sources": set()})
                entry["score"] += 1 / (self._settings.rrf_k + rank)
                entry["sources"].add(source)
                if source == "vector": entry["vector_rank"] = rank
        ordered = sorted(merged.values(), key=lambda entry: (-entry["score"], entry["vector_rank"], entry["item"]["doc_id"], entry["item"]["chunk_index"]))
        return [dict(entry["item"], retrieval_mode="hybrid", source="both" if len(entry["sources"]) == 2 else next(iter(entry["sources"])), rank=index) for index, entry in enumerate(ordered, 1)]
