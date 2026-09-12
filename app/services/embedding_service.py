"""
Embedding 服务 —— 调用 OpenAI 兼容接口生成文本向量。

支持：
  - 单条文本向量化
  - 批量向量化（减少 API 往返）
"""
from __future__ import annotations

from openai import OpenAI

from app.config import get_settings


class EmbeddingService:
    """文本向量化服务"""

    def __init__(self):
        settings = get_settings()
        api_base = settings.embedding_api_base or settings.llm_api_base
        api_key = settings.embedding_api_key or settings.llm_api_key
        self._client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        """单条文本 → 向量"""
        resp = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表"""
        if not texts:
            return []
        # OpenAI 单次最多 2048 条，这里按 100 分批
        results: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            # 响应中 data 按 index 排序
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            results.extend(d.embedding for d in sorted_data)
        return results
