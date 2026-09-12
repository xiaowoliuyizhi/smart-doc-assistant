from unittest.mock import MagicMock

import redis

from app.services.retrieval_service import RetrievalService


def chunk(doc_id):
    return {"doc_id": doc_id, "chunk_index": 0, "content": doc_id, "score": 0.9}


def test_hybrid_rrf_boosts_chunk_found_by_both():
    embedder = MagicMock(); embedder.embed.return_value = [0.1]
    store = MagicMock()
    store.search.return_value = [chunk("a"), chunk("b")]
    store.search_text.return_value = [chunk("b"), chunk("c")]
    service = RetrievalService(embedder, store)
    results = service.retrieve_hybrid("question", candidate_k=2)
    assert [item["doc_id"] for item in results] == ["b", "a", "c"]
    assert results[0]["source"] == "both"


def test_hybrid_falls_back_to_vector_when_bm25_fails():
    embedder = MagicMock(); embedder.embed.return_value = [0.1]
    store = MagicMock(); store.search.return_value = [chunk("a")]
    store.search_text.side_effect = redis.RedisError("unavailable")
    result = RetrievalService(embedder, store).retrieve_hybrid("question")
    assert result[0]["source"] == "vector"
