"""VectorStore 检索结果来源信息测试。"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.vector_store import VectorStore


def test_search_includes_document_title_from_document_metadata():
    """检索片段应带上所属文档标题，而不是只返回内部 ID。"""
    document = SimpleNamespace(
        distance="0.2",
        content="CBAM 介绍",
        doc_id="doc-1",
        chunk_index="2",
        heading_path="模块说明",
    )
    with patch("app.services.vector_store.redis") as redis_module:
        redis_client = MagicMock()
        redis_client.ft.return_value.search.return_value = SimpleNamespace(docs=[document])
        redis_client.hget.return_value = "CBAM模块论文.pdf"
        redis_module.from_url.return_value = redis_client
        store = VectorStore()

        results = store.search([0.1] * store._dim)

    assert results[0]["doc_title"] == "CBAM模块论文.pdf"


def test_search_text_returns_bm25_chunk_with_document_title():
    """BM25 检索应返回与 KNN 相同的来源信息。"""
    document = SimpleNamespace(
        score="2.3",
        content="CBAM 介绍",
        doc_id="doc-1",
        chunk_index="2",
        heading_path="模块说明",
    )
    with patch("app.services.vector_store.redis") as redis_module:
        redis_client = MagicMock()
        redis_client.ft.return_value.search.return_value = SimpleNamespace(docs=[document])
        redis_client.hget.return_value = "CBAM模块论文.pdf"
        redis_module.from_url.return_value = redis_client
        store = VectorStore()

        results = store.search_text("CBAM attention", top_k=10)

    assert results[0]["doc_title"] == "CBAM模块论文.pdf"
    assert results[0]["score"] == 2.3


def test_search_text_escapes_redis_query_operators():
    """用户输入的 RediSearch 语法字符应按普通文本查询。"""
    with patch("app.services.vector_store.redis") as redis_module:
        redis_client = MagicMock()
        redis_client.ft.return_value.search.return_value = SimpleNamespace(docs=[])
        redis_module.from_url.return_value = redis_client
        store = VectorStore()

        store.search_text("C++ (v2) | test", top_k=10)

    query = redis_client.ft.return_value.search.call_args.args[0].query_string()
    assert "C\\+\\+" in query
    assert "\\|" in query
