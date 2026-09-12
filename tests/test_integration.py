"""
集成测试 —— 需要真实的 Redis Stack 和 LLM API。
默认跳过，设置环境变量 RUN_INTEGRATION=1 后运行。

运行方式:
  RUN_INTEGRATION=1 pytest tests/test_integration.py -v -s
"""
import os
import time

import pytest

# 默认跳过，除非显式开启
pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"),
    reason="集成测试需要真实 Redis + LLM，设置 RUN_INTEGRATION=1 启用",
)


@pytest.fixture(scope="module")
def sample_doc():
    """测试用文档"""
    return """# Redis 向量检索指南

## 什么是 RediSearch

RediSearch 是 Redis Stack 中的搜索与查询模块，支持全文检索、
数值过滤、聚合查询以及向量相似度搜索。它通过 FT.CREATE 创建索引，
使用 FT.SEARCH 进行查询。

## 向量字段类型

RediSearch 支持两种向量索引算法：
- FLAT：暴力检索，适合小规模数据集（< 10 万向量）
- HNSW：分层导航小世界图，适合大规模数据集，查询更快但精度略低

向量距离度量支持：COSINE（余弦相似度）、L2（欧氏距离）、IP（内积）

## 创建向量索引示例

使用 FT.CREATE 命令创建索引：
1. 定义 TEXT 字段存储文本内容
2. 定义 VECTOR 字段存储嵌入向量
3. 指定向量维度、距离度量和索引算法

## KNN 查询

使用 FT.SEARCH 配合 KNN 语法进行向量相似度检索：
`FT.SEARCH idx (*)=>[KNN 5 @embedding $query_vec]`
其中 5 是返回的最近邻数量，$query_vec 是参数化查询向量。

## 性能建议

- 批量写入时使用 pipeline 减少 RTT
- 向量维度建议 256-1536 之间
- HNSW 的 M 参数建议 16-48
- 使用 COSINE 度量时向量会自动归一化
"""


class TestDocumentImportIntegration:

    def test_full_import_and_retrieve(self, sample_doc):
        """完整流程：导入 → 检索"""
        from app.services.document_service import DocumentChunker
        from app.services.embedding_service import EmbeddingService
        from app.services.vector_store import VectorStore

        # 1. 切片
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.split(sample_doc, content_type="markdown")
        assert len(chunks) >= 3
        print(f"\n切片数: {len(chunks)}")
        for c in chunks:
            print(f"  [{c.chunk_index}] {c.heading_path} | {c.content[:50]}...")

        # 2. 向量化
        embedder = EmbeddingService()
        embeddings = embedder.embed_batch([c.content for c in chunks])
        assert len(embeddings) == len(chunks)

        # 3. 存储
        store = VectorStore()
        store.ensure_index()
        doc_id = f"integration_test_{int(time.time())}"
        chunk_dicts = [
            {
                "content": c.content,
                "chunk_index": c.chunk_index,
                "heading_path": c.heading_path,
            }
            for c in chunks
        ]
        stored = store.store_chunks(doc_id, chunk_dicts, embeddings)
        assert stored == len(chunks)
        print(f"存储成功: {stored} 个片段, doc_id={doc_id}")

        # 4. 检索
        query = "RediSearch 支持哪些向量索引算法？"
        query_vec = embedder.embed(query)
        results = store.search(query_vec, top_k=3)
        assert len(results) > 0
        print(f"\n检索结果 (query='{query}'):")
        for r in results:
            print(f"  score={r['score']:.4f} doc={r['doc_id']} idx={r['chunk_index']} | {r['content'][:60]}...")

        # 5. 清理
        store.delete_doc(doc_id)
        print(f"已清理 doc_id={doc_id}")


class TestChatIntegration:

    def test_streaming_chat(self, sample_doc):
        """测试流式问答"""
        import asyncio
        from app.services.chat_service import ChatService
        from app.services.document_service import DocumentChunker
        from app.services.embedding_service import EmbeddingService
        from app.services.vector_store import VectorStore
        from app.services.session_service import SessionService

        # 先导入文档
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.split(sample_doc, content_type="markdown")
        embedder = EmbeddingService()
        embeddings = embedder.embed_batch([c.content for c in chunks])
        store = VectorStore()
        store.ensure_index()
        doc_id = f"chat_test_{int(time.time())}"
        chunk_dicts = [
            {"content": c.content, "chunk_index": c.chunk_index, "heading_path": c.heading_path}
            for c in chunks
        ]
        store.store_chunks(doc_id, chunk_dicts, embeddings)

        # 流式问答
        session_id = f"test_session_{int(time.time())}"
        chat = ChatService()
        question = "RediSearch 的 FLAT 和 HNSW 有什么区别？"

        async def run():
            full_answer = []
            retrieved = None
            async for sse in chat.stream_chat(session_id, question):
                # 解析 SSE
                data = sse.replace("data: ", "").strip()
                if data:
                    event = __import__("json").loads(data)
                    if event.get("retrieved_chunks"):
                        retrieved = event["retrieved_chunks"]
                    if event.get("delta"):
                        full_answer.append(event["delta"])
                    if event.get("finish"):
                        break
            return "".join(full_answer), retrieved

        answer, retrieved = asyncio.run(run())
        print(f"\n问题: {question}")
        print(f"检索片段数: {len(retrieved) if retrieved else 0}")
        print(f"回答: {answer}")
        assert len(answer) > 0

        # 验证历史记录
        session_svc = SessionService()
        history = session_svc.get_history(session_id)
        assert len(history) >= 2  # user + assistant
        print(f"\n会话历史: {len(history)} 条消息")

        # 清理
        session_svc.clear_session(session_id)
        store.delete_doc(doc_id)
        print("清理完成")
