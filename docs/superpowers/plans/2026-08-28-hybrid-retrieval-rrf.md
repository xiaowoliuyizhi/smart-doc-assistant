# Hybrid Retrieval with RRF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add speed-first BM25 plus vector retrieval with RRF while retaining vector-only Recall@k baseline.

**Architecture:** Redis Stack remains the only datastore. `VectorStore` adds BM25 retrieval; `RetrievalService` selects vector or hybrid from settings. Hybrid takes ten results per route, fuses by chunk identity, and returns four results.

**Tech Stack:** Python, FastAPI, redis-py/RediSearch, Pydantic Settings, pytest, browser JavaScript.

**Spec:** `docs/superpowers/specs/2026-08-28-hybrid-retrieval-rrf-design.md`

## Global Constraints

- Support only `vector` and `hybrid` modes; retain vector behavior unchanged.
- Default to `TOP_K=4`, `VECTOR_CANDIDATE_K=10`, `BM25_CANDIDATE_K=10`, and `RRF_K=60`.
- De-duplicate using `{doc_id}:{chunk_index}`; never display a raw RRF value as a percent.
- Do not add reranking, a second datastore, online experiments, or Chinese segmentation dependencies.

---

### Task 1: Configure retrieval modes and metadata

**Files:**
- Modify: `app/config.py:17-37`
- Modify: `app/models/schemas.py:60-71,112-120`
- Modify: `tests/test_schemas.py`

**Interfaces:** Produces `Settings.retrieval_mode: Literal["vector", "hybrid"]`, `vector_candidate_k`, `bm25_candidate_k`, and `rrf_k`; produces optional `RetrievedChunk.retrieval_mode`, `source`, and `rank`.

- [ ] **Step 1: Write failing tests**

```python
def test_retrieved_chunk_accepts_hybrid_metadata():
    item = RetrievedChunk(doc_id="d", doc_title="a.pdf", chunk_index=1,
        content="x", score=0.0, retrieval_mode="hybrid", source="both", rank=1)
    assert (item.source, item.rank) == ("both", 1)

def test_invalid_retrieval_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "wrong")
    get_settings.cache_clear()
    with pytest.raises(ValidationError): get_settings()
```

- [ ] **Step 2: Verify RED**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_schemas.py -q`

Expected: FAIL because the fields and validation are absent.

- [ ] **Step 3: Implement the fields**

```python
retrieval_mode: Literal["vector", "hybrid"] = "vector"
vector_candidate_k: int = 10
bm25_candidate_k: int = 10
rrf_k: int = 60
```

Add nullable `retrieval_mode`, `source`, and `rank` to `RetrievedChunk` using the literals above.

- [ ] **Step 4: Verify GREEN**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_schemas.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add app/config.py app/models/schemas.py tests/test_schemas.py`

Run: `git commit -m "feat: add retrieval mode configuration"`

### Task 2: Add safe Redis BM25 search

**Files:**
- Modify: `app/services/vector_store.py:126-180`
- Modify: `tests/test_vector_store.py`

**Interfaces:** Produces `search_text(query: str, top_k: int = 10) -> list[dict[str, Any]]`, returning the same fields as KNN plus `doc_title` in BM25 rank order.

- [ ] **Step 1: Write failing tests**

```python
def test_search_text_returns_chunk_with_title():
    redis_client.ft.return_value.search.return_value = SimpleNamespace(docs=[document])
    redis_client.hget.return_value = "manual.pdf"
    assert store.search_text("CBAM attention", 10)[0]["doc_title"] == "manual.pdf"

def test_search_text_escapes_query_operators():
    store.search_text("C++ (v2) | test", 10)
    query = redis_client.ft.return_value.search.call_args.args[0].query_string
    assert "\\|" in query
```

- [ ] **Step 2: Verify RED**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_vector_store.py -q`

Expected: FAIL because `search_text` is missing.

- [ ] **Step 3: Implement `search_text`**

```python
def search_text(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    safe_query = _escape_redis_text_query(query)
    q = Query(f"@content:({safe_query})").paging(0, top_k).with_scores().dialect(2)
    return self._decode_search_results(self._redis.ft(self._index_name).search(q))
```

Add a fixed Redis query-operator escape map. Extract shared title-caching/result-decoding from KNN search so both methods have identical chunk structures.

- [ ] **Step 4: Verify GREEN**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_vector_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add app/services/vector_store.py tests/test_vector_store.py`

Run: `git commit -m "feat: add Redis BM25 chunk search"`

### Task 3: Orchestrate vector/hybrid retrieval and RRF

**Files:**
- Modify: `app/services/retrieval_service.py:12-36`
- Create: `tests/test_retrieval_service.py`

**Interfaces:** Produces `retrieve_vector(query, candidate_k=None)`, `retrieve_hybrid(query, candidate_k=None)`, and mode-selecting `retrieve(query, top_k)`.

- [ ] **Step 1: Write failing tests**

```python
def test_hybrid_rrf_boosts_chunk_found_by_both():
    store.search.return_value = [chunk("a", 0), chunk("b", 0)]
    store.search_text.return_value = [chunk("b", 0), chunk("c", 0)]
    results = service.retrieve_hybrid("question", candidate_k=2)
    assert [x["doc_id"] for x in results] == ["b", "a", "c"]
    assert results[0]["source"] == "both"

def test_hybrid_falls_back_when_bm25_fails():
    store.search.return_value = [chunk("a", 0)]
    store.search_text.side_effect = redis.RedisError("unavailable")
    assert service.retrieve_hybrid("question")[0]["source"] == "vector"
```

- [ ] **Step 2: Verify RED**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_retrieval_service.py -q`

Expected: FAIL because hybrid methods are absent.

- [ ] **Step 3: Implement RRF**

```python
def _rrf_merge(vector_results, bm25_results, rrf_k):
    merged = {}
    for source, results in (("vector", vector_results), ("bm25", bm25_results)):
        for rank, chunk in enumerate(results, start=1):
            key = f"{chunk['doc_id']}:{chunk['chunk_index']}"
            entry = merged.setdefault(key, {
                "chunk": chunk.copy(), "rrf_score": 0.0,
                "vector_rank": float("inf"), "sources": set(),
            })
            entry["rrf_score"] += 1 / (rrf_k + rank)
            entry["sources"].add(source)
            if source == "vector": entry["vector_rank"] = rank
    return sorted(merged.values(), key=lambda entry: (
        -entry["rrf_score"], entry["vector_rank"],
        entry["chunk"]["doc_id"], entry["chunk"]["chunk_index"],
    ))
```

Annotate vector results with original score, mode, source, rank. Annotate hybrid results with mode, source (`vector`, `bm25`, `both`), and rank. Catch BM25 Redis errors only and then use KNN candidates.

- [ ] **Step 4: Verify GREEN**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_retrieval_service.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add app/services/retrieval_service.py tests/test_retrieval_service.py`

Run: `git commit -m "feat: add hybrid RRF retrieval mode"`

### Task 4: Propagate and render retrieval metadata

**Files:**
- Modify: `app/services/chat_service.py:138-257`
- Modify: `static/index.html:321-330,1013-1029`
- Modify: `tests/test_chat_service.py`
- Modify: `tests/test_api.py`

**Interfaces:** Stream, non-stream and saved history retain mode, source, rank, title and score. `appendSources` renders vector similarity or hybrid rank.

- [ ] **Step 1: Write failing tests**

```python
def test_non_stream_chat_preserves_hybrid_metadata():
    retrieval.retrieve.return_value = [chunk("doc-1", 2, retrieval_mode="hybrid", source="both", rank=1)]
    result = asyncio.run(service.chat("s1", "question"))
    assert result["retrieved_chunks"][0]["source"] == "both"
```

- [ ] **Step 2: Verify RED**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_chat_service.py tests/test_api.py -q`

Expected: FAIL because current serialization drops the metadata.

- [ ] **Step 3: Implement serialization and labels**

```javascript
const label = s.retrieval_mode === 'hybrid'
  ? `综合排名 #${s.rank}`
  : `语义相似度 ${(s.score * 100).toFixed(0)}%`;
```

Use one ChatService serializer for stream, non-stream, and persisted history. Keep the existing `s.doc_title || \`文档 ${s.doc_id}\`` fallback.

- [ ] **Step 4: Verify GREEN**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_chat_service.py tests/test_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add app/services/chat_service.py static/index.html tests/test_chat_service.py tests/test_api.py`

Run: `git commit -m "feat: show hybrid retrieval sources"`

### Task 5: Document and verify both modes

**Files:**
- Modify: `.env.example`, `README.md`, `docs/ARCHITECTURE.md`, `docs/API.md`
- Create: `tests/test_config_docs.py`

- [ ] **Step 1: Write a failing documentation test**

```python
def test_env_example_lists_hybrid_settings():
    content = Path(".env.example").read_text(encoding="utf-8")
    assert "RETRIEVAL_MODE=vector" in content
    assert "RRF_K=60" in content
```

- [ ] **Step 2: Verify RED**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_config_docs.py -q`

Expected: FAIL because hybrid settings are absent.

- [ ] **Step 3: Document settings and API fields**

Document all five settings, both flows, restart requirement, and `retrieval_mode`, `source`, `rank` source labels. State that evaluation calls `retrieve_vector` and `retrieve_hybrid` directly for Recall@k.

- [ ] **Step 4: Verify all tests and live modes**

Run: `& .venv\Scripts\python.exe -m pytest tests -q`

Expected: PASS, with integration tests skipped unless `RUN_INTEGRATION=1`.

Start once with `RETRIEVAL_MODE=vector` and once with `RETRIEVAL_MODE=hybrid`; the same question must show `语义相似度` then `综合排名`.

- [ ] **Step 5: Commit**

Run: `git add .env.example README.md docs/ARCHITECTURE.md docs/API.md tests/test_config_docs.py`

Run: `git commit -m "docs: describe hybrid RRF retrieval"`
