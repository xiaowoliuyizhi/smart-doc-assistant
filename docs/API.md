# API 接口文档 — 智能文档助手

> Base URL: `http://localhost:8000`
>
> 所有接口返回 JSON 格式（流式接口除外），字段命名使用 `snake_case`。

---

## 接口总览

| # | 方法 | 路径 | 功能 |
|---|------|------|------|
| 1 | POST | `/api/v1/document/import` | 提交文档内容 |
| 2 | POST | `/api/v1/chat/completions` | 发起对话（支持流式） |
| 3 | GET | `/api/v1/chat/history/{sessionId}` | 查看会话历史 |
| 4 | DELETE | `/api/v1/chat/session/{sessionId}` | 清除会话 |

---

## 1. POST /api/v1/document/import

提交文档内容，系统自动切片、向量化并持久化存储到 Redis。

### 请求

```json
POST /api/v1/document/import
Content-Type: application/json

{
  "content": "Redis 是一个高性能的键值数据库...(文档正文)",
  "title": "Redis 入门指南",
  "doc_id": "doc_redis_001",
  "content_type": "markdown"
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `content` | string | ✅ | — | 文档正文（Markdown 或纯文本），不可为空 |
| `title` | string \| null | ❌ | null | 文档标题，用于溯源展示 |
| `doc_id` | string \| null | ❌ | 自动生成 | 自定义文档 ID，格式 `doc_{uuid_hex}` |
| `content_type` | `"markdown"` \| `"text"` | ❌ | `"text"` | 内容格式，影响切片策略 |

### 响应 (200 OK)

```json
{
  "doc_id": "doc_redis_001",
  "title": "Redis 入门指南",
  "chunk_count": 12,
  "total_chars": 5234,
  "imported_at": 1722521270.123
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_id` | string | 文档唯一标识 |
| `title` | string \| null | 文档标题 |
| `chunk_count` | integer | 切分后的片段总数 |
| `total_chars` | integer | 原文总字符数 |
| `imported_at` | number | 导入时间戳 (Unix) |

### 错误响应

| 状态码 | 说明 |
|--------|------|
| 400 | 文档内容为空或切片结果为空 |
| 502 | 向量化服务调用失败 |
| 500 | Redis 存储失败 |

---

## 2. POST /api/v1/chat/completions

基于 RAG 发起对话，支持流式 (SSE) 和非流式两种模式。

### 请求

```json
POST /api/v1/chat/completions
Content-Type: application/json

{
  "session_id": "user-session-001",
  "question": "RediSearch 支持哪些向量索引算法？",
  "stream": true,
  "top_k": 4
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `session_id` | string | ✅ | — | 会话唯一标识，用于多轮对话 |
| `question` | string | ✅ | — | 用户提问内容 |
| `stream` | boolean | ❌ | true | 是否流式返回 |
| `top_k` | integer \| null | ❌ | 全局默认 (4) | 检索片段数 |

### 响应 — 流式模式 (stream=true)

**Content-Type:** `text/event-stream`

以 SSE 格式逐条推送，每条格式为 `data: {json}\n\n`：

```
data: {"session_id":"user-session-001","delta":"","retrieved_chunks":[{"doc_id":"doc_1","chunk_index":0,"content":"...","score":0.95}],"finish":false}

data: {"session_id":"user-session-001","delta":"RediSearch","finish":false}

data: {"session_id":"user-session-001","delta":" 支持","finish":false}

data: {"session_id":"user-session-001","delta":" FLAT 和 HNSW 两种算法。","finish":false}

data: {"session_id":"user-session-001","delta":"","finish":true}
```

**SSE 事件结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |
| `delta` | string | 本次增量文本（首条和末条可能为空） |
| `retrieved_chunks` | array \| null | 检索到的片段列表，**仅首条事件携带** |
| `finish` | boolean | 是否为最后一条事件 |

**retrieved_chunks 元素结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_id` | string | 来源文档 ID |
| `chunk_index` | integer | 片段序号 |
| `content` | string | 片段原文 |
| `score` | number | 相似度得分 [0, 1]，越高越相关 |

### 响应 — 非流式模式 (stream=false)

```json
{
  "session_id": "user-session-001",
  "answer": "RediSearch 支持 FLAT 和 HNSW 两种向量索引算法。FLAT 适合小规模数据集...",
  "retrieved_chunks": [
    {
      "doc_id": "doc_redis_001",
      "chunk_index": 3,
      "content": "RediSearch 支持两种向量索引算法：FLAT 和 HNSW...",
      "score": 0.95
    },
    {
      "doc_id": "doc_redis_001",
      "chunk_index": 4,
      "content": "HNSW 是分层导航小世界图算法...",
      "score": 0.88
    }
  ],
  "created_at": 1722521270.456
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |
| `answer` | string | 完整回答文本 |
| `retrieved_chunks` | array | 检索到的片段列表（同上） |
| `created_at` | number | 生成时间戳 (Unix) |

### 错误响应

| 状态码 | 说明 |
|--------|------|
| 422 | 请求参数校验失败（缺少必填字段） |
| 502 | LLM 服务调用失败 |

---

## 3. GET /api/v1/chat/history/{sessionId}

查看指定会话的上下文记录（对话历史）。

### 请求

```
GET /api/v1/chat/history/user-session-001
```

| 路径参数 | 类型 | 说明 |
|----------|------|------|
| `sessionId` | string | 会话唯一标识 |

### 响应 (200 OK)

```json
{
  "session_id": "user-session-001",
  "messages": [
    {
      "role": "user",
      "content": "RediSearch 支持哪些向量索引算法？",
      "timestamp": 1722521260.123
    },
    {
      "role": "assistant",
      "content": "RediSearch 支持 FLAT 和 HNSW 两种向量索引算法...",
      "timestamp": 1722521265.456
    },
    {
      "role": "user",
      "content": "FLAT 和 HNSW 有什么区别？",
      "timestamp": 1722521270.789
    },
    {
      "role": "assistant",
      "content": "FLAT 采用暴力检索...",
      "timestamp": 1722521275.012
    }
  ],
  "message_count": 4
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |
| `messages` | array | 消息列表（按时间正序排列） |
| `message_count` | integer | 消息总数 |

**messages 元素结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `role` | `"user"` \| `"assistant"` | 消息角色 |
| `content` | string | 消息内容 |
| `timestamp` | number | 消息时间戳 (Unix) |

---

## 4. DELETE /api/v1/chat/session/{sessionId}

清除指定会话的全部历史记录。

### 请求

```
DELETE /api/v1/chat/session/user-session-001
```

| 路径参数 | 类型 | 说明 |
|----------|------|------|
| `sessionId` | string | 要清除的会话 ID |

### 响应 (200 OK)

```json
{
  "session_id": "user-session-001",
  "deleted": true,
  "message": "会话历史已清除"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |
| `deleted` | boolean | 是否成功删除 |
| `message` | string | 操作结果描述 |

### 错误响应

| 状态码 | 说明 |
|--------|------|
| 404 | 会话不存在或已为空 |

---

## 辅助接口

### GET /health

健康检查，返回服务与 Redis 连接状态。

```json
{
  "status": "ok",
  "redis": "connected",
  "chunk_count": 42
}
```

### GET /

根路由，返回服务基本信息。

```json
{
  "service": "Smart Doc Assistant",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

## 完整调用示例

```bash
# 1. 导入文档
curl -X POST http://localhost:8000/api/v1/document/import \
  -H "Content-Type: application/json" \
  -d '{"content": "# Redis 指南\n\nRedis 是高性能键值数据库...", "content_type": "markdown", "title": "Redis 指南"}'

# 2. 流式对话
curl -N http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-1", "question": "Redis 是什么？", "stream": true}'

# 3. 查看历史
curl http://localhost:8000/api/v1/chat/history/sess-1

# 4. 清除会话
curl -X DELETE http://localhost:8000/api/v1/chat/session/sess-1
```
