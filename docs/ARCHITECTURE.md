# 架构设计说明 — 智能文档助手 (Smart Doc Assistant)

## 1. 系统总览

本系统是一个基于 **RAG (Retrieval-Augmented Generation)** 架构的智能文档问答服务，核心链路为：

```
用户上传文档 → 切片(Chunking) → 向量化(Embedding) → 存入 Redis
                                                         ↓
用户提问 → 问题向量化 → Redis 向量检索(Top-K) → 组装 Prompt → LLM 流式生成 → 返回
                                                         ↓
                                                   会话历史(Redis List)
```

### 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Web 框架 | FastAPI | 异步 API、SSE 流式响应 |
| 向量数据库 | Redis Stack (RediSearch) | 向量存储 + KNN 相似度检索 |
| Embedding | OpenAI 兼容 API | 文本向量化 |
| LLM | OpenAI 兼容 API | 生成式问答 |
| 会话存储 | Redis (List) | 多轮对话历史持久化 |

### 项目结构

```
smart-doc-assistant/
├── app/
│   ├── main.py                  # FastAPI 入口 + 生命周期管理
│   ├── config.py                # 环境变量配置
│   ├── models/
│   │   └── schemas.py           # 请求/响应数据模型 (Pydantic)
│   ├── services/
│   │   ├── document_service.py  # 文档切片器
│   │   ├── embedding_service.py # 向量化服务
│   │   ├── vector_store.py      # Redis 向量存储 + 检索
│   │   ├── retrieval_service.py # 检索编排
│   │   ├── chat_service.py      # RAG 问答 + 流式生成
│   │   └── session_service.py   # 会话历史管理
│   └── routers/
│       ├── document.py          # POST /api/v1/document/import
│       ├── chat.py              # POST /api/v1/chat/completions
│       └── session.py           # GET history / DELETE session
├── tests/                       # 单元测试 + API 测试 + 集成测试
├── docs/
│   ├── ARCHITECTURE.md          # 本文档
│   └── API.md                   # API 接口文档
├── docker-compose.yml           # Redis Stack 一键启动
├── requirements.txt
└── .env.example
```

---

## 2. 切片策略 (Chunking Strategy)

### 2.1 设计目标

- **语义完整性**：尽量在段落/标题边界切分，避免硬截断导致语义断裂
- **检索粒度**：单个片段包含足够信息回答问题，但又不过长导致噪声
- **上下文连续**：相邻片段保留重叠，避免边界信息丢失

### 2.2 Markdown 切片

对于 Markdown 格式文档，采用**两阶段切片**：

**阶段一：按标题层级切分**
- 使用正则匹配 `#` ~ `######` 标题
- 维护标题栈，构建当前节的标题路径（如 `顶层 > 子标题 A > 子标题 A.1`）
- 每个标题到下一个标题之间的内容为一个"节"

**阶段二：滑动窗口二次切分**
- 如果某个"节"的长度超过 `chunk_size`（默认 500 字符），使用滑动窗口进一步切分
- 相邻片段重叠 `chunk_overlap`（默认 50 字符）
- 切分时优先在段落分隔符（`\n\n`）、换行符（`\n`）、句号（`。`）等边界处断开
- 标题路径作为前缀附加到片段内容中（`[标题路径]\n正文`），增强检索时的上下文信息

### 2.3 纯文本切片

对于无格式的纯文本：
- 先按段落（`\n\n`）初步分割
- 再用滑动窗口合并/切分到目标长度
- 同样在句子边界处优先断开

### 2.4 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 500 | 每个片段目标字符数 |
| `chunk_overlap` | 50 | 相邻片段重叠字符数 |

### 2.5 边界处理

- 在 `chunk_size` 附近向前搜索最近的段落/句子边界（搜索范围 100 字符）
- 优先级：`\n\n` > `\n` > `。` > `. ` > `！` > `？` > `; ` > `；`
- 若找不到边界，在 `chunk_size` 处硬截断

---

## 3. Redis 的具体用途

Redis Stack 在本系统中承担**两个核心职责**：

### 3.1 向量数据库（RediSearch 模块）

**存储结构：**
- Key 格式：`doc:chunk:{doc_id}:{chunk_index}`
- 数据类型：Redis Hash
- 字段：
  - `content` (TEXT) — 片段原文
  - `doc_id` (TAG) — 所属文档 ID
  - `chunk_index` (NUMERIC) — 片段序号
  - `heading_path` (TEXT) — Markdown 标题路径
  - `embedding` (VECTOR) — float32 向量

**索引：**
- 索引名：`idx:doc_chunks`
- 使用 `FT.CREATE` 创建，前缀匹配 `doc:chunk:`
- 向量字段配置：`FLAT` 索引、`FLOAT32` 类型、`COSINE` 距离度量
- 选择 FLAT 而非 HNSW：中小规模知识库下精度更高，实现更简单

**检索：**
- 使用 `FT.SEARCH` + KNN 语法：`(*)=>[KNN {top_k} @embedding $query_vec AS distance]`
- 返回 cosine distance，转换为 similarity score：`similarity = 1 - distance/2`

### 3.2 会话历史存储

**存储结构：**
- Key 格式：`session:{session_id}:history`
- 数据类型：Redis List
- 元素：JSON 序列化的 `{role, content, timestamp}`
- 写入方式：`LPUSH`（最新在前）+ `LTRIM`（截断保留最近 N 条）

**为什么用 List 而非 Stream：**
- List 的 LPUSH/LTRIM/LRANGE 操作足以满足需求
- 消息量级有限（受 max_history_turns 限制），不需要 Stream 的消费者组能力
- 实现更简单，调试更方便

### 3.3 为什么选择 Redis

| 优势 | 说明 |
|------|------|
| 一站式 | 向量检索 + 会话存储 + 缓存，单一中间件覆盖全部需求 |
| 低延迟 | 内存数据库，检索延迟 < 1ms |
| 运维简单 | Redis Stack 单容器启动，无需额外向量数据库 |
| 生态成熟 | redis-py 官方支持 RediSearch，文档完善 |

---

## 4. 长对话处理策略

### 4.1 核心挑战

- LLM 有 token 上限（如 GPT-4o-mini 为 128K，但实际可用远小于此）
- 多轮对话历史不断增长，prompt 会超出限制
- 过长的历史上下文会降低回答质量（噪声增加）

### 4.2 解决方案

**滑动窗口截断**
- 只保留最近 `max_history_turns`（默认 6）轮对话 = 12 条消息
- 每轮 = 1 条 user + 1 条 assistant
- 超出时自动 `LTRIM` 截断最旧的消息

**Token 预算控制**
- `max_context_tokens`（默认 3000）限制拼入 prompt 的上下文 token 上限
- 检索片段 + 对话历史 + 当前问题的总 token 不应超过此值
- 若检索结果过长，会截断低分片段

**检索结果不持久化**
- 检索到的文档片段仅在当前轮使用，不存入会话历史
- 避免历史膨胀，每轮重新检索保证最新知识库状态

### 4.3 Prompt 组装顺序

```
1. System Prompt — 角色定义 + 严格基于文档的规则约束
2. 检索上下文 — 本轮检索到的 Top-K 片段（格式化）
3. 对话历史 — 最近 N 轮的 user/assistant 消息
4. 当前问题 — 用户本轮提问
```

System Prompt 中明确要求：
- 回答必须基于参考资料
- 不得编造、猜测
- 资料不足时如实回答"无法回答"

### 4.4 扩展方案（未来优化）

若需支持更长的对话历史：
- **历史摘要压缩**：将超出窗口的旧对话用 LLM 生成摘要，作为 system prompt 的一部分
- **分层检索**：对历史对话也做向量化，按相关性检索历史片段
- **Token 计数器**：使用 tiktoken 精确计算 token，动态调整保留的历史轮数

---

## 5. 流式输出实现

### 5.1 SSE (Server-Sent Events)

- 使用 FastAPI 的 `StreamingResponse`，Content-Type: `text/event-stream`
- 每个事件格式：`data: {json}\n\n`
- 首个事件携带 `retrieved_chunks`（检索到的片段，供前端展示来源）
- 中间事件携带 `delta`（增量文本）
- 末尾事件 `finish: true` 标记结束

### 5.2 异步流式生成

- 使用 `AsyncOpenAI` 客户端的 `stream=True` 模式
- `chat_service.stream_chat()` 是一个 async generator
- LLM 每产出一个 token/chunk，立即 yield 为 SSE 事件
- 实现"边生成边返回"的实时体验

---

## 6. 数据流详解

### 6.1 文档导入流程

```
POST /api/v1/document/import
  │
  ├── 1. DocumentChunker.split()  →  切分为 Chunks
  ├── 2. EmbeddingService.embed_batch()  →  批量向量化
  ├── 3. VectorStore.ensure_index()  →  确保 RediSearch 索引存在
  ├── 4. VectorStore.store_chunks()  →  Pipeline 批量写入 Redis
  └── 5. 返回 {doc_id, chunk_count, total_chars}
```

### 6.2 问答流程

```
POST /api/v1/chat/completions
  │
  ├── 1. RetrievalService.retrieve()
  │     ├── EmbeddingService.embed(question)  →  问题向量
  │     └── VectorStore.search(vec, top_k)  →  Top-K 片段
  │
  ├── 2. ChatService._build_messages()
  │     ├── System Prompt (严格基于文档)
  │     ├── 检索上下文 (格式化片段)
  │     ├── SessionService.get_history()  →  对话历史
  │     └── 当前问题
  │
  ├── 3. SessionService.add_message("user", question)  →  保存问题
  │
  ├── 4. AsyncOpenAI.chat.completions.create(stream=True)
  │     └── 逐 token yield SSE 事件
  │
  ├── 5. SessionService.add_message("assistant", answer)  →  保存回答
  │
  └── 6. yield finish 事件
```
