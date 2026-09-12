# 智能文档助手 (Smart Doc Assistant)

基于 RAG (检索增强生成) 的智能文档问答系统。支持文档导入、语义检索、流式问答与多轮对话管理。

## 功能特性

- **知识库导入**：支持 Markdown / 纯文本，自动切片 + 向量化 + Redis 持久化
- **语义检索**：基于 RediSearch 向量相似度 (KNN) 检索 Top-N 相关片段
- **增强生成问答**：严格基于文档内容回答，流式 SSE 输出，支持多轮对话
- **会话管理**：查看 / 清除会话历史

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
cd smart-doc-assistant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动 Redis Stack

```bash
# 方式一：Docker Compose（推荐）
docker compose up -d

# 方式二：直接使用本地 Redis Stack
# 确保 Redis >= 7.2 且加载了 RediSearch 模块
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 查看 Swagger 文档。

### 5. 运行测试

```bash
# 单元测试 + API 测试（不需要 Redis / LLM）
pytest tests/ -v

# 集成测试（需要真实 Redis + LLM）
RUN_INTEGRATION=1 pytest tests/test_integration.py -v -s
```

## 项目结构

```
smart-doc-assistant/
├── app/
│   ├── main.py                  # FastAPI 入口
│   ├── config.py                # 环境变量配置
│   ├── models/schemas.py        # 数据模型
│   ├── services/                # 核心服务层
│   │   ├── document_service.py  # 文档切片
│   │   ├── embedding_service.py # 向量化
│   │   ├── vector_store.py      # Redis 向量存储
│   │   ├── retrieval_service.py # 语义检索
│   │   ├── chat_service.py      # RAG 问答
│   │   └── session_service.py   # 会话管理
│   └── routers/                 # API 路由
├── tests/                       # 测试
├── docs/
│   ├── ARCHITECTURE.md          # 架构设计
│   └── API.md                   # API 文档
├── docker-compose.yml           # Redis Stack
└── requirements.txt
```

## 技术栈

| 组件 | 说明 |
|------|------|
| FastAPI | 异步 Web 框架，支持 SSE 流式响应 |
| Redis Stack | 向量数据库 (RediSearch) + 会话存储 |
| OpenAI API | Embedding + LLM（支持任何 OpenAI 兼容接口） |

## 文档

- [架构设计说明](docs/ARCHITECTURE.md)
- [API 接口文档](docs/API.md)
