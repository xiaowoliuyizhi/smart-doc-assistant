"""
FastAPI 应用入口 —— 智能文档助手 (Smart Doc Assistant)

启动: uvicorn app.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import chat, document, session
from app.services.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时检查 Redis 连接并预建索引。"""
    store = VectorStore()
    if store.ping():
        store.ensure_index()
        print("[Startup] Redis 连接成功，索引已就绪")
    else:
        print("[Startup] ⚠ Redis 未连接，请确保 Redis Stack 已启动")
    yield
    print("[Shutdown] 应用关闭")


settings = get_settings()

app = FastAPI(
    title="智能文档助手",
    description="""
基于 RAG（检索增强生成）的智能文档问答系统。

### 核心功能
- 📄 **文档导入** — 上传 Markdown/纯文本，自动切片+向量化存储
- 🔍 **语义检索** — 根据提问在知识库中检索最相关的文档片段
- 🤖 **增强问答** — 结合检索结果与 LLM 流式生成精准回答
- 💬 **多轮对话** — 支持上下文理解，保留对话历史
- 🗑️ **会话管理** — 查看和清除任意会话记录
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理：返回 JSON 格式的错误
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"},
    )

# 注册路由
app.include_router(document.router)
app.include_router(chat.router)
app.include_router(session.router)

# 静态文件
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 前端页面
@app.get("/app", tags=["界面"], summary="智能文档助手界面")
async def app_page():
    """返回智能文档助手前端页面"""
    from fastapi.responses import FileResponse
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health", tags=["系统"], summary="服务健康检查")
async def health():
    """检查 Redis 连接状态与知识库数据量"""
    store = VectorStore()
    redis_ok = store.ping()
    chunk_count = store.count_chunks() if redis_ok else 0
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
        "chunk_count": chunk_count,
    }


@app.get("/", tags=["系统"], summary="服务信息")
async def root():
    """查看服务基本信息"""
    return {
        "service": "智能文档助手",
        "version": "1.0.0",
        "docs": "/docs",
    }
