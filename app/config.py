"""
全局配置 —— 通过环境变量 / .env 文件注入。
"""
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"
    redis_index_name: str = "idx:doc_chunks"

    # ---- LLM ----
    llm_api_base: str = "https://api.deepseek.com/v1"
    llm_api_key: str = "sk-your-api-key-here"
    llm_model: str = "deepseek-chat"

    # ---- Embedding（可独立配置，默认跟 LLM 共用）----
    embedding_api_base: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024

    # ---- RAG 参数 ----
    chunk_size: int = 500        # 每个片段目标字符数
    chunk_overlap: int = 50      # 片段间重叠字符数
    top_k: int = 4               # 检索返回的片段数
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    vector_candidate_k: int = 10
    bm25_candidate_k: int = 10
    rrf_k: int = 60
    max_history_turns: int = 6   # 保留的最近对话轮数
    max_context_tokens: int = 3000  # 拼入 prompt 的上下文 token 上限

    # ---- 服务 ----
    app_host: str = "0.0.0.0"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
