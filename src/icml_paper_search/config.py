"""集中配置。所有值可通过 `ICML_PAPERS_` 前缀的环境变量覆盖。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """运行时配置，从环境变量与 `.env` 读取。"""

    model_config = SettingsConfigDict(
        env_prefix="ICML_PAPERS_",
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 数据与向量库 ---
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        description="index.json / abstracts.json 所在目录",
    )
    milvus_uri: str = Field(
        default="http://localhost:20340", description="Milvus gRPC 入口"
    )
    milvus_token: str = Field(default="", description="Milvus 认证 token，留空表示无认证")
    collection: str = Field(
        default="icml2026repro_papers", description="论文 collection 名"
    )

    # --- Embedding 模型 ---
    embedding_model_path: str = Field(
        default="/home/yyx/projects/infra/model/jina-embeddings-v5-text-small",
        description="jina-embeddings-v5 本地路径或 HF repo id",
    )
    embedding_device: str = Field(default="cuda", description="torch device")
    embedding_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16", description="模型权重精度"
    )
    embedding_dim: int = Field(default=1024, description="dense 向量维度")
    embedding_batch_size: int = Field(default=64, description="入库时的编码 batch")

    # --- 混合检索排序权重 ---
    dense_weight: float = Field(default=0.8, ge=0.0, description="向量召回权重")
    bm25_weight: float = Field(default=0.2, ge=0.0, description="BM25 召回权重")

    # --- 检索规模 ---
    default_top_k: int = Field(default=10, ge=1, description="未指定时返回的新条目数")
    max_top_k: int = Field(default=50, ge=1, description="单次检索允许的最大新条目数")
    depth_multiplier: int = Field(
        default=4,
        ge=1,
        description="检索窗口 = top_k × 该倍数；窗口内做去重，决定一个 query 能挖多深",
    )
    max_search_depth: int = Field(
        default=200, ge=1, description="检索窗口深度上限，防止下探到无关结果"
    )

    # --- 会话 ---
    session_idle_ttl_seconds: int = Field(
        default=86_400, ge=60, description="会话空闲多久后可被回收"
    )
    max_sessions: int = Field(default=32, ge=1, description="同时存活的会话上限")
    exhausted_after_empty_rounds: int = Field(
        default=3, ge=1, description="连续多少轮零新增即判定该方向检索完毕"
    )

    # --- 输出裁剪 ---
    abstract_preview_chars: int = Field(
        default=360, ge=0, description="列表里摘要的截断长度，0 表示不带摘要"
    )
    max_authors_shown: int = Field(default=4, ge=0, description="列表里展示的作者数")

    # --- MCP server（streamable HTTP transport）---
    server_host: str = Field(
        default="127.0.0.1", description="MCP server 监听地址，本机用 127.0.0.1"
    )
    server_port: int = Field(
        default=20441, ge=1, le=65535, description="MCP server 监听端口（仓库端口段 20440-20449）"
    )

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.dense_weight + self.bm25_weight <= 0:
            raise ValueError("dense_weight 与 bm25_weight 不能同时为 0")
        if self.max_search_depth < self.max_top_k:
            raise ValueError("max_search_depth 必须 >= max_top_k")
        return self

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.json"

    @property
    def abstracts_path(self) -> Path:
        return self.data_dir / "abstracts.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置。"""
    return Settings()
