"""Central configuration, overridable through `ICML_PAPERS_` environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import EmbeddingConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Define runtime boundaries for the open-source single-node service."""

    model_config = SettingsConfigDict(
        env_prefix="ICML_PAPERS_",
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    index_root: Path = Field(default=PROJECT_ROOT / ".data" / "index")
    state_db_path: Path = Field(default=PROJECT_ROOT / ".data/state/runtime.sqlite3")
    sqlite_busy_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)

    dashscope_api_key: SecretStr | None = Field(default=None, repr=False)
    dashscope_api_key_file: Path | None = None
    dashscope_http_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dim: int = Field(default=1_024, ge=1)
    embedding_batch_size: int = Field(default=20, ge=1, le=20)
    embedding_max_attempts: int = Field(default=3, ge=1, le=5)
    embedding_deadline_seconds: float = Field(default=12.0, gt=0, le=60)

    dense_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)

    default_top_k: int = Field(default=10, ge=1)
    max_top_k: int = Field(default=50, ge=1)
    depth_multiplier: int = Field(default=4, ge=1)
    max_search_depth: int = Field(default=200, ge=1)
    max_query_chars: int = Field(default=2_000, ge=1)
    max_get_paper_ids: int = Field(default=50, ge=1)

    session_idle_ttl_seconds: int = Field(default=86_400, ge=60)
    max_sessions: int = Field(default=256, ge=1)
    exhausted_after_empty_rounds: int = Field(default=3, ge=1)
    max_session_rounds: int = Field(default=200, ge=1)
    max_session_seen: int = Field(default=6_341, ge=1)
    max_session_audit_records: int = Field(default=200, ge=0)
    session_summary_paper_limit: int = Field(default=500, ge=0)
    session_label_max_chars: int = Field(default=120, ge=1)

    provider_concurrency: int = Field(default=4, ge=1)
    chroma_concurrency: int = Field(default=2, ge=1)
    search_queue_size: int = Field(default=64, ge=1)
    queue_timeout_seconds: float = Field(default=10.0, gt=0)
    ip_rate_per_minute: int = Field(default=6, ge=1)
    ip_rate_burst: int = Field(default=3, ge=1)
    ip_searches_per_day: int = Field(default=200, ge=1)
    ip_session_open_rate_per_minute: int = Field(default=3, ge=1)
    ip_session_open_burst: int = Field(default=2, ge=1)
    ip_session_opens_per_day: int = Field(default=20, ge=1)
    global_rate_per_minute: int = Field(default=120, ge=1)
    daily_token_budget: int = Field(default=500_000, ge=1)
    monthly_token_budget: int = Field(default=10_000_000, ge=1)
    embedding_cache_max_entries: int = Field(default=10_000, ge=0)
    embedding_cache_ttl_seconds: int = Field(default=86_400, ge=0)

    abstract_preview_chars: int = Field(default=360, ge=0)
    max_authors_shown: int = Field(default=4, ge=0)

    server_host: str = "127.0.0.1"
    server_port: int = Field(default=20_441, ge=1, le=65_535)
    allowed_hosts: tuple[str, ...] = (
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
        "icml-mcp:*",
    )
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:*",
        "http://localhost:*",
    )
    trusted_proxy_enabled: bool = False
    trusted_proxy_header: str = "X-ICML-Client-IP"
    trusted_proxy_networks: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Settings:
        """Reject combinations that would violate index, cost, or service boundaries."""
        if abs((self.dense_weight + self.bm25_weight) - 1.0) > 1e-9:
            raise ValueError("dense_weight and bm25_weight must sum to 1")
        if self.embedding_model == "qwen3.7-text-embedding" and self.embedding_dim != 1_024:
            raise ValueError("qwen3.7-text-embedding requires the 1024-dimensional index contract")
        if self.max_search_depth < self.max_top_k:
            raise ValueError("max_search_depth must be >= max_top_k")
        if self.monthly_token_budget < self.daily_token_budget:
            raise ValueError("monthly_token_budget must be >= daily_token_budget")
        if self.ip_rate_burst > self.ip_rate_per_minute:
            raise ValueError("ip_rate_burst must not exceed ip_rate_per_minute")
        if self.ip_session_open_burst > self.ip_session_open_rate_per_minute:
            raise ValueError(
                "ip_session_open_burst must not exceed "
                "ip_session_open_rate_per_minute"
            )
        if self.max_session_audit_records > self.max_session_rounds:
            raise ValueError("max_session_audit_records must not exceed max_session_rounds")
        if self.index_root.resolve() == self.state_db_path.resolve():
            raise ValueError("index_root and state_db_path must not resolve to the same path")
        if self.trusted_proxy_enabled and not self.trusted_proxy_networks:
            raise ValueError("trusted_proxy_networks is required when trusted proxy mode is enabled")
        if any(not _valid_allowed_host(value) for value in self.allowed_hosts):
            raise ValueError("allowed_hosts entries must use host[:port] or host:* format")
        if any(not _valid_allowed_origin(value) for value in self.allowed_origins):
            raise ValueError("allowed_origins entries must be HTTP(S) origins or use a :* port wildcard")
        parsed = urlparse(self.dashscope_http_base_url)
        host = (parsed.hostname or "").lower()
        beijing = host == "dashscope.aliyuncs.com" or host.endswith(
            ".cn-beijing.maas.aliyuncs.com"
        )
        if parsed.scheme != "https" or not beijing or parsed.path.rstrip("/") != "/api/v1":
            raise ValueError("dashscope_http_base_url must be the Beijing-region HTTPS native API")
        return self

    def resolve_dashscope_api_key(self) -> str:
        """Resolve the key with secret-file precedence and return only sanitized errors."""
        if self.dashscope_api_key_file is not None:
            try:
                value = self.dashscope_api_key_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise EmbeddingConfigurationError() from exc
            if not value:
                raise EmbeddingConfigurationError()
            return value
        if self.dashscope_api_key is not None:
            value = self.dashscope_api_key.get_secret_value().strip()
            if value:
                return value
        raise EmbeddingConfigurationError()

    @property
    def index_path(self) -> Path:
        """Return the paper metadata snapshot path."""
        return self.data_dir / "index.json"

    @property
    def abstracts_path(self) -> Path:
        """Return the paper abstract snapshot path."""
        return self.data_dir / "abstracts.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-default settings."""
    return Settings()


def _valid_allowed_host(value: str) -> bool:
    """Validate exact hosts or FastMCP-supported port wildcards."""
    item = value.strip()
    return bool(item) and "://" not in item and "/" not in item and (
        "*" not in item or item.endswith(":*") and item.count("*") == 1
    )


def _valid_allowed_origin(value: str) -> bool:
    """Validate exact HTTP origins, allowing wildcards only in the port position."""
    item = value.strip()
    parsed = urlparse(item[:-2] if item.endswith(":*") else item)
    wildcard_valid = "*" not in item or item.endswith(":*") and item.count("*") == 1
    return bool(
        wildcard_valid
        and parsed.scheme in {"http", "https"}
        and parsed.hostname
        and not parsed.path.rstrip("/")
        and not parsed.query
        and not parsed.fragment
    )
