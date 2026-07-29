"""Validate open-source runtime settings, secret precedence, and sanitization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from icml_paper_search.config import Settings
from icml_paper_search.domain.errors import (
    EmbeddingConfigurationError,
    public_error_text,
)


def test_open_source_defaults_are_local_and_bounded():
    settings = Settings(_env_file=None)

    assert settings.embedding_model == "qwen3.7-text-embedding"
    assert settings.embedding_dim == 1024
    assert settings.embedding_batch_size == 20
    assert settings.milvus_uri == "http://127.0.0.1:19530"
    assert settings.provider_concurrency == 64
    assert settings.milvus_concurrency == 32
    assert settings.max_sessions == 4096
    assert settings.provider_requests_per_minute == 24_000
    assert settings.provider_tokens_per_minute == 1_000_000
    assert settings.ip_rate_per_minute == 600
    assert all("agent4sci.com" not in host for host in settings.allowed_hosts)


def test_secret_file_takes_precedence(tmp_path):
    secret_file = tmp_path / "dashscope-key"
    secret_file.write_text("file-sentinel\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        dashscope_api_key="environment-sentinel",
        dashscope_api_key_file=secret_file,
    )

    assert settings.resolve_dashscope_api_key() == "file-sentinel"
    assert "environment-sentinel" not in repr(settings)


def test_missing_secret_fails_closed_without_path_or_value(tmp_path):
    missing = tmp_path / "sentinel-secret-path"
    settings = Settings(_env_file=None, dashscope_api_key_file=missing)

    with pytest.raises(EmbeddingConfigurationError) as captured:
        settings.resolve_dashscope_api_key()

    public = public_error_text(captured.value)
    assert "sentinel-secret-path" not in public
    assert "embedding_unavailable" in public


@pytest.mark.parametrize(
    "override",
    [
        {"embedding_batch_size": 21},
        {"embedding_dim": 768},
        {"milvus_uri": "tcp://127.0.0.1:19530"},
        {"milvus_collection_prefix": "invalid-prefix"},
        {"global_rate_per_minute": 24_001},
        {
            "ip_session_open_rate_per_minute": 2,
            "ip_session_open_burst": 3,
        },
        {"trusted_proxy_enabled": True},
        {"allowed_hosts": ("https://invalid.example",)},
        {"allowed_origins": ("https://*.invalid.example",)},
        {"dashscope_http_base_url": "https://dashscope-intl.aliyuncs.com/api/v1"},
    ],
)
def test_invalid_boundary_combinations_are_rejected(override):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)
