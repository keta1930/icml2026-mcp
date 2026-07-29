"""Validate open-source runtime settings, secret precedence, and sanitization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from icml_paper_search.config import Settings
from icml_paper_search.errors import EmbeddingConfigurationError, public_error_text


def test_open_source_defaults_are_local_and_bounded():
    settings = Settings(_env_file=None)

    assert settings.embedding_model == "qwen3.7-text-embedding"
    assert settings.embedding_dim == 1024
    assert settings.embedding_batch_size == 20
    assert settings.provider_concurrency == 4
    assert settings.chroma_concurrency == 2
    assert settings.max_sessions == 256
    assert settings.ip_session_open_rate_per_minute == 3
    assert settings.ip_session_open_burst == 2
    assert settings.ip_session_opens_per_day == 20
    assert all("agent4sci.com" not in host for host in settings.allowed_hosts)
    assert "milvus" not in repr(settings).lower()


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
        {"dense_weight": 0.9, "bm25_weight": 0.2},
        {"embedding_batch_size": 21},
        {"embedding_dim": 768},
        {"daily_token_budget": 100, "monthly_token_budget": 99},
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
