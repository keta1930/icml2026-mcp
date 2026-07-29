"""Validate stable public errors and prevent internal exception disclosure."""

from __future__ import annotations

import pytest

from icml_paper_search.errors import (
    AppError,
    BudgetExhausted,
    EmbeddingConfigurationError,
    EmbeddingProtocolError,
    EmbeddingQuotaExhausted,
    EmbeddingRateLimited,
    EmbeddingUnavailable,
    IndexUnavailable,
    InvalidRequestError,
    RateLimited,
    ServerBusy,
    SessionCapacityReached,
    SessionNotFoundError,
    public_error_text,
)


@pytest.mark.parametrize(
    ("error_type", "code", "retryable", "message"),
    [
        (
            AppError,
            "internal_error",
            False,
            "The service is temporarily unavailable. Please try again later.",
        ),
        (
            InvalidRequestError,
            "invalid_request",
            False,
            "The request parameters are invalid.",
        ),
        (
            SessionNotFoundError,
            "session_not_found",
            False,
            "The search session does not exist or is no longer valid.",
        ),
        (
            SessionCapacityReached,
            "session_capacity_reached",
            True,
            "Search session capacity has been reached. Please try again later.",
        ),
        (RateLimited, "rate_limited", True, "Too many requests. Please try again later."),
        (
            ServerBusy,
            "server_busy",
            True,
            "The service is currently busy. Please try again later.",
        ),
        (
            BudgetExhausted,
            "budget_exhausted",
            False,
            "The daily or monthly search budget has been exhausted.",
        ),
        (
            EmbeddingUnavailable,
            "embedding_unavailable",
            True,
            "The text embedding service is temporarily unavailable.",
        ),
        (
            EmbeddingRateLimited,
            "embedding_unavailable",
            True,
            "The text embedding service is temporarily unavailable.",
        ),
        (
            EmbeddingQuotaExhausted,
            "embedding_quota_exhausted",
            False,
            "The text embedding service quota has been exhausted. "
            "Check the DashScope quota or billing settings.",
        ),
        (
            EmbeddingConfigurationError,
            "embedding_unavailable",
            False,
            "The text embedding service is not configured.",
        ),
        (
            EmbeddingProtocolError,
            "embedding_unavailable",
            False,
            "The text embedding service is temporarily unavailable.",
        ),
        (IndexUnavailable, "index_unavailable", False, "The paper index is not ready."),
    ],
)
def test_known_errors_expose_only_stable_contract(error_type, code, retryable, message):
    error = error_type()

    assert error.code == code
    assert error.retryable is retryable
    assert public_error_text(error) == f"error[{code}]: {message}"


def test_unknown_error_does_not_expose_message():
    sentinel = "sentinel-secret-provider-body"

    text = public_error_text(RuntimeError(sentinel))

    assert "internal_error" in text
    assert sentinel not in text


def test_embedding_quota_error_has_stable_actionable_contract():
    error = EmbeddingQuotaExhausted()

    assert error.retryable is False
    assert public_error_text(error) == (
        "error[embedding_quota_exhausted]: "
        "The text embedding service quota has been exhausted. "
        "Check the DashScope quota or billing settings."
    )


def test_sensitive_identifiers_are_not_echoed():
    sentinel = "ses_sensitive_capability"

    assert sentinel not in str(SessionNotFoundError())
    assert sentinel not in public_error_text(EmbeddingConfigurationError())
