"""Stable, sanitized application error contract."""

from __future__ import annotations


class AppError(Exception):
    """Carry a public error code without exposing internal exception details."""

    code = "internal_error"
    public_message = "The service is temporarily unavailable. Please try again later."
    retryable = False

    def __init__(
        self,
        *,
        code: str | None = None,
        public_message: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.code = code or type(self).code
        self.public_message = public_message or type(self).public_message
        self.retryable = type(self).retryable if retryable is None else retryable
        super().__init__(self.public_message)

    def __str__(self) -> str:
        return self.public_message


class InvalidRequestError(AppError):
    """Indicate that client parameters violate the public contract."""

    code = "invalid_request"
    public_message = "The request parameters are invalid."


class SessionNotFoundError(AppError):
    """Hide invalid, expired, and closed session tokens behind one error."""

    code = "session_not_found"
    public_message = "The search session does not exist or is no longer valid."


class SessionCapacityReached(AppError):
    """Indicate that no additional business sessions can be created."""

    code = "session_capacity_reached"
    public_message = "Search session capacity has been reached. Please try again later."
    retryable = True


class RateLimited(AppError):
    """Indicate that a caller or global rate limit has been reached."""

    code = "rate_limited"
    public_message = "Too many requests. Please try again later."
    retryable = True


class ServerBusy(AppError):
    """Indicate that the bounded execution queue has no available capacity."""

    code = "server_busy"
    public_message = "The service is currently busy. Please try again later."
    retryable = True


class EmbeddingUnavailable(AppError):
    """Indicate that the embedding provider is temporarily or permanently unavailable."""

    code = "embedding_unavailable"
    public_message = "The text embedding service is temporarily unavailable."
    retryable = True


class EmbeddingRateLimited(EmbeddingUnavailable):
    """Indicate a retryable provider rate limit."""


class EmbeddingQuotaExhausted(EmbeddingUnavailable):
    """Indicate exhausted provider quota for which retries cannot help."""

    code = "embedding_quota_exhausted"
    public_message = (
        "The text embedding service quota has been exhausted. "
        "Check the DashScope quota or billing settings."
    )
    retryable = False


class EmbeddingConfigurationError(EmbeddingUnavailable):
    """Indicate a missing server-side embedding secret or region configuration."""

    public_message = "The text embedding service is not configured."
    retryable = False


class EmbeddingProtocolError(EmbeddingUnavailable):
    """Indicate that a provider response violates the vector contract."""

    retryable = False


class IndexUnavailable(AppError):
    """Indicate that the Milvus paper index is missing or invalid."""

    code = "index_unavailable"
    public_message = "The paper index is not ready."


class IndexBuildLimitExceeded(AppError):
    """Indicate that an explicit index-build input cap was reached."""

    code = "index_build_limit_exceeded"
    public_message = "The index build reached its configured input-token limit."


def public_error_text(exc: BaseException) -> str:
    """Map any exception to a short public message without internal details."""
    if isinstance(exc, AppError):
        error = exc
    elif isinstance(exc, (TypeError, ValueError)):
        error = InvalidRequestError()
    else:
        error = AppError()
    return f"error[{error.code}]: {error.public_message}"
