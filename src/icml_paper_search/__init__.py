"""Local hybrid-search MCP service for ICML 2026 papers."""

from .config import Settings, get_settings
from .embedding import TextEncoder
from .errors import (
    AppError,
    BudgetExhausted,
    EmbeddingConfigurationError,
    EmbeddingProtocolError,
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
from .models import Paper, QueryRecord, RoundCommit, SearchFilters, SearchHit, SearchOutcome, SessionSummary
__version__ = "0.2.0"

__all__ = [
    "AppError", "BudgetExhausted", "EmbeddingConfigurationError",
    "EmbeddingProtocolError", "EmbeddingRateLimited", "EmbeddingUnavailable",
    "IndexUnavailable", "InvalidRequestError", "Paper", "PaperSearchService",
    "QueryRecord", "RateLimited", "RoundCommit", "SearchEngine", "SearchFilters",
    "SearchHit", "SearchOutcome", "ServerBusy", "SessionCapacityReached",
    "SessionManager", "SessionNotFoundError", "SessionStore", "SessionSummary",
    "Settings", "TextEncoder", "__version__", "get_settings", "public_error_text",
]


def __getattr__(name: str):
    """Lazily export runtime protocols while keeping package imports side-effect free."""
    if name == "SearchEngine":
        from .retrieval import SearchEngine

        return SearchEngine
    if name == "PaperSearchService":
        from .service import PaperSearchService

        return PaperSearchService
    if name in {"SessionManager", "SessionStore"}:
        from .sessions import SessionManager, SessionStore

        return {"SessionManager": SessionManager, "SessionStore": SessionStore}[name]
    raise AttributeError(name)
