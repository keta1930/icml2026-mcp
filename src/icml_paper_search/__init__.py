"""ICML 2026 论文检索 MCP server：带会话去重的语义 + BM25 混合检索。"""

from .config import Settings, get_settings
from .models import Paper, SearchFilters, SearchHit, SearchOutcome, SessionSummary
from .service import PaperSearchService
from .sessions import SearchSession, SessionManager, SessionNotFoundError

__version__ = "0.1.0"

__all__ = [
    "Paper",
    "PaperSearchService",
    "SearchFilters",
    "SearchHit",
    "SearchOutcome",
    "SearchSession",
    "SessionManager",
    "SessionNotFoundError",
    "SessionSummary",
    "Settings",
    "__version__",
    "get_settings",
]
