"""验证公开错误文本不会泄露内部异常。"""

from __future__ import annotations

from icml_paper_search.errors import (
    BudgetExhausted,
    EmbeddingConfigurationError,
    SessionNotFoundError,
    public_error_text,
)


def test_known_error_exposes_only_stable_contract():
    text = public_error_text(BudgetExhausted())

    assert text == "error[budget_exhausted]: 今日或本月的检索额度已用尽。"


def test_unknown_error_does_not_expose_message():
    sentinel = "sentinel-secret-provider-body"

    text = public_error_text(RuntimeError(sentinel))

    assert "internal_error" in text
    assert sentinel not in text


def test_sensitive_identifiers_are_not_echoed():
    sentinel = "ses_sensitive_capability"

    assert sentinel not in str(SessionNotFoundError())
    assert sentinel not in public_error_text(EmbeddingConfigurationError())
