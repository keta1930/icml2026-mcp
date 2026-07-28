"""DashScope embedding payload、协议校验、重试与脱敏测试。"""

from __future__ import annotations

from http import HTTPStatus

import pytest

from icml_paper_search.config import Settings
from icml_paper_search.embedding import DashScopeEncoder
from icml_paper_search.errors import (
    EmbeddingProtocolError,
    EmbeddingRateLimited,
    EmbeddingUnavailable,
)

DIMENSION = 1024
SENTINEL = "secret-sentinel-never-print"


def response(
    count: int,
    *,
    status: int = HTTPStatus.OK,
    order: list[int] | None = None,
    dimension: int = DIMENSION,
    tokens: int = 7,
):
    indices = order if order is not None else list(range(count))
    return {
        "status_code": status,
        "output": {
            "embeddings": [
                {"text_index": index, "embedding": [float(index)] * dimension}
                for index in indices
            ]
        },
        "usage": {"total_tokens": tokens},
    }


def make_settings(**values) -> Settings:
    return Settings(dashscope_api_key=SENTINEL, **values)


def test_query_payload_and_usage_are_explicit():
    calls = []

    def call(**payload):
        calls.append(payload)
        return response(1, tokens=11)

    encoder = DashScopeEncoder(make_settings(), call=call)
    result = encoder.encode_query_with_usage("retrieval augmented generation")

    assert len(result.vectors[0]) == DIMENSION
    assert result.total_tokens == 11
    assert calls[0]["model"] == "qwen3.7-text-embedding"
    assert calls[0]["input"] == ["retrieval augmented generation"]
    assert calls[0]["text_type"] == "query"
    assert calls[0]["dimension"] == DIMENSION
    assert calls[0]["output_type"] == "dense"
    assert calls[0]["base_address"] == "https://dashscope.aliyuncs.com/api/v1"
    assert calls[0]["request_timeout"] > 0
    assert calls[0]["api_key"] == SENTINEL


def test_documents_batch_at_20_and_restore_text_index_order():
    batch_sizes = []

    def call(**payload):
        size = len(payload["input"])
        batch_sizes.append(size)
        return response(size, order=list(reversed(range(size))), tokens=size)

    encoder = DashScopeEncoder(make_settings(), call=call)
    result = encoder.encode_documents_with_usage([f"paper {i}" for i in range(41)])

    assert batch_sizes == [20, 20, 1]
    assert result.total_tokens == 41
    assert [vector[0] for vector in result.vectors[:20]] == list(map(float, range(20)))
    assert len(result.vectors) == 41


@pytest.mark.parametrize(
    "bad_response",
    [
        response(1, dimension=DIMENSION - 1),
        response(1, order=[1]),
        {"status_code": 200, "output": {"embeddings": []}, "usage": {"total_tokens": 1}},
        {
            "status_code": 200,
            "output": {"embeddings": [{"text_index": 0, "embedding": [True] * DIMENSION}]},
            "usage": {"total_tokens": 1},
        },
    ],
)
def test_protocol_mismatch_is_rejected_without_retry(bad_response):
    calls = 0

    def call(**_payload):
        nonlocal calls
        calls += 1
        return bad_response

    with pytest.raises(EmbeddingProtocolError):
        DashScopeEncoder(make_settings(), call=call).encode_query("query")
    assert calls == 1


def test_retries_only_transient_status_with_full_jitter():
    statuses = iter([429, 503, 200])
    sleeps = []

    def call(**_payload):
        return response(1, status=next(statuses))

    encoder = DashScopeEncoder(
        make_settings(),
        call=call,
        sleep=sleeps.append,
        random_uniform=lambda low, high: high,
    )
    assert len(encoder.encode_query("query")) == DIMENSION
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("status", [400, 404, 422])
def test_non_retryable_4xx_is_called_once(status):
    calls = 0

    def call(**_payload):
        nonlocal calls
        calls += 1
        return response(1, status=status)

    with pytest.raises(EmbeddingUnavailable) as caught:
        DashScopeEncoder(make_settings(), call=call).encode_query("query")
    assert caught.value.retryable is False
    assert calls == 1


def test_rate_limit_exhaustion_has_stable_secret_free_error(caplog):
    def call(**_payload):
        return response(1, status=429)

    with pytest.raises(EmbeddingRateLimited) as caught:
        DashScopeEncoder(
            make_settings(embedding_max_attempts=2),
            call=call,
            sleep=lambda _seconds: None,
        ).encode_query(SENTINEL)

    combined = f"{caught.value!s} {caught.value!r} {caplog.text}"
    assert SENTINEL not in combined


def test_network_failure_retries_but_unknown_exception_does_not():
    network_calls = 0

    def network_call(**_payload):
        nonlocal network_calls
        network_calls += 1
        raise ConnectionError(SENTINEL)

    with pytest.raises(EmbeddingUnavailable) as caught:
        DashScopeEncoder(
            make_settings(embedding_max_attempts=2),
            call=network_call,
            sleep=lambda _seconds: None,
        ).encode_query("query")
    assert network_calls == 2
    assert SENTINEL not in str(caught.value)

    unknown_calls = 0

    def unknown_call(**_payload):
        nonlocal unknown_calls
        unknown_calls += 1
        raise RuntimeError(SENTINEL)

    with pytest.raises(EmbeddingUnavailable) as caught:
        DashScopeEncoder(make_settings(), call=unknown_call).encode_query("query")
    assert unknown_calls == 1
    assert caught.value.retryable is False
    assert SENTINEL not in str(caught.value)
