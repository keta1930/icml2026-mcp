"""通过 DashScope 为查询与论文文档生成 dense embedding。"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Protocol, runtime_checkable

from .config import Settings, get_settings
from .errors import (
    EmbeddingConfigurationError,
    EmbeddingProtocolError,
    EmbeddingQuotaExhausted,
    EmbeddingRateLimited,
    EmbeddingUnavailable,
)

logger = logging.getLogger(__name__)

_QUOTA_EXHAUSTED_RESPONSES = frozenset(
    {
        (HTTPStatus.BAD_REQUEST, "Arrearage"),
        (HTTPStatus.FORBIDDEN, "AllocationQuota.FreeTierOnly"),
    }
)


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """一次或多次 provider 调用产生的向量和实际 Token 用量。"""

    vectors: tuple[tuple[float, ...], ...]
    total_tokens: int


@runtime_checkable
class TextEncoder(Protocol):
    """检索侧需要的最小同步编码能力。"""

    def encode_query(self, text: str) -> list[float]: ...

    def encode_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]: ...


class DashScopeEncoder:
    """DashScope qwen3.7 embedding 的同步、可测试封装。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        call: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._settings = settings or get_settings()
        self._call = call
        self._sleep = sleep
        self._monotonic = monotonic
        self._random_uniform = random_uniform
        self._local = threading.local()
        self._sessions: set[Any] = set()
        self._sessions_lock = threading.Lock()

    def encode_query(self, text: str) -> list[float]:
        """以 query 语义编码单条检索文本。"""
        result = self.encode_query_with_usage(text)
        return list(result.vectors[0])

    def encode_query_with_usage(self, text: str) -> EmbeddingBatch:
        """编码查询并返回 provider 报告的 Token 用量。"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query 不能为空")
        return self._encode_batch([text], text_type="query")

    def encode_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]:
        """按 provider 批次上限编码文档并保持输入顺序。"""
        result = self.encode_documents_with_usage(texts, show_progress=show_progress)
        return [list(vector) for vector in result.vectors]

    def encode_documents_with_usage(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> EmbeddingBatch:
        """分批编码文档并汇总实际 Token 用量。"""
        del show_progress  # 云端批调用没有本地模型进度条。
        items = list(texts)
        if not items:
            return EmbeddingBatch(vectors=(), total_tokens=0)
        if any(not isinstance(text, str) or not text.strip() for text in items):
            raise ValueError("document 不能为空")

        vectors: list[tuple[float, ...]] = []
        total_tokens = 0
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(items), batch_size):
            result = self._encode_batch(
                items[start : start + batch_size], text_type="document"
            )
            vectors.extend(result.vectors)
            total_tokens += result.total_tokens
        return EmbeddingBatch(vectors=tuple(vectors), total_tokens=total_tokens)

    def _encode_batch(self, texts: list[str], *, text_type: str) -> EmbeddingBatch:
        """执行一个不超过 20 条的 provider 批次。"""
        if len(texts) > 20:
            raise ValueError("单批 embedding 输入不能超过 20 条")
        cfg = self._settings
        api_key = cfg.resolve_dashscope_api_key()
        response = self._call_with_retry(
            api_key=api_key,
            model=cfg.embedding_model,
            input=texts,
            text_type=text_type,
            dimension=cfg.embedding_dim,
            output_type="dense",
            base_address=cfg.dashscope_http_base_url,
        )
        return self._parse_response(response, expected_count=len(texts))

    def _call_with_retry(self, **payload: Any) -> Any:
        """在总 deadline 内仅重试临时网络和 HTTP 故障。"""
        cfg = self._settings
        started = self._monotonic()
        call = self._call or self._default_call

        for attempt in range(1, cfg.embedding_max_attempts + 1):
            remaining = cfg.embedding_deadline_seconds - (self._monotonic() - started)
            if remaining <= 0:
                raise EmbeddingUnavailable()
            try:
                response = call(**payload, request_timeout=remaining)
            except Exception as exc:
                if not _is_network_error(exc):
                    raise EmbeddingUnavailable(retryable=False) from None
                if attempt >= cfg.embedding_max_attempts:
                    raise EmbeddingUnavailable() from None
                self._backoff(attempt, started)
                continue

            status = _response_status(response)
            if status == HTTPStatus.OK:
                return response
            provider_code = _field(response, "code")
            provider_error = (status, provider_code)
            if (
                isinstance(provider_code, str)
                and provider_error in _QUOTA_EXHAUSTED_RESPONSES
            ):
                raise EmbeddingQuotaExhausted()
            if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
                raise EmbeddingConfigurationError()
            if status == HTTPStatus.TOO_MANY_REQUESTS:
                if attempt >= cfg.embedding_max_attempts:
                    raise EmbeddingRateLimited()
                self._backoff(attempt, started)
                continue
            if status == HTTPStatus.REQUEST_TIMEOUT or 500 <= status <= 599:
                if attempt >= cfg.embedding_max_attempts:
                    raise EmbeddingUnavailable()
                self._backoff(attempt, started)
                continue
            raise EmbeddingUnavailable(retryable=False)
        raise EmbeddingUnavailable()

    def _backoff(self, attempt: int, started: float) -> None:
        """执行受总 deadline 约束的 full-jitter 退避。"""
        remaining = self._settings.embedding_deadline_seconds - (
            self._monotonic() - started
        )
        if remaining <= 0:
            raise EmbeddingUnavailable()
        ceiling = min(0.5 * (2 ** (attempt - 1)), remaining)
        self._sleep(self._random_uniform(0.0, ceiling))

    def _parse_response(self, response: Any, *, expected_count: int) -> EmbeddingBatch:
        """校验 provider 响应顺序、维度、数值和 usage。"""
        output = _field(response, "output")
        embeddings = _field(output, "embeddings")
        if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes)):
            raise EmbeddingProtocolError()
        if len(embeddings) != expected_count:
            raise EmbeddingProtocolError()

        ordered: list[tuple[float, ...] | None] = [None] * expected_count
        for item in embeddings:
            index = _field(item, "text_index")
            vector = _field(item, "embedding")
            if not isinstance(index, int) or isinstance(index, bool):
                raise EmbeddingProtocolError()
            if index < 0 or index >= expected_count or ordered[index] is not None:
                raise EmbeddingProtocolError()
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
                raise EmbeddingProtocolError()
            if len(vector) != self._settings.embedding_dim:
                raise EmbeddingProtocolError()
            parsed: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EmbeddingProtocolError()
                number = float(value)
                if not math.isfinite(number):
                    raise EmbeddingProtocolError()
                parsed.append(number)
            ordered[index] = tuple(parsed)

        if any(vector is None for vector in ordered):
            raise EmbeddingProtocolError()
        usage = _field(response, "usage")
        total_tokens = _field(usage, "total_tokens")
        if not isinstance(total_tokens, int) or isinstance(total_tokens, bool):
            raise EmbeddingProtocolError()
        if total_tokens < 0:
            raise EmbeddingProtocolError()
        return EmbeddingBatch(
            vectors=tuple(vector for vector in ordered if vector is not None),
            total_tokens=total_tokens,
        )

    def close(self) -> None:
        """关闭各工作线程持有的 HTTP Session。"""
        with self._sessions_lock:
            sessions, self._sessions = self._sessions, set()
        for session in sessions:
            session.close()
        self._local.session = None
        close = getattr(self._call, "close", None)
        if callable(close):
            close()

    def _default_call(self, **payload: Any) -> Any:
        """按工作线程复用连接并延迟导入 DashScope SDK。"""
        import dashscope
        import requests

        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
            with self._sessions_lock:
                self._sessions.add(session)
        return dashscope.TextEmbedding.call(**payload, session=session)


def _response_status(response: Any) -> int:
    """兼容 SDK object 与测试 mapping 的状态字段。"""
    status = _field(response, "status_code")
    if isinstance(status, HTTPStatus):
        return int(status)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    raise EmbeddingProtocolError()


def _field(container: Any, name: str) -> Any:
    """读取 SDK object 或 mapping 字段。"""
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def _is_network_error(exc: BaseException) -> bool:
    """识别不会暴露 provider 响应内容的网络异常。"""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    module = type(exc).__module__.split(".", 1)[0]
    return module in {"requests", "urllib3", "httpx", "httpcore"}
