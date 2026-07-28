"""稳定、脱敏的应用错误契约。"""

from __future__ import annotations


class AppError(Exception):
    """携带公开错误码但不暴露内部异常细节。"""

    code = "internal_error"
    public_message = "服务暂时不可用，请稍后重试。"
    retryable = False

    def __init__(self, *, code: str | None = None, public_message: str | None = None,
                 retryable: bool | None = None) -> None:
        self.code = code or type(self).code
        self.public_message = public_message or type(self).public_message
        self.retryable = type(self).retryable if retryable is None else retryable
        super().__init__(self.public_message)

    def __str__(self) -> str:
        return self.public_message


class InvalidRequestError(AppError):
    """表示客户端参数不符合公开契约。"""
    code = "invalid_request"
    public_message = "请求参数无效。"


class SessionNotFoundError(AppError):
    """统一隐藏错误、过期或已关闭的 session token。"""
    code = "session_not_found"
    public_message = "检索会话不存在或已失效。"


class SessionCapacityReached(AppError):
    """表示当前不能再创建业务 session。"""
    code = "session_capacity_reached"
    public_message = "检索会话已达容量上限，请稍后重试。"
    retryable = True


class RateLimited(AppError):
    """表示调用方或全局速率已达到上限。"""
    code = "rate_limited"
    public_message = "请求过于频繁，请稍后重试。"
    retryable = True


class ServerBusy(AppError):
    """表示有界执行队列没有可用容量。"""
    code = "server_busy"
    public_message = "服务当前繁忙，请稍后重试。"
    retryable = True


class BudgetExhausted(AppError):
    """表示在线 embedding Token 预算已经耗尽。"""
    code = "budget_exhausted"
    public_message = "今日或本月的检索额度已用尽。"


class EmbeddingUnavailable(AppError):
    """表示 embedding provider 暂时或永久不可用。"""
    code = "embedding_unavailable"
    public_message = "文本向量服务暂时不可用。"
    retryable = True


class EmbeddingRateLimited(EmbeddingUnavailable):
    """表示 provider 返回可重试的限流。"""


class EmbeddingQuotaExhausted(EmbeddingUnavailable):
    """表示 provider 账户额度已耗尽且继续重试无效。"""
    code = "embedding_quota_exhausted"
    public_message = "文本向量服务额度已耗尽，请检查 DashScope 额度或付费设置。"
    retryable = False


class EmbeddingConfigurationError(EmbeddingUnavailable):
    """表示服务端 embedding secret 或地域配置缺失。"""
    public_message = "文本向量服务尚未配置。"
    retryable = False


class EmbeddingProtocolError(EmbeddingUnavailable):
    """表示 provider 响应不符合向量契约。"""
    retryable = False


class IndexUnavailable(AppError):
    """表示本地索引不存在、损坏或版本不一致。"""
    code = "index_unavailable"
    public_message = "论文索引尚未就绪。"


def public_error_text(exc: BaseException) -> str:
    """把任意异常映射为不含内部 message 的公开短文本。"""
    if isinstance(exc, AppError):
        error = exc
    elif isinstance(exc, (TypeError, ValueError)):
        error = InvalidRequestError()
    else:
        error = AppError()
    return f"error[{error.code}]: {error.public_message}"
