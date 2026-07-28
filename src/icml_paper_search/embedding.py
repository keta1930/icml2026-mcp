"""查询/文档向量编码。默认实现是本地 jina-embeddings-v5-text-small。"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, runtime_checkable

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


@runtime_checkable
class TextEncoder(Protocol):
    """检索侧需要的最小编码能力，便于测试替换与更换模型。"""

    def encode_query(self, text: str) -> list[float]: ...

    def encode_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]: ...


class JinaEncoder:
    """jina-embeddings-v5 的 sentence-transformers 封装。

    模型加载昂贵（GPU 上约数秒），因此延迟到首次编码时才加载，
    让 server 启动与首次握手不被阻塞。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None

    @property
    def model(self):
        """首次访问时加载模型（线程内单次）。"""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self):
        # 重依赖延后到此处 import，保证纯 CPU 场景下模块可被导入。
        import torch
        from sentence_transformers import SentenceTransformer

        cfg = self._settings
        logger.info(
            "loading embedding model %s on %s (%s)",
            cfg.embedding_model_path,
            cfg.embedding_device,
            cfg.embedding_dtype,
        )
        dtype = getattr(torch, _DTYPES[cfg.embedding_dtype])
        return SentenceTransformer(
            cfg.embedding_model_path,
            trust_remote_code=True,
            device=cfg.embedding_device,
            model_kwargs={"dtype": dtype},
        )

    def encode_query(self, text: str) -> list[float]:
        """编码检索 query（query 侧 prompt）。"""
        return self._encode([text], prompt_name="query")[0]

    def encode_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]:
        """编码入库文档（document 侧 prompt）。"""
        return self._encode(
            list(texts), prompt_name="document", show_progress=show_progress
        )

    def _encode(
        self, texts: list[str], *, prompt_name: str, show_progress: bool = False
    ) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            task="retrieval",
            prompt_name=prompt_name,
            batch_size=self._settings.embedding_batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def warmup(self) -> None:
        """主动触发模型加载，用于 CLI 或需要可预测首查延迟的场景。"""
        self.encode_query("warmup")
