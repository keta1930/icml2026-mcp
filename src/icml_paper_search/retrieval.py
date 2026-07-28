"""Milvus 混合检索：jina dense 向量 + BM25 sparse，WeightedRanker 加权融合。"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, runtime_checkable

from .config import Settings, get_settings
from .embedding import JinaEncoder, TextEncoder
from .models import Paper, SearchFilters, SearchHit

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = [
    "orid",
    "title",
    "abstract",
    "authors",
    "area",
    "sub",
    "type",
    "spot",
    "arxiv",
]


@runtime_checkable
class SearchEngine(Protocol):
    """编排层依赖的检索能力，实现可替换（Milvus / 内存 / 远端服务）。"""

    def search(
        self, query: str, *, limit: int, filters: SearchFilters
    ) -> list[SearchHit]: ...

    def fetch(self, paper_ids: Sequence[str]) -> list[Paper]: ...


class MilvusSearchEngine:
    """基于 Milvus `hybrid_search` 的检索引擎。

    连接与 embedding 模型都延迟初始化，构造本身零成本。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        encoder: TextEncoder | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._encoder = encoder or JinaEncoder(self._settings)
        self._client = None

    @property
    def client(self):
        """首次访问时建立 Milvus 连接。"""
        if self._client is None:
            from pymilvus import MilvusClient

            cfg = self._settings
            logger.info("connecting to milvus %s", cfg.milvus_uri)
            self._client = MilvusClient(
                uri=cfg.milvus_uri, token=cfg.milvus_token or None
            )
        return self._client

    def search(
        self, query: str, *, limit: int, filters: SearchFilters
    ) -> list[SearchHit]:
        """执行一次加权混合检索，返回按融合得分降序的命中。"""
        from pymilvus import AnnSearchRequest, WeightedRanker

        cfg = self._settings
        vector = self._encoder.encode_query(query)
        expr = filters.to_expr()

        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=limit,
            expr=expr or None,
        )
        bm25_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit,
            expr=expr or None,
        )
        results = self.client.hybrid_search(
            self._settings.collection,
            [dense_req, bm25_req],
            WeightedRanker(cfg.dense_weight, cfg.bm25_weight),
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )
        return [
            SearchHit(paper=Paper.from_entity(hit["entity"]), score=float(hit["distance"]))
            for hit in (results[0] if results else [])
        ]

    def fetch(self, paper_ids: Sequence[str]) -> list[Paper]:
        """按主键精确取回论文，保持传入顺序，跳过不存在的 id。"""
        ids = [pid for pid in dict.fromkeys(paper_ids) if pid]
        if not ids:
            return []
        quoted = ", ".join(f'"{pid}"' for pid in ids)
        rows = self.client.query(
            self._settings.collection,
            filter=f"orid in [{quoted}]",
            output_fields=OUTPUT_FIELDS,
            limit=len(ids),
        )
        by_id = {row["orid"]: Paper.from_entity(row) for row in rows}
        return [by_id[pid] for pid in ids if pid in by_id]

    def count(self) -> int:
        """collection 内的论文总数。"""
        stats = self.client.get_collection_stats(self._settings.collection)
        return int(stats.get("row_count", 0))
