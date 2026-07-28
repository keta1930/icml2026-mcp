"""Chroma COSINE 与 SQLite BM25 的本地加权 RRF 检索。"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .bm25 import BM25Index, RankedPaperId
from .config import Settings, get_settings
from .embedding import DashScopeEncoder, TextEncoder
from .errors import IndexUnavailable
from .manifest import IndexManifest, resolve_current, validate_index_version
from .models import Paper, SearchFilters, SearchHit


@runtime_checkable
class SearchEngine(Protocol):
    """业务编排层依赖的最小检索能力。"""

    def search(
        self, query: str, *, limit: int, filters: SearchFilters
    ) -> list[SearchHit]: ...

    def fetch(self, paper_ids: Sequence[str]) -> list[Paper]: ...


class ChromaHybridSearchEngine:
    """单进程打开当前版本并执行固定窗口双路召回。"""

    def __init__(
        self,
        settings: Settings | None = None,
        encoder: TextEncoder | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._encoder = encoder or DashScopeEncoder(self._settings)
        self._chroma_limit = threading.BoundedSemaphore(
            self._settings.chroma_concurrency
        )
        self._client: Any | None = None
        self._collection: Any | None = None
        self._bm25: BM25Index | None = None
        self._manifest: IndexManifest | None = None
        self._version_path = None

    @property
    def manifest(self) -> IndexManifest:
        """延迟打开并返回当前索引 manifest。"""
        with self._chroma_limit:
            self._open_current()
            assert self._manifest is not None
            return self._manifest

    def search(
        self, query: str, *, limit: int, filters: SearchFilters
    ) -> list[SearchHit]:
        """两路各召回一次后按 weighted RRF 返回固定窗口。"""
        if limit <= 0:
            return []
        # provider 不占 Chroma semaphore，二者可按各自容量独立背压。
        vector = self._encoder.encode_query(query)
        with self._chroma_limit:
            self._open_current()
            assert self._collection is not None
            assert self._bm25 is not None
            assert self._manifest is not None

            dense = self._dense_search(vector, limit=limit, filters=filters)
            sparse = self._bm25.search(query, limit=limit, filters=filters)
            ranked = _weighted_rrf(
                dense,
                sparse,
                dense_weight=self._manifest.dense_weight,
                bm25_weight=self._manifest.bm25_weight,
                rrf_k=self._manifest.rrf_k,
                limit=limit,
            )
            papers = self._fetch_locked([paper_id for paper_id, _score in ranked])

        by_id = {paper.paper_id: paper for paper in papers}
        return [
            SearchHit(paper=by_id[paper_id], score=score)
            for paper_id, score in ranked
            if paper_id in by_id
        ]

    def fetch(self, paper_ids: Sequence[str]) -> list[Paper]:
        """按输入顺序取回论文，去重并跳过未知 id。"""
        ids = [paper_id for paper_id in dict.fromkeys(paper_ids) if paper_id]
        if not ids:
            return []
        with self._chroma_limit:
            self._open_current()
            return self._fetch_locked(ids)

    def count(self) -> int:
        """返回当前 Chroma collection 论文数。"""
        with self._chroma_limit:
            self._open_current()
            assert self._collection is not None
            return int(self._collection.count())

    def validate(self) -> IndexManifest:
        """零云调用重新核对当前版本的双索引一致性。"""
        with self._chroma_limit:
            version = resolve_current(self._settings.index_root)
            manifest = validate_index_version(version, settings=self._settings)
            if self._version_path is not None and version != self._version_path:
                raise IndexUnavailable()
            return manifest

    def close(self) -> None:
        """关闭 encoder 与 FTS 连接；Chroma client 由 SDK 管理。"""
        with self._chroma_limit:
            if self._bm25 is not None:
                self._bm25.close()
            self._bm25 = None
            self._collection = None
            self._client = None
            self._manifest = None
            self._version_path = None
        close_encoder = getattr(self._encoder, "close", None)
        if callable(close_encoder):
            close_encoder()

    def _open_current(self) -> None:
        """首次使用时校验并打开 current，不在运行中热切版本。"""
        if self._collection is not None:
            return
        version = resolve_current(self._settings.index_root)
        manifest = validate_index_version(version, settings=self._settings)
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(version / "chroma"))
            collection = client.get_collection(
                name=manifest.collection_name, embedding_function=None
            )
            bm25 = BM25Index(version / "bm25.sqlite3")
        except Exception as exc:
            raise IndexUnavailable() from exc
        self._client = client
        self._collection = collection
        self._bm25 = bm25
        self._manifest = manifest
        self._version_path = version

    def _dense_search(
        self, vector: list[float], *, limit: int, filters: SearchFilters
    ) -> list[RankedPaperId]:
        """执行一次 Chroma COSINE fixed-limit 查询。"""
        assert self._collection is not None
        assert self._manifest is not None
        n_results = min(limit, self._manifest.record_count)
        if n_results <= 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=n_results,
                where=_to_chroma_where(filters),
                include=["distances"],
            )
            ids = (result.get("ids") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
        except Exception as exc:
            raise IndexUnavailable() from exc
        if len(ids) != len(distances):
            raise IndexUnavailable()
        return [
            RankedPaperId(
                paper_id=str(paper_id), rank=rank, raw_score=1.0 - float(distance)
            )
            for rank, (paper_id, distance) in enumerate(
                zip(ids, distances, strict=True), start=1
            )
        ]

    def _fetch_locked(self, paper_ids: Sequence[str]) -> list[Paper]:
        """在已持有 Chroma semaphore 时批量取回 metadata。"""
        assert self._collection is not None
        ids = [paper_id for paper_id in dict.fromkeys(paper_ids) if paper_id]
        if not ids:
            return []
        try:
            result = self._collection.get(
                ids=ids, include=["metadatas", "documents"]
            )
            result_ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            documents = result.get("documents") or []
        except Exception as exc:
            raise IndexUnavailable() from exc
        if len(result_ids) != len(metadatas):
            raise IndexUnavailable()

        by_id: dict[str, Paper] = {}
        for index, (paper_id, metadata) in enumerate(
            zip(result_ids, metadatas, strict=True)
        ):
            if not isinstance(metadata, Mapping):
                raise IndexUnavailable()
            mapping = dict(metadata)
            mapping.setdefault("orid", str(paper_id))
            if index < len(documents) and not mapping.get("abstract"):
                mapping["abstract"] = _abstract_from_document(documents[index])
            by_id[str(paper_id)] = _paper_from_mapping(mapping)
        return [by_id[paper_id] for paper_id in ids if paper_id in by_id]


def _to_chroma_where(filters: SearchFilters) -> dict[str, Any] | None:
    """把领域过滤条件转换为 Chroma 结构化 where。"""
    clauses: list[dict[str, Any]] = []
    if filters.area:
        clauses.append({"area": {"$eq": filters.area}})
    if filters.paper_type:
        clauses.append({"type": {"$eq": filters.paper_type}})
    if filters.spotlight_only:
        clauses.append({"spot": {"$eq": True}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _weighted_rrf(
    dense: Sequence[RankedPaperId],
    bm25: Sequence[RankedPaperId],
    *,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
    limit: int,
) -> list[tuple[str, float]]:
    """按理论最大值归一化 weighted RRF 并稳定排序。"""
    dense_ranks = {item.paper_id: item.rank for item in dense}
    bm25_ranks = {item.paper_id: item.rank for item in bm25}
    paper_ids = dense_ranks.keys() | bm25_ranks.keys()
    maximum = (dense_weight + bm25_weight) / (rrf_k + 1)
    if maximum <= 0:
        return []
    missing_rank = 2**31 - 1
    scored: list[tuple[str, float, int, int]] = []
    for paper_id in paper_ids:
        dense_rank = dense_ranks.get(paper_id, missing_rank)
        bm25_rank = bm25_ranks.get(paper_id, missing_rank)
        numerator = 0.0
        if dense_rank != missing_rank:
            numerator += dense_weight / (rrf_k + dense_rank)
        if bm25_rank != missing_rank:
            numerator += bm25_weight / (rrf_k + bm25_rank)
        score = min(1.0, max(0.0, numerator / maximum))
        scored.append((paper_id, score, dense_rank, bm25_rank))
    scored.sort(key=lambda item: (-item[1], item[2], item[3], item[0]))
    return [(paper_id, score) for paper_id, score, _dense, _bm25 in scored[:limit]]


def _paper_from_mapping(mapping: Mapping[str, Any]) -> Paper:
    """适配领域模型迁移期间的新旧构造方法。"""
    constructor = getattr(Paper, "from_mapping", None)
    if constructor is not None:
        return constructor(mapping)
    return Paper.from_entity(dict(mapping))


def _abstract_from_document(document: Any) -> str:
    """仅在 metadata 缺摘要时从 title+abstract 文档回退。"""
    if not isinstance(document, str):
        return ""
    _title, separator, abstract = document.partition("\n")
    return abstract if separator else ""
