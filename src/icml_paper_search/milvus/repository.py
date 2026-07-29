"""Milvus repository for dense, BM25, and ID-based paper retrieval."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from ..config import Settings, get_settings
from ..domain.errors import IndexUnavailable
from ..domain.models import Paper, SearchFilters, SearchHit
from ..embeddings.dashscope import DashScopeEncoder, TextEncoder
from ..indexing.manifest import IndexManifest, validate_current_index
from .schema import (
    DENSE_FIELD,
    PAPER_OUTPUT_FIELDS,
    RRF_K,
    SPARSE_FIELD,
)


@runtime_checkable
class SearchEngine(Protocol):
    """Retrieval boundary consumed by the application service."""

    def search(
        self, query: str, *, limit: int, filters: SearchFilters
    ) -> list[SearchHit]: ...

    def fetch(self, paper_ids: Sequence[str]) -> list[Paper]: ...


class MilvusPaperRepository:
    """Serve the collection named by the current index manifest."""

    def __init__(
        self,
        settings: Settings | None = None,
        encoder: TextEncoder | None = None,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._encoder = encoder or DashScopeEncoder(self._settings)
        self._client_factory = client_factory
        self._local = threading.local()
        self._clients: dict[int, Any] = {}
        self._clients_lock = threading.Lock()
        self._search_limit = threading.BoundedSemaphore(
            self._settings.milvus_concurrency
        )
        self._manifest: IndexManifest | None = None
        self._manifest_lock = threading.Lock()

    @property
    def manifest(self) -> IndexManifest:
        """Return the validated manifest loaded by this process."""
        if self._manifest is None:
            self.validate()
        assert self._manifest is not None
        return self._manifest

    def search(
        self, query: str, *, limit: int, filters: SearchFilters
    ) -> list[SearchHit]:
        """Run one native Milvus dense plus BM25 search with standard RRF."""
        if limit <= 0:
            return []
        manifest = self.manifest
        vector = self._encoder.encode_query(query)
        expression = _filter_expression(filters)
        request_options: dict[str, Any] = {"limit": limit}
        if expression:
            request_options["expr"] = expression
        requests = [
            AnnSearchRequest(
                [vector],
                DENSE_FIELD,
                {"metric_type": "COSINE", "params": {}},
                **request_options,
            ),
            AnnSearchRequest(
                [query],
                SPARSE_FIELD,
                {"metric_type": "BM25", "params": {}},
                **request_options,
            ),
        ]
        try:
            with self._search_limit:
                result = self._client().hybrid_search(
                    manifest.collection_name,
                    requests,
                    RRFRanker(),
                    limit,
                    output_fields=list(PAPER_OUTPUT_FIELDS),
                    timeout=self._settings.milvus_timeout_seconds,
                )
        except Exception as exc:
            raise IndexUnavailable() from exc
        hits = result[0] if result else []
        maximum = 2.0 / (RRF_K + 1)
        parsed: list[SearchHit] = []
        for hit in hits:
            entity = _hit_entity(hit)
            score = min(1.0, max(0.0, _hit_score(hit) / maximum))
            parsed.append(SearchHit(Paper.from_mapping(entity), score))
        return parsed

    def fetch(self, paper_ids: Sequence[str]) -> list[Paper]:
        """Fetch papers by ID while preserving caller order and removing duplicates."""
        wanted = list(dict.fromkeys(str(value) for value in paper_ids if str(value)))
        if not wanted:
            return []
        try:
            with self._search_limit:
                rows = self._client().get(
                    self.manifest.collection_name,
                    ids=wanted,
                    output_fields=list(PAPER_OUTPUT_FIELDS),
                    timeout=self._settings.milvus_timeout_seconds,
                )
        except Exception as exc:
            raise IndexUnavailable() from exc
        papers = [Paper.from_mapping(row) for row in rows]
        by_id = {paper.paper_id: paper for paper in papers}
        return [by_id[paper_id] for paper_id in wanted if paper_id in by_id]

    def validate(self) -> IndexManifest:
        """Validate the manifest and complete Milvus collection without cloud calls."""
        with self._manifest_lock:
            self._manifest = validate_current_index(self._settings, self._client())
            return self._manifest

    def close(self) -> None:
        """Close encoder and Milvus clients created by worker threads."""
        close_encoder = getattr(self._encoder, "close", None)
        if callable(close_encoder):
            close_encoder()
        with self._clients_lock:
            clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        self._local.client = None

    def _client(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            client = (
                self._client_factory()
                if self._client_factory is not None
                else MilvusClient(uri=self._settings.milvus_uri)
            )
            self._local.client = client
            with self._clients_lock:
                self._clients[id(client)] = client
        return client


def _filter_expression(filters: SearchFilters) -> str:
    clauses: list[str] = []
    if filters.area:
        clauses.append(f"area == {json.dumps(filters.area, ensure_ascii=False)}")
    if filters.paper_type:
        clauses.append(
            f"paper_type == {json.dumps(filters.paper_type, ensure_ascii=False)}"
        )
    if filters.spotlight_only:
        clauses.append("spotlight == true")
    return " and ".join(clauses)


def _hit_entity(hit: Any) -> Mapping[str, Any]:
    if isinstance(hit, Mapping):
        entity = hit.get("entity")
    else:
        entity = getattr(hit, "entity", None)
    if not isinstance(entity, Mapping):
        raise IndexUnavailable()
    return entity


def _hit_score(hit: Any) -> float:
    value = (
        hit.get("distance")
        if isinstance(hit, Mapping)
        else getattr(hit, "distance", None)
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndexUnavailable()
    return float(value)
