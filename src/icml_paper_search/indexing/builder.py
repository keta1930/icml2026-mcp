"""Build the current Milvus dense plus BM25 paper collection."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from ..config import Settings, get_settings
from ..corpus.catalog import (
    PaperRecord,
    corpus_hash,
    data_file_hashes,
    load_records,
    paper_ids_hash,
)
from ..domain.errors import IndexBuildLimitExceeded, IndexUnavailable
from ..embeddings.dashscope import DashScopeEncoder, EmbeddingBatch, TextEncoder
from ..milvus.schema import (
    DENSE_FIELD,
    PAPER_OUTPUT_FIELDS,
    SPARSE_FIELD,
    collection_name,
    create_collection,
    validate_collection_schema,
)
from .manifest import (
    MANIFEST_FORMAT,
    IndexManifest,
    current_manifest_path,
    load_current_manifest,
    publish_manifest,
    validate_current_index,
)

_TOKEN_OVERHEAD_PER_INPUT = 32
_CHECKPOINT_NAME = "build-checkpoint.json"
_FAILURE_NAME = "build-failure.json"


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Offline preflight result."""

    build_id: str
    collection_name: str
    record_count: int
    corpus_hash: str
    data_file_hashes: dict[str, str]
    estimated_input_tokens: int
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class BuildCheckpoint:
    """Progress recorded after each complete Milvus insert."""

    build_id: str
    completed_ids: tuple[str, ...]
    total_tokens: int
    embedding_model: str
    embedding_dim: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Result of preflight or an executed build."""

    plan: BuildPlan
    executed: bool
    resumed: bool
    reused: bool
    total_tokens: int
    manifest_path: Path | None


def plan_build(settings: Settings | None = None) -> BuildPlan:
    """Read the local corpus and calculate a deterministic Milvus build."""
    cfg = settings or get_settings()
    records = load_records(cfg)
    digest = corpus_hash(records)
    identity = hashlib.sha256(
        f"{MANIFEST_FORMAT}\0{digest}\0{cfg.embedding_model}\0{cfg.embedding_dim}".encode()
    ).hexdigest()[:16]
    return BuildPlan(
        build_id=identity,
        collection_name=collection_name(cfg.milvus_collection_prefix, identity),
        record_count=len(records),
        corpus_hash=digest,
        data_file_hashes=data_file_hashes(cfg),
        estimated_input_tokens=sum(
            _estimate_tokens(record.document) for record in records
        ),
        checkpoint_path=cfg.index_root / _CHECKPOINT_NAME,
    )


def build_index(
    settings: Settings | None = None,
    encoder: TextEncoder | None = None,
    *,
    max_input_tokens: int | None,
    execute: bool = False,
    resume: bool = False,
    client_factory: Callable[[], Any] | None = None,
) -> BuildResult:
    """Build, validate, and publish one Milvus collection."""
    cfg = settings or get_settings()
    plan = plan_build(cfg)
    if not execute:
        return BuildResult(plan, False, False, False, 0, None)
    if max_input_tokens is None or max_input_tokens <= 0:
        raise ValueError(
            "executing an index build requires a positive max_input_tokens value"
        )

    cfg.index_root.mkdir(parents=True, exist_ok=True)
    records = load_records(cfg)
    provider = encoder or DashScopeEncoder(cfg)
    client = (
        client_factory()
        if client_factory is not None
        else MilvusClient(uri=cfg.milvus_uri)
    )
    owns_client = client_factory is None
    failure_path = cfg.index_root / _FAILURE_NAME
    active_manifest: IndexManifest | None = None
    try:
        try:
            active_manifest = load_current_manifest(cfg.index_root)
        except IndexUnavailable:
            active_manifest = None
        if (
            active_manifest is not None
            and active_manifest.collection_name == plan.collection_name
        ):
            validate_current_index(cfg, client)
            return BuildResult(
                plan,
                True,
                False,
                True,
                0,
                current_manifest_path(cfg.index_root),
            )

        exists = client.has_collection(
            plan.collection_name, timeout=cfg.milvus_timeout_seconds
        )
        if resume:
            if (
                not exists
                or not plan.checkpoint_path.is_file()
                or failure_path.exists()
            ):
                raise IndexUnavailable()
            checkpoint = _load_checkpoint(plan.checkpoint_path, plan=plan, settings=cfg)
            validate_collection_schema(
                client,
                plan.collection_name,
                cfg.embedding_dim,
                timeout=cfg.milvus_timeout_seconds,
            )
            _validate_completed_ids(client, plan, checkpoint)
        else:
            if exists:
                client.drop_collection(
                    plan.collection_name, timeout=cfg.milvus_timeout_seconds
                )
            plan.checkpoint_path.unlink(missing_ok=True)
            failure_path.unlink(missing_ok=True)
            create_collection(
                client,
                plan.collection_name,
                cfg.embedding_dim,
                timeout=cfg.milvus_timeout_seconds,
            )
            checkpoint = BuildCheckpoint(
                build_id=plan.build_id,
                completed_ids=(),
                total_tokens=0,
                embedding_model=cfg.embedding_model,
                embedding_dim=cfg.embedding_dim,
                batch_size=cfg.embedding_batch_size,
            )
            _save_checkpoint(plan.checkpoint_path, checkpoint)

        start = len(checkpoint.completed_ids)
        total_tokens = checkpoint.total_tokens
        while start < len(records):
            batch, reserved_tokens = _select_batch_within_limit(
                records,
                start=start,
                batch_size=cfg.embedding_batch_size,
                remaining_tokens=max_input_tokens - total_tokens,
            )
            if not batch:
                raise IndexBuildLimitExceeded()
            encoded = _encode_documents_with_usage(provider, batch)
            _validate_vectors(
                encoded, expected_count=len(batch), dimension=cfg.embedding_dim
            )
            if (
                encoded.total_tokens > reserved_tokens
                or total_tokens + encoded.total_tokens > max_input_tokens
            ):
                _save_failure(
                    failure_path,
                    plan=plan,
                    completed_count=start,
                    attempted_ids=(record.paper_id for record in batch),
                    reserved_tokens=reserved_tokens,
                    actual_tokens=encoded.total_tokens,
                    total_tokens_before=total_tokens,
                    hard_cap=max_input_tokens,
                )
                raise IndexBuildLimitExceeded()

            entities = [
                record.milvus_entity(vector)
                for record, vector in zip(batch, encoded.vectors, strict=True)
            ]
            result = client.insert(
                plan.collection_name,
                entities,
                timeout=cfg.milvus_timeout_seconds,
            )
            if int(result.get("insert_count", -1)) != len(batch):
                raise IndexUnavailable()
            total_tokens += encoded.total_tokens
            start += len(batch)
            checkpoint = BuildCheckpoint(
                build_id=plan.build_id,
                completed_ids=tuple(record.paper_id for record in records[:start]),
                total_tokens=total_tokens,
                embedding_model=cfg.embedding_model,
                embedding_dim=cfg.embedding_dim,
                batch_size=cfg.embedding_batch_size,
            )
            _save_checkpoint(plan.checkpoint_path, checkpoint)

        client.flush(plan.collection_name, timeout=cfg.milvus_timeout_seconds)
        client.load_collection(plan.collection_name, timeout=cfg.milvus_timeout_seconds)
        _verify_collection(client, plan, records, cfg)
        manifest = IndexManifest.create(
            build_id=plan.build_id,
            collection_name=plan.collection_name,
            corpus_digest=plan.corpus_hash,
            ids_digest=paper_ids_hash(record.paper_id for record in records),
            record_count=len(records),
            source_hashes=plan.data_file_hashes,
            settings=cfg,
        )
        path = publish_manifest(cfg.index_root, manifest)
        plan.checkpoint_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        if (
            active_manifest is not None
            and active_manifest.collection_name != manifest.collection_name
            and client.has_collection(
                active_manifest.collection_name, timeout=cfg.milvus_timeout_seconds
            )
        ):
            client.drop_collection(
                active_manifest.collection_name,
                timeout=cfg.milvus_timeout_seconds,
            )
        return BuildResult(plan, True, resume, False, total_tokens, path)
    except (IndexBuildLimitExceeded, IndexUnavailable):
        raise
    except Exception as exc:
        raise IndexUnavailable() from exc
    finally:
        close_provider = getattr(provider, "close", None)
        if callable(close_provider):
            close_provider()
        if owns_client:
            close_client = getattr(client, "close", None)
            if callable(close_client):
                close_client()


def _verify_collection(
    client: Any,
    plan: BuildPlan,
    records: list[PaperRecord],
    settings: Settings,
) -> None:
    timeout = settings.milvus_timeout_seconds
    validate_collection_schema(
        client, plan.collection_name, settings.embedding_dim, timeout=timeout
    )
    stats = client.get_collection_stats(plan.collection_name, timeout=timeout)
    if int(stats.get("row_count", -1)) != len(records):
        raise IndexUnavailable()
    rows = client.query(
        plan.collection_name,
        filter="",
        output_fields=["paper_id"],
        limit=len(records) + 1,
        timeout=timeout,
    )
    if paper_ids_hash(str(row["paper_id"]) for row in rows) != paper_ids_hash(
        record.paper_id for record in records
    ):
        raise IndexUnavailable()
    sample = client.get(
        plan.collection_name,
        ids=[records[0].paper_id],
        output_fields=[DENSE_FIELD, "title"],
        timeout=timeout,
    )
    if (
        len(sample) != 1
        or len(sample[0].get(DENSE_FIELD, [])) != settings.embedding_dim
    ):
        raise IndexUnavailable()
    requests = [
        AnnSearchRequest(
            [sample[0][DENSE_FIELD]],
            DENSE_FIELD,
            {"metric_type": "COSINE", "params": {}},
            limit=1,
        ),
        AnnSearchRequest(
            [records[0].title],
            SPARSE_FIELD,
            {"metric_type": "BM25", "params": {}},
            limit=1,
        ),
    ]
    result = client.hybrid_search(
        plan.collection_name,
        requests,
        RRFRanker(),
        1,
        output_fields=list(PAPER_OUTPUT_FIELDS),
        timeout=timeout,
    )
    if not result or not result[0]:
        raise IndexUnavailable()


def _validate_completed_ids(
    client: Any, plan: BuildPlan, checkpoint: BuildCheckpoint
) -> None:
    rows = client.query(
        plan.collection_name,
        filter="",
        output_fields=["paper_id"],
        limit=plan.record_count + 1,
    )
    actual = {str(row["paper_id"]) for row in rows}
    if actual != set(checkpoint.completed_ids):
        raise IndexUnavailable()


def _encode_documents_with_usage(
    encoder: TextEncoder, records: list[PaperRecord]
) -> EmbeddingBatch:
    texts = [record.document for record in records]
    method = getattr(encoder, "encode_documents_with_usage", None)
    if callable(method):
        return method(texts, show_progress=False)
    vectors = encoder.encode_documents(texts, show_progress=False)
    return EmbeddingBatch(
        vectors=tuple(tuple(float(value) for value in vector) for vector in vectors),
        total_tokens=sum(_estimate_tokens(text) for text in texts),
    )


def _validate_vectors(
    encoded: EmbeddingBatch, *, expected_count: int, dimension: int
) -> None:
    if len(encoded.vectors) != expected_count or encoded.total_tokens < 0:
        raise IndexUnavailable()
    if any(len(vector) != dimension for vector in encoded.vectors):
        raise IndexUnavailable()


def _load_checkpoint(
    path: Path, *, plan: BuildPlan, settings: Settings
) -> BuildCheckpoint:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["completed_ids"] = tuple(payload["completed_ids"])
        checkpoint = BuildCheckpoint(**payload)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise IndexUnavailable() from exc
    valid = (
        checkpoint.build_id == plan.build_id
        and checkpoint.embedding_model == settings.embedding_model
        and checkpoint.embedding_dim == settings.embedding_dim
        and checkpoint.batch_size == settings.embedding_batch_size
        and checkpoint.total_tokens >= 0
    )
    if not valid:
        raise IndexUnavailable()
    return checkpoint


def _save_checkpoint(path: Path, checkpoint: BuildCheckpoint) -> None:
    _atomic_json(path, asdict(checkpoint))


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8"))) + _TOKEN_OVERHEAD_PER_INPUT


def _select_batch_within_limit(
    records: list[PaperRecord],
    *,
    start: int,
    batch_size: int,
    remaining_tokens: int,
) -> tuple[list[PaperRecord], int]:
    batch: list[PaperRecord] = []
    reserved_tokens = 0
    for record in records[start : start + batch_size]:
        reservation = _estimate_tokens(record.document)
        if reserved_tokens + reservation > remaining_tokens:
            break
        batch.append(record)
        reserved_tokens += reservation
    return batch, reserved_tokens


def _save_failure(
    path: Path,
    *,
    plan: BuildPlan,
    completed_count: int,
    attempted_ids: Iterable[str],
    reserved_tokens: int,
    actual_tokens: int,
    total_tokens_before: int,
    hard_cap: int,
) -> None:
    _atomic_json(
        path,
        {
            "reason": "provider_usage_exceeded_build_limit",
            "build_id": plan.build_id,
            "completed_count": completed_count,
            "attempted_ids_hash": paper_ids_hash(attempted_ids),
            "reserved_tokens": reserved_tokens,
            "actual_tokens": actual_tokens,
            "total_tokens_before": total_tokens_before,
            "hard_cap": hard_cap,
        },
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
