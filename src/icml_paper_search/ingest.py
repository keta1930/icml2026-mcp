"""显式、可恢复地构建 Chroma + FTS5 版本化索引。"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bm25 import BM25Index, BM25IndexBuilder
from .catalog import (
    PaperRecord,
    corpus_hash,
    data_file_hashes,
    load_records,
    paper_ids_hash,
)
from .config import Settings, get_settings
from .embedding import DashScopeEncoder, EmbeddingBatch, TextEncoder
from .errors import BudgetExhausted, IndexUnavailable
from .manifest import (
    IndexManifest,
    IndexPaths,
    collection_metadata_for,
    collection_name_for,
    promote_staging,
    validate_index_version,
    write_manifest,
)
from .models import SearchFilters

_TOKEN_OVERHEAD_PER_INPUT = 32
_BUILD_FAILURE_AUDIT = "failure-audit.json"


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """不读取 key、不访问网络的建库预检结果。"""

    build_id: str
    record_count: int
    corpus_hash: str
    data_file_hashes: dict[str, str]
    estimated_input_tokens: int
    staging_path: Path


@dataclass(frozen=True, slots=True)
class BuildCheckpoint:
    """仅在完整批次提交后前移的恢复点。"""

    build_id: str
    corpus_hash: str
    completed_ids: tuple[str, ...]
    total_tokens: int
    embedding_model: str
    embedding_dim: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class BuildResult:
    """建库命令的预检或执行结果。"""

    plan: BuildPlan
    executed: bool
    resumed: bool
    total_tokens: int
    version_path: Path | None


def plan_build(settings: Settings | None = None) -> BuildPlan:
    """只读取本地语料并给出稳定 build id 与费用上界估算。"""
    cfg = settings or get_settings()
    records = load_records(cfg)
    digest = corpus_hash(records)
    build_id = f"v1-{digest[:16]}-{cfg.embedding_dim}d"
    return BuildPlan(
        build_id=build_id,
        record_count=len(records),
        corpus_hash=digest,
        data_file_hashes=data_file_hashes(cfg),
        estimated_input_tokens=sum(_estimate_tokens(record.text) for record in records),
        staging_path=IndexPaths.for_root(cfg.index_root).staging(build_id),
    )


def build_index(
    settings: Settings | None = None,
    encoder: TextEncoder | None = None,
    *,
    max_input_tokens: int | None,
    execute: bool = False,
    resume: bool = False,
) -> BuildResult:
    """在 staging 构建并完整校验后原子发布 current。"""
    cfg = settings or get_settings()
    plan = plan_build(cfg)
    if not execute:
        return BuildResult(
            plan=plan,
            executed=False,
            resumed=False,
            total_tokens=0,
            version_path=None,
        )
    if max_input_tokens is None or max_input_tokens <= 0:
        raise ValueError("执行建库必须设置正数 max_input_tokens")

    records = load_records(cfg)
    paths = IndexPaths.for_root(cfg.index_root)
    staging = paths.staging(plan.build_id)
    final = paths.version(plan.build_id)
    if final.exists():
        raise IndexUnavailable()
    if resume:
        if not staging.is_dir():
            raise IndexUnavailable()
    else:
        if staging.exists():
            raise IndexUnavailable()
        staging.mkdir(parents=True)

    checkpoint_path = staging / "checkpoint.json"
    checkpoint = (
        _load_checkpoint(checkpoint_path, plan=plan, settings=cfg)
        if resume
        else _new_checkpoint(plan, cfg)
    )
    if not resume:
        _save_checkpoint(checkpoint_path, checkpoint)
    completed_count = len(checkpoint.completed_ids)
    expected_prefix = tuple(record.paper_id for record in records[:completed_count])
    if checkpoint.completed_ids != expected_prefix:
        raise IndexUnavailable()

    provider = encoder or DashScopeEncoder(cfg)
    collection_name = collection_name_for(plan.build_id)
    manifest = IndexManifest.create(
        build_id=plan.build_id,
        corpus_hash=plan.corpus_hash,
        paper_ids_hash=paper_ids_hash(record.paper_id for record in records),
        record_count=plan.record_count,
        data_file_hashes=plan.data_file_hashes,
        settings=cfg,
        collection_name=collection_name,
    )
    failure_audit_path = staging / _BUILD_FAILURE_AUDIT
    if resume and failure_audit_path.exists():
        # provider usage 违反预留契约后不能继续自动花费；需删除 staging 重建。
        raise IndexUnavailable()
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(staging / "chroma"))
        expected_collection_metadata = collection_metadata_for(manifest)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata=expected_collection_metadata,
            embedding_function=None,
        )
        actual_collection_metadata = collection.metadata or {}
        if any(
            actual_collection_metadata.get(key) != value
            for key, value in expected_collection_metadata.items()
        ):
            raise IndexUnavailable()
    except Exception as exc:
        raise IndexUnavailable() from exc

    builder = BM25IndexBuilder(staging / "bm25.sqlite3", resume=resume)
    total_tokens = checkpoint.total_tokens
    try:
        start = completed_count
        while start < len(records):
            remaining_tokens = max_input_tokens - total_tokens
            batch, reserved_tokens = _select_batch_within_budget(
                records,
                start=start,
                batch_size=cfg.embedding_batch_size,
                remaining_tokens=remaining_tokens,
            )
            if not batch:
                raise BudgetExhausted()

            encoded = _encode_documents_with_usage(provider, batch)
            if (
                encoded.total_tokens > reserved_tokens
                or total_tokens + encoded.total_tokens > max_input_tokens
            ):
                _save_failure_audit(
                    failure_audit_path,
                    plan=plan,
                    completed_count=start,
                    attempted_ids=(record.paper_id for record in batch),
                    reserved_tokens=reserved_tokens,
                    actual_tokens=encoded.total_tokens,
                    total_tokens_before=total_tokens,
                    hard_cap=max_input_tokens,
                )
                raise BudgetExhausted()
            _validate_vectors(
                encoded,
                expected_count=len(batch),
                dimension=cfg.embedding_dim,
            )
            total_tokens += encoded.total_tokens
            _upsert_chroma(collection, batch, encoded)
            builder.add_records(batch)
            start += len(batch)

            checkpoint = BuildCheckpoint(
                build_id=plan.build_id,
                corpus_hash=plan.corpus_hash,
                completed_ids=tuple(record.paper_id for record in records[:start]),
                total_tokens=total_tokens,
                embedding_model=cfg.embedding_model,
                embedding_dim=cfg.embedding_dim,
                batch_size=cfg.embedding_batch_size,
            )
            _save_checkpoint(checkpoint_path, checkpoint)

        builder.finalize(corpus_hash=plan.corpus_hash)
    finally:
        builder.close()

    write_manifest(staging, manifest)
    _verify_staging(staging, records=records, manifest=manifest, settings=cfg)
    checkpoint_path.unlink(missing_ok=True)
    version_path = promote_staging(cfg.index_root, plan.build_id)
    return BuildResult(
        plan=plan,
        executed=True,
        resumed=resume,
        total_tokens=total_tokens,
        version_path=version_path,
    )


def _encode_documents_with_usage(
    encoder: TextEncoder, records: list[PaperRecord]
) -> EmbeddingBatch:
    """优先使用 usage-aware 接口，测试替身则保守估算。"""
    texts = [record.text for record in records]
    method = getattr(encoder, "encode_documents_with_usage", None)
    if callable(method):
        return method(texts, show_progress=False)
    vectors = encoder.encode_documents(texts, show_progress=False)
    return EmbeddingBatch(
        vectors=tuple(tuple(float(value) for value in vector) for vector in vectors),
        total_tokens=sum(_estimate_tokens(text) for text in texts),
    )


def _upsert_chroma(
    collection: Any, records: list[PaperRecord], encoded: EmbeddingBatch
) -> None:
    """显式传入 embedding，禁止 Chroma 调用隐式模型。"""
    collection.upsert(
        ids=[record.paper_id for record in records],
        embeddings=[list(vector) for vector in encoded.vectors],
        documents=[record.text for record in records],
        metadatas=[dict(record.fields) for record in records],
    )


def _verify_staging(
    staging: Path,
    *,
    records: list[PaperRecord],
    manifest: IndexManifest,
    settings: Settings,
) -> None:
    """校验 count、维度、fetch、过滤和两路基本查询。"""
    validate_index_version(
        staging,
        settings=settings,
        expected_corpus_hash=manifest.corpus_hash,
    )
    if not records:
        return
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(staging / "chroma"))
        collection = client.get_collection(
            name=manifest.collection_name, embedding_function=None
        )
        sample = collection.get(
            ids=[records[0].paper_id], include=["embeddings", "metadatas"]
        )
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) != 1:
            raise IndexUnavailable()
        vector = embeddings[0]
        if len(vector) != settings.embedding_dim:
            raise IndexUnavailable()
        area = str(records[0].fields.get("area", ""))
        where = {"area": {"$eq": area}} if area else None
        dense = collection.query(
            query_embeddings=[list(vector)],
            n_results=1,
            where=where,
            include=["distances"],
        )
        if not (dense.get("ids") or [[]])[0]:
            raise IndexUnavailable()
        bm25 = BM25Index(staging / "bm25.sqlite3")
        try:
            title = str(records[0].fields.get("title", ""))
            if title and not bm25.search(
                title, limit=1, filters=SearchFilters(area=area or None)
            ):
                raise IndexUnavailable()
        finally:
            bm25.close()
    except IndexUnavailable:
        raise
    except Exception as exc:
        raise IndexUnavailable() from exc


def _validate_vectors(
    encoded: EmbeddingBatch, *, expected_count: int, dimension: int
) -> None:
    """在写盘前验证批次数量与维度。"""
    if len(encoded.vectors) != expected_count or encoded.total_tokens < 0:
        raise IndexUnavailable()
    if any(len(vector) != dimension for vector in encoded.vectors):
        raise IndexUnavailable()


def _new_checkpoint(plan: BuildPlan, settings: Settings) -> BuildCheckpoint:
    """创建空恢复点。"""
    return BuildCheckpoint(
        build_id=plan.build_id,
        corpus_hash=plan.corpus_hash,
        completed_ids=(),
        total_tokens=0,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )


def _load_checkpoint(
    path: Path, *, plan: BuildPlan, settings: Settings
) -> BuildCheckpoint:
    """读取并核对 staging 恢复点。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["completed_ids"] = tuple(payload["completed_ids"])
        checkpoint = BuildCheckpoint(**payload)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise IndexUnavailable() from exc
    valid = (
        checkpoint.build_id == plan.build_id
        and checkpoint.corpus_hash == plan.corpus_hash
        and checkpoint.embedding_model == settings.embedding_model
        and checkpoint.embedding_dim == settings.embedding_dim
        and checkpoint.batch_size == settings.embedding_batch_size
        and checkpoint.total_tokens >= 0
    )
    if not valid:
        raise IndexUnavailable()
    return checkpoint


def _save_checkpoint(path: Path, checkpoint: BuildCheckpoint) -> None:
    """批次提交后原子前移 checkpoint。"""
    temporary = path.with_name(".checkpoint.json.tmp")
    payload = json.dumps(asdict(checkpoint), ensure_ascii=False, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _estimate_tokens(text: str) -> int:
    """按 UTF-8 bytes 与固定输入开销给出 provider Token 上界。"""
    # qwen embedding 的 input usage 以 tokenizer token 计；每个普通 token 至少
    # 消耗一个 UTF-8 byte，额外协议 token 由每条 32-token 余量覆盖。
    return max(1, len(text.encode("utf-8"))) + _TOKEN_OVERHEAD_PER_INPUT


def _select_batch_within_budget(
    records: list[PaperRecord],
    *,
    start: int,
    batch_size: int,
    remaining_tokens: int,
) -> tuple[list[PaperRecord], int]:
    """按严格语料前缀缩小最后一批，确保调用前预留不越界。"""
    batch: list[PaperRecord] = []
    reserved_tokens = 0
    for record in records[start : start + batch_size]:
        record_reservation = _estimate_tokens(record.text)
        if reserved_tokens + record_reservation > remaining_tokens:
            break
        batch.append(record)
        reserved_tokens += record_reservation
    return batch, reserved_tokens


def _save_failure_audit(
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
    """记录 provider usage 违反预留契约的脱敏、不可恢复审计。"""
    payload = {
        "reason": "provider_usage_exceeded_reservation",
        "build_id": plan.build_id,
        "corpus_hash": plan.corpus_hash,
        "completed_count": completed_count,
        "attempted_ids_hash": paper_ids_hash(attempted_ids),
        "reserved_tokens": reserved_tokens,
        "actual_tokens": actual_tokens,
        "total_tokens_before": total_tokens_before,
        "hard_cap": hard_cap,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
