"""版本化本地索引的 manifest、校验与原子切换。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bm25 import BM25Index, BM25_TOKENIZER
from .catalog import (
    corpus_hash as calculate_corpus_hash,
    data_file_hashes as calculate_data_file_hashes,
    load_records,
    paper_ids_hash as calculate_paper_ids_hash,
)
from .config import Settings
from .errors import IndexUnavailable

INDEX_SCHEMA_VERSION = 1
_BUILD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """一个不可拆分 Chroma/FTS5 索引版本的契约。"""

    schema_version: int
    build_id: str
    corpus_hash: str
    paper_ids_hash: str
    record_count: int
    data_file_hashes: dict[str, str]
    embedding_model: str
    embedding_dim: int
    document_text_type: str
    dense_metric: str
    bm25_tokenizer: str
    dense_weight: float
    bm25_weight: float
    rrf_k: int
    collection_name: str
    built_at: str

    @classmethod
    def create(
        cls,
        *,
        build_id: str,
        corpus_hash: str,
        paper_ids_hash: str,
        record_count: int,
        data_file_hashes: dict[str, str],
        settings: Settings,
        collection_name: str,
    ) -> IndexManifest:
        """按当前固定协议创建 manifest。"""
        return cls(
            schema_version=INDEX_SCHEMA_VERSION,
            build_id=build_id,
            corpus_hash=corpus_hash,
            paper_ids_hash=paper_ids_hash,
            record_count=record_count,
            data_file_hashes=dict(sorted(data_file_hashes.items())),
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            document_text_type="document",
            dense_metric="cosine",
            bm25_tokenizer=BM25_TOKENIZER,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
            rrf_k=settings.rrf_k,
            collection_name=collection_name,
            built_at=datetime.now(UTC).isoformat(),
        )


def collection_metadata_for(manifest: IndexManifest) -> dict[str, str | int]:
    """生成 Chroma collection 必须持久化的索引契约。"""
    return {
        "hnsw:space": manifest.dense_metric,
        "index_schema_version": manifest.schema_version,
        "corpus_hash": manifest.corpus_hash,
        "paper_ids_hash": manifest.paper_ids_hash,
        "embedding_model": manifest.embedding_model,
        "embedding_dim": manifest.embedding_dim,
        "dense_metric": manifest.dense_metric,
    }


@dataclass(frozen=True, slots=True)
class IndexPaths:
    """索引根目录及其版本路径约定。"""

    root: Path
    versions: Path
    current: Path

    @classmethod
    def for_root(cls, root: Path) -> IndexPaths:
        """从配置根目录构造路径集合。"""
        root = Path(root)
        return cls(root=root, versions=root / "versions", current=root / "current")

    def version(self, build_id: str) -> Path:
        """返回已完成版本目录。"""
        _validate_build_id(build_id)
        return self.versions / build_id

    def staging(self, build_id: str) -> Path:
        """返回可恢复 staging 目录。"""
        _validate_build_id(build_id)
        return self.versions / f"{build_id}.staging"


def collection_name_for(build_id: str) -> str:
    """生成仅含 Chroma 允许字符的稳定 collection 名。"""
    _validate_build_id(build_id)
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", build_id).strip("-_")
    return f"icml-papers-{normalized}"[:63]


def load_manifest(version_path: Path) -> IndexManifest:
    """从版本目录或 manifest 文件读取并校验结构。"""
    path = Path(version_path)
    if path.is_dir():
        path = path / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = IndexManifest(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IndexUnavailable() from exc
    _validate_manifest_fields(manifest)
    return manifest


def write_manifest(version_path: Path, manifest: IndexManifest) -> Path:
    """原子写入规范 JSON manifest。"""
    directory = Path(version_path)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "manifest.json"
    temporary = directory / ".manifest.json.tmp"
    payload = json.dumps(
        asdict(manifest), ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, target)
    _fsync_directory(directory)
    return target


def resolve_current(root: Path) -> Path:
    """安全解析 current 文本指针并返回存在的版本目录。"""
    paths = IndexPaths.for_root(root)
    try:
        build_id = paths.current.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise IndexUnavailable() from exc
    _validate_build_id(build_id)
    version = paths.version(build_id)
    if not version.is_dir():
        raise IndexUnavailable()
    return version


def validate_index_version(
    version_path: Path,
    *,
    settings: Settings | None = None,
    expected_corpus_hash: str | None = None,
) -> IndexManifest:
    """零云调用校验 manifest、Chroma 与 BM25 的一致性。"""
    path = Path(version_path)
    manifest = load_manifest(path)
    if expected_corpus_hash is not None and manifest.corpus_hash != expected_corpus_hash:
        raise IndexUnavailable()
    if settings is not None:
        expected = (
            manifest.embedding_model == settings.embedding_model
            and manifest.embedding_dim == settings.embedding_dim
            and abs(manifest.dense_weight - settings.dense_weight) <= 1e-12
            and abs(manifest.bm25_weight - settings.bm25_weight) <= 1e-12
            and manifest.rrf_k == settings.rrf_k
        )
        if not expected:
            raise IndexUnavailable()
        # 开发环境与部署镜像都必须携带原始语料快照；readiness 不允许
        # 在缺文件时退化为只相信 index 内部的自声明指纹。
        try:
            current_data_hashes = calculate_data_file_hashes(settings)
            source_records = load_records(settings)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IndexUnavailable() from exc
        if (
            current_data_hashes != manifest.data_file_hashes
            or calculate_corpus_hash(source_records) != manifest.corpus_hash
            or calculate_paper_ids_hash(
                record.paper_id for record in source_records
            )
            != manifest.paper_ids_hash
        ):
            raise IndexUnavailable()

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(path / "chroma"))
        collection = client.get_collection(
            name=manifest.collection_name, embedding_function=None
        )
        _validate_collection_metadata(collection.metadata, manifest)
        chroma_count = int(collection.count())
        id_result = collection.get(include=[])
        collection_ids = id_result.get("ids") or []
        if calculate_paper_ids_hash(str(value) for value in collection_ids) != (
            manifest.paper_ids_hash
        ):
            raise IndexUnavailable()
        if manifest.record_count:
            sample = collection.get(
                ids=[str(collection_ids[0])], include=["embeddings"]
            )
            sample_ids = sample.get("ids") or []
            embeddings = sample.get("embeddings")
            if len(sample_ids) != 1 or embeddings is None or len(embeddings) != 1:
                raise IndexUnavailable()
            if len(embeddings[0]) != manifest.embedding_dim:
                raise IndexUnavailable()
        bm25 = BM25Index(path / "bm25.sqlite3")
        try:
            bm25_count = bm25.count()
            bm25_hash = bm25.corpus_hash()
        finally:
            bm25.close()
    except IndexUnavailable:
        raise
    except Exception as exc:
        raise IndexUnavailable() from exc

    if (
        chroma_count != manifest.record_count
        or len(collection_ids) != manifest.record_count
        or bm25_count != manifest.record_count
        or bm25_hash != manifest.corpus_hash
    ):
        raise IndexUnavailable()
    return manifest


def promote_staging(root: Path, build_id: str) -> Path:
    """持久化 staging 后原子发布版本并切换 current 指针。"""
    paths = IndexPaths.for_root(root)
    staging = paths.staging(build_id)
    final = paths.version(build_id)
    if not staging.is_dir() or final.exists():
        raise IndexUnavailable()
    paths.versions.mkdir(parents=True, exist_ok=True)
    _fsync_tree(staging)
    os.replace(staging, final)
    _fsync_directory(paths.versions)

    temporary = paths.root / ".current.tmp"
    paths.root.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as fh:
        fh.write(f"{build_id}\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, paths.current)
    _fsync_directory(paths.root)
    return final


def _validate_manifest_fields(manifest: IndexManifest) -> None:
    """拒绝不符合当前索引协议的 manifest。"""
    _validate_build_id(manifest.build_id)
    valid = (
        manifest.schema_version == INDEX_SCHEMA_VERSION
        and bool(re.fullmatch(r"[0-9a-f]{64}", manifest.corpus_hash))
        and bool(re.fullmatch(r"[0-9a-f]{64}", manifest.paper_ids_hash))
        and manifest.record_count >= 0
        and set(manifest.data_file_hashes) == {"index.json", "abstracts.json"}
        and all(
            bool(re.fullmatch(r"[0-9a-f]{64}", value))
            for value in manifest.data_file_hashes.values()
        )
        and manifest.embedding_dim > 0
        and manifest.document_text_type == "document"
        and manifest.dense_metric == "cosine"
        and manifest.bm25_tokenizer == BM25_TOKENIZER
        and manifest.rrf_k > 0
        and manifest.dense_weight >= 0
        and manifest.bm25_weight >= 0
        and abs(manifest.dense_weight + manifest.bm25_weight - 1.0) <= 1e-9
        and bool(manifest.collection_name)
    )
    if not valid:
        raise IndexUnavailable()


def _validate_collection_metadata(
    metadata: Any, manifest: IndexManifest
) -> None:
    """核对 Chroma 自身保存的 schema、语料与 embedding 契约。"""
    if not isinstance(metadata, dict):
        raise IndexUnavailable()
    expected = collection_metadata_for(manifest)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise IndexUnavailable()


def _validate_build_id(build_id: str) -> None:
    """防止 current 指针逃逸索引根目录。"""
    if not isinstance(build_id, str) or not _BUILD_ID_RE.fullmatch(build_id):
        raise IndexUnavailable()


def _fsync_tree(root: Path) -> None:
    """发布前持久化 staging 中的普通文件与目录。"""
    for directory, _subdirs, files in os.walk(root, topdown=False):
        path = Path(directory)
        for name in files:
            with (path / name).open("rb") as fh:
                os.fsync(fh.fileno())
        _fsync_directory(path)


def _fsync_directory(path: Path) -> None:
    """持久化目录项更新。"""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
