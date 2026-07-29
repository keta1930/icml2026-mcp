"""Current Milvus index manifest and readiness validation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Settings
from ..corpus.catalog import (
    corpus_hash,
    data_file_hashes,
    load_records,
    paper_ids_hash,
)
from ..domain.errors import IndexUnavailable
from ..milvus.schema import PRIMARY_FIELD, validate_collection_schema

MANIFEST_FORMAT = "milvus-dense-bm25-rrf"
MILVUS_VERSION = "2.5.27"
PYMILVUS_VERSION = "2.5.18"
_BUILD_ID = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Complete contract for the collection currently served."""

    format: str
    build_id: str
    collection_name: str
    corpus_hash: str
    paper_ids_hash: str
    record_count: int
    data_file_hashes: dict[str, str]
    embedding_model: str
    embedding_dim: int
    milvus_version: str
    pymilvus_version: str
    dense_metric: str
    sparse_metric: str
    fusion: str
    built_at: str

    @classmethod
    def create(
        cls,
        *,
        build_id: str,
        collection_name: str,
        corpus_digest: str,
        ids_digest: str,
        record_count: int,
        source_hashes: dict[str, str],
        settings: Settings,
    ) -> IndexManifest:
        """Create the manifest for a completed Milvus build."""
        return cls(
            format=MANIFEST_FORMAT,
            build_id=build_id,
            collection_name=collection_name,
            corpus_hash=corpus_digest,
            paper_ids_hash=ids_digest,
            record_count=record_count,
            data_file_hashes=dict(sorted(source_hashes.items())),
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            milvus_version=MILVUS_VERSION,
            pymilvus_version=PYMILVUS_VERSION,
            dense_metric="COSINE",
            sparse_metric="BM25",
            fusion="RRF",
            built_at=datetime.now(UTC).isoformat(),
        )


def current_manifest_path(index_root: Path) -> Path:
    """Return the only manifest path read by the service."""
    return Path(index_root) / "current.json"


def load_current_manifest(index_root: Path) -> IndexManifest:
    """Load and validate the current manifest."""
    path = current_manifest_path(index_root)
    try:
        manifest = IndexManifest(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IndexUnavailable() from exc
    _validate_manifest(manifest)
    return manifest


def publish_manifest(index_root: Path, manifest: IndexManifest) -> Path:
    """Atomically publish a completed collection as the current index."""
    _validate_manifest(manifest)
    root = Path(index_root)
    root.mkdir(parents=True, exist_ok=True)
    target = current_manifest_path(root)
    temporary = root / ".current.json.tmp"
    payload = json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True, indent=2)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def validate_current_index(settings: Settings, client: Any) -> IndexManifest:
    """Validate source data, manifest, collection schema, count, and complete ID set."""
    manifest = load_current_manifest(settings.index_root)
    records = load_records(settings)
    expected = (
        manifest.embedding_model == settings.embedding_model
        and manifest.embedding_dim == settings.embedding_dim
        and manifest.data_file_hashes == data_file_hashes(settings)
        and manifest.corpus_hash == corpus_hash(records)
        and manifest.paper_ids_hash
        == paper_ids_hash(record.paper_id for record in records)
        and manifest.record_count == len(records)
    )
    if not expected:
        raise IndexUnavailable()

    try:
        timeout = settings.milvus_timeout_seconds
        if not client.has_collection(manifest.collection_name, timeout=timeout):
            raise IndexUnavailable()
        validate_collection_schema(
            client,
            manifest.collection_name,
            settings.embedding_dim,
            timeout=timeout,
        )
        stats = client.get_collection_stats(manifest.collection_name, timeout=timeout)
        count = int(stats.get("row_count", -1))
        rows = client.query(
            manifest.collection_name,
            filter="",
            output_fields=[PRIMARY_FIELD],
            limit=manifest.record_count + 1,
            timeout=timeout,
        )
        ids = [str(row[PRIMARY_FIELD]) for row in rows]
    except IndexUnavailable:
        raise
    except Exception as exc:
        raise IndexUnavailable() from exc
    if count != manifest.record_count or paper_ids_hash(ids) != manifest.paper_ids_hash:
        raise IndexUnavailable()
    return manifest


def _validate_manifest(manifest: IndexManifest) -> None:
    valid = (
        manifest.format == MANIFEST_FORMAT
        and bool(_BUILD_ID.fullmatch(manifest.build_id))
        and bool(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", manifest.collection_name)
        )
        and bool(re.fullmatch(r"[0-9a-f]{64}", manifest.corpus_hash))
        and bool(re.fullmatch(r"[0-9a-f]{64}", manifest.paper_ids_hash))
        and manifest.record_count > 0
        and set(manifest.data_file_hashes) == {"index.json", "abstracts.json"}
        and all(
            bool(re.fullmatch(r"[0-9a-f]{64}", value))
            for value in manifest.data_file_hashes.values()
        )
        and manifest.embedding_dim > 0
        and manifest.milvus_version == MILVUS_VERSION
        and manifest.pymilvus_version == PYMILVUS_VERSION
        and manifest.dense_metric == "COSINE"
        and manifest.sparse_metric == "BM25"
        and manifest.fusion == "RRF"
    )
    if not valid:
        raise IndexUnavailable()
