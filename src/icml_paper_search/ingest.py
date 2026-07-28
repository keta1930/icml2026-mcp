"""建库：把 ICML 2026 全量论文写入 Milvus，dense 向量 + BM25 sparse。

- dense：jina-embeddings-v5-text-small 对 title+abstract 做 retrieval embedding
  （1024 维，FLAT/COSINE 暴力搜索，6k 量级语料无需 HNSW）
- sparse：Milvus 内置 BM25 Function 从 text 字段自动生成，insert 时不用提供
"""

from __future__ import annotations

import logging
import time

from .catalog import load_records
from .config import Settings, get_settings
from .embedding import JinaEncoder, TextEncoder

logger = logging.getLogger(__name__)


def build_schema(settings: Settings):
    """collection schema，含 BM25 Function 与 dense 向量字段。"""
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("orid", DataType.VARCHAR, is_primary=True, max_length=32)
    schema.add_field("i", DataType.INT64)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("abstract", DataType.VARCHAR, max_length=4096)
    # text = title + abstract，交给 analyzer 分词供 BM25 使用
    schema.add_field("text", DataType.VARCHAR, max_length=4096, enable_analyzer=True)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
    schema.add_field("authors", DataType.VARCHAR, max_length=2048)
    schema.add_field("area", DataType.VARCHAR, max_length=64)
    schema.add_field("sub", DataType.VARCHAR, max_length=128)
    schema.add_field("type", DataType.VARCHAR, max_length=16)
    schema.add_field("spot", DataType.BOOL)
    schema.add_field("pid", DataType.VARCHAR, max_length=16)
    schema.add_field("arxiv", DataType.VARCHAR, max_length=32)
    schema.add_function(
        Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )
    return schema


def build_index_params():
    """sparse 走 BM25 倒排，dense 走 FLAT 暴力搜索。"""
    from pymilvus import MilvusClient

    params = MilvusClient.prepare_index_params()
    params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    params.add_index("dense", index_type="FLAT", metric_type="COSINE")
    return params


def build_collection(
    settings: Settings | None = None,
    encoder: TextEncoder | None = None,
    *,
    drop_existing: bool = True,
) -> int:
    """全量重建 collection，返回写入行数。

    幂等：`drop_existing=True` 时已存在的 collection 会先 drop 再重建。
    """
    from pymilvus import MilvusClient

    cfg = settings or get_settings()
    encoder = encoder or JinaEncoder(cfg)
    client = MilvusClient(uri=cfg.milvus_uri, token=cfg.milvus_token or None)

    if client.has_collection(cfg.collection):
        if not drop_existing:
            raise RuntimeError(
                f"collection {cfg.collection} 已存在；用 drop_existing=True 覆盖重建"
            )
        client.drop_collection(cfg.collection)
        logger.info("dropped existing collection %s", cfg.collection)

    client.create_collection(
        collection_name=cfg.collection,
        schema=build_schema(cfg),
        index_params=build_index_params(),
    )
    logger.info("created collection %s (dense FLAT/COSINE + BM25)", cfg.collection)

    records = load_records(cfg)
    logger.info("loaded %d papers from %s", len(records), cfg.data_dir)

    started = time.perf_counter()
    vectors = encoder.encode_documents([r.text for r in records], show_progress=True)
    logger.info(
        "embedded %d docs in %.1fs", len(vectors), time.perf_counter() - started
    )

    started = time.perf_counter()
    batch_size = cfg.embedding_batch_size
    for lo in range(0, len(records), batch_size):
        rows = [
            {**record.fields, "dense": vector}
            for record, vector in zip(
                records[lo : lo + batch_size], vectors[lo : lo + batch_size]
            )
        ]
        client.insert(cfg.collection, rows)
    logger.info(
        "inserted %d rows in %.1fs", len(records), time.perf_counter() - started
    )

    client.load_collection(cfg.collection)
    return len(records)
