"""Milvus collection schema for dense and BM25 hybrid retrieval."""

from __future__ import annotations

import re
from typing import Any

from pymilvus import DataType, Function, FunctionType

from ..domain.errors import IndexUnavailable

DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
DOCUMENT_FIELD = "document"
PRIMARY_FIELD = "paper_id"
RRF_K = 60

PAPER_OUTPUT_FIELDS = (
    "paper_id",
    "title",
    "abstract",
    "authors_json",
    "area",
    "subarea",
    "paper_type",
    "spotlight",
    "arxiv_id",
)

COLLECTION_FIELDS = {
    PRIMARY_FIELD,
    DOCUMENT_FIELD,
    DENSE_FIELD,
    SPARSE_FIELD,
    *PAPER_OUTPUT_FIELDS,
}


def collection_name(prefix: str, build_id: str) -> str:
    """Return a deterministic Milvus collection name for one corpus build."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,39}", prefix):
        raise ValueError("invalid Milvus collection prefix")
    if not re.fullmatch(r"[0-9a-f]{16}", build_id):
        raise ValueError("invalid build ID")
    return f"{prefix}_{build_id}"


def create_collection(
    client: Any, name: str, dimension: int, *, timeout: float
) -> None:
    """Create and load the current dense plus BM25 collection schema."""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(PRIMARY_FIELD, DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(
        DOCUMENT_FIELD,
        DataType.VARCHAR,
        max_length=65_535,
        enable_analyzer=True,
    )
    schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("title", DataType.VARCHAR, max_length=4_096)
    schema.add_field("abstract", DataType.VARCHAR, max_length=65_535)
    schema.add_field("authors_json", DataType.VARCHAR, max_length=16_384)
    schema.add_field("area", DataType.VARCHAR, max_length=512)
    schema.add_field("subarea", DataType.VARCHAR, max_length=512)
    schema.add_field("paper_type", DataType.VARCHAR, max_length=128)
    schema.add_field("spotlight", DataType.BOOL)
    schema.add_field("arxiv_id", DataType.VARCHAR, max_length=128)
    schema.add_function(
        Function(
            name="paper_bm25",
            input_field_names=[DOCUMENT_FIELD],
            output_field_names=SPARSE_FIELD,
            function_type=FunctionType.BM25,
        )
    )

    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name=DENSE_FIELD,
        index_name="dense_cosine",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    indexes.add_index(
        field_name=SPARSE_FIELD,
        index_name="sparse_bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"bm25_k1": 1.2, "bm25_b": 0.75},
    )
    client.create_collection(
        name,
        schema=schema,
        index_params=indexes,
        consistency_level="Strong",
        timeout=timeout,
    )
    client.load_collection(name, timeout=timeout)


def validate_collection_schema(
    client: Any, name: str, dimension: int, *, timeout: float
) -> None:
    """Reject a collection that does not match the current field contract."""
    description = client.describe_collection(name, timeout=timeout)
    fields = description.get("fields") if isinstance(description, dict) else None
    if not isinstance(fields, list):
        raise IndexUnavailable()
    names = {
        str(field.get("name"))
        for field in fields
        if isinstance(field, dict) and field.get("name")
    }
    if names != COLLECTION_FIELDS:
        raise IndexUnavailable()

    dense = next(
        (
            field
            for field in fields
            if isinstance(field, dict) and field.get("name") == DENSE_FIELD
        ),
        None,
    )
    params = dense.get("params", {}) if isinstance(dense, dict) else {}
    if int(params.get("dim", 0)) != dimension:
        raise IndexUnavailable()

    functions = description.get("functions", [])
    if not isinstance(functions, list) or len(functions) != 1:
        raise IndexUnavailable()
    function = functions[0]
    expected_function = (
        isinstance(function, dict)
        and function.get("name") == "paper_bm25"
        and function.get("type") == FunctionType.BM25
        and function.get("input_field_names") == [DOCUMENT_FIELD]
        and function.get("output_field_names") == [SPARSE_FIELD]
    )
    if not expected_function:
        raise IndexUnavailable()

    index_names = set(client.list_indexes(name, timeout=timeout))
    if index_names != {"dense_cosine", "sparse_bm25"}:
        raise IndexUnavailable()
    dense_index = client.describe_index(name, "dense_cosine", timeout=timeout)
    sparse_index = client.describe_index(name, "sparse_bm25", timeout=timeout)
    if not (
        dense_index.get("field_name") == DENSE_FIELD
        and dense_index.get("index_type") == "AUTOINDEX"
        and dense_index.get("metric_type") == "COSINE"
    ):
        raise IndexUnavailable()
    if not (
        sparse_index.get("field_name") == SPARSE_FIELD
        and sparse_index.get("index_type") == "SPARSE_INVERTED_INDEX"
        and sparse_index.get("metric_type") == "BM25"
        and float(sparse_index.get("bm25_k1", 0)) == 1.2
        and float(sparse_index.get("bm25_b", 0)) == 0.75
    ):
        raise IndexUnavailable()
