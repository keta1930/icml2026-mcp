"""显式建库、预算、checkpoint 恢复与原子发布测试。"""

from __future__ import annotations

import json

import pytest

from icml_paper_search.catalog import load_records
from icml_paper_search.config import Settings
from icml_paper_search.embedding import EmbeddingBatch
from icml_paper_search.errors import BudgetExhausted, IndexUnavailable
from icml_paper_search.ingest import _estimate_tokens, build_index, plan_build
from icml_paper_search.manifest import load_manifest, resolve_current

DIMENSION = 1024


class FakeUsageEncoder:
    """返回固定维度向量和可控 usage 的离线 provider。"""

    def __init__(self, *, fail_on_call=None, tokens_per_batch=3):
        self.fail_on_call = fail_on_call
        self.tokens_per_batch = tokens_per_batch
        self.calls = []

    def encode_documents_with_usage(self, texts, *, show_progress=False):
        del show_progress
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("offline provider failure")
        vectors = []
        for index, _text in enumerate(texts):
            vector = [0.0] * DIMENSION
            vector[index % DIMENSION] = 1.0
            vectors.append(tuple(vector))
        return EmbeddingBatch(tuple(vectors), self.tokens_per_batch)

    def encode_query(self, _text):
        raise AssertionError("建库不应调用 query embedding")

    def encode_documents(self, texts, *, show_progress=False):
        return [list(vector) for vector in self.encode_documents_with_usage(
            texts, show_progress=show_progress
        ).vectors]


class ReservationUsageEncoder(FakeUsageEncoder):
    """让离线 usage 精确等于建库调用前的保守预留。"""

    def encode_documents_with_usage(self, texts, *, show_progress=False):
        self.tokens_per_batch = sum(_estimate_tokens(text) for text in texts)
        return super().encode_documents_with_usage(
            texts, show_progress=show_progress
        )


def write_corpus(path, count=3, *, suffix=""):
    papers = []
    abstracts = {}
    for index in range(count):
        paper_id = f"p{index}{suffix}"
        papers.append(
            {
                "i": index,
                "pid": str(index),
                "orid": paper_id,
                "title": f"Paper {index} {suffix}",
                "authors": ["Alice"],
                "area": "Theory" if index % 2 else "Vision",
                "sub": "",
                "type": "Poster",
                "spot": index == 0,
                "arxiv": "",
            }
        )
        abstracts[paper_id] = f"graph learning abstract {index}"
    (path / "index.json").write_text(json.dumps({"papers": papers}), encoding="utf-8")
    (path / "abstracts.json").write_text(json.dumps(abstracts), encoding="utf-8")


@pytest.fixture
def build_settings(tmp_path):
    write_corpus(tmp_path)
    return Settings(
        data_dir=tmp_path,
        index_root=tmp_path / "index-store",
        embedding_batch_size=2,
    )


def test_plan_and_default_build_are_preflight_without_provider(build_settings):
    plan = plan_build(build_settings)
    result = build_index(build_settings, max_input_tokens=None)

    assert plan.record_count == 3
    assert len(plan.corpus_hash) == 64
    assert plan.estimated_input_tokens > 0
    assert result.plan == plan
    assert result.executed is False
    assert not build_settings.index_root.exists()


def test_execute_requires_positive_hard_cap(build_settings):
    with pytest.raises(ValueError, match="max_input_tokens"):
        build_index(build_settings, FakeUsageEncoder(), max_input_tokens=0, execute=True)


def test_token_reservation_bounds_utf8_bytes_with_per_input_overhead():
    assert _estimate_tokens("a") > len("a".encode("utf-8"))
    assert _estimate_tokens("衣服") - len("衣服".encode("utf-8")) == (
        _estimate_tokens("a") - len("a".encode("utf-8"))
    )


def test_build_batches_verifies_and_atomically_promotes(build_settings):
    encoder = FakeUsageEncoder(tokens_per_batch=4)
    result = build_index(
        build_settings,
        encoder,
        max_input_tokens=100_000,
        execute=True,
    )

    assert result.executed is True
    assert result.total_tokens == 8
    assert [len(batch) for batch in encoder.calls] == [2, 1]
    assert result.version_path == resolve_current(build_settings.index_root)
    manifest = load_manifest(result.version_path)
    assert manifest.record_count == 3
    assert manifest.embedding_dim == DIMENSION
    assert not (result.version_path / "checkpoint.json").exists()


def test_budget_stops_before_provider_and_never_switches_current(build_settings):
    encoder = FakeUsageEncoder()
    with pytest.raises(BudgetExhausted):
        build_index(build_settings, encoder, max_input_tokens=1, execute=True)

    assert encoder.calls == []
    assert not (build_settings.index_root / "current").exists()
    assert plan_build(build_settings).staging_path.is_dir()


def test_last_batch_shrinks_to_remaining_hard_cap_and_resume_is_prefix_safe(
    build_settings,
):
    records = load_records(build_settings)
    reservations = [_estimate_tokens(record.text) for record in records]
    encoder = ReservationUsageEncoder()

    with pytest.raises(BudgetExhausted):
        build_index(
            build_settings,
            encoder,
            max_input_tokens=reservations[0] + reservations[1] - 1,
            execute=True,
        )

    assert [len(batch) for batch in encoder.calls] == [1]
    checkpoint_path = plan_build(build_settings).staging_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["completed_ids"] == [records[0].paper_id]
    assert checkpoint["total_tokens"] == reservations[0]
    assert not (build_settings.index_root / "current").exists()

    resumed = ReservationUsageEncoder()
    result = build_index(
        build_settings,
        resumed,
        max_input_tokens=sum(reservations),
        execute=True,
        resume=True,
    )
    assert [len(batch) for batch in resumed.calls] == [2]
    assert result.total_tokens == sum(reservations)


def test_provider_usage_above_reservation_is_audited_without_checkpoint_commit(
    build_settings,
):
    first_reservation = _estimate_tokens(load_records(build_settings)[0].text)
    encoder = FakeUsageEncoder(tokens_per_batch=first_reservation + 1)

    with pytest.raises(BudgetExhausted):
        build_index(
            build_settings,
            encoder,
            max_input_tokens=first_reservation,
            execute=True,
        )

    staging = plan_build(build_settings).staging_path
    checkpoint = json.loads((staging / "checkpoint.json").read_text(encoding="utf-8"))
    audit = json.loads((staging / "failure-audit.json").read_text(encoding="utf-8"))
    assert checkpoint["completed_ids"] == []
    assert checkpoint["total_tokens"] == 0
    assert audit["reason"] == "provider_usage_exceeded_reservation"
    assert audit["reserved_tokens"] == first_reservation
    assert audit["actual_tokens"] == first_reservation + 1
    assert "Paper 0" not in json.dumps(audit)
    assert not (build_settings.index_root / "current").exists()

    with pytest.raises(IndexUnavailable):
        build_index(
            build_settings,
            FakeUsageEncoder(),
            max_input_tokens=100_000,
            execute=True,
            resume=True,
        )


def test_failed_build_keeps_current_and_resume_skips_complete_batch(build_settings):
    stable_dir = build_settings.index_root / "versions" / "stable"
    stable_dir.mkdir(parents=True)
    (build_settings.index_root / "current").write_text("stable\n", encoding="utf-8")
    failing = FakeUsageEncoder(fail_on_call=2)

    with pytest.raises(RuntimeError, match="offline provider failure"):
        build_index(
            build_settings,
            failing,
            max_input_tokens=100_000,
            execute=True,
        )
    assert (build_settings.index_root / "current").read_text().strip() == "stable"
    assert [len(batch) for batch in failing.calls] == [2, 1]

    resumed = FakeUsageEncoder(tokens_per_batch=2)
    result = build_index(
        build_settings,
        resumed,
        max_input_tokens=100_000,
        execute=True,
        resume=True,
    )
    assert result.resumed is True
    assert [len(batch) for batch in resumed.calls] == [1]
    assert resolve_current(build_settings.index_root) == result.version_path
