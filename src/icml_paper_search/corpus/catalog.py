"""Load and normalize the ICML 2026 corpus shipped with the project."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class PaperRecord:
    """Canonical paper record used by indexing and validation."""

    paper_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    area: str
    subarea: str
    paper_type: str
    spotlight: bool
    arxiv_id: str

    @property
    def document(self) -> str:
        """Return the text used for dense and BM25 indexing."""
        return f"{self.title}\n{self.abstract}".strip()

    def milvus_entity(self, vector: Iterable[float]) -> dict[str, Any]:
        """Build one entity matching the current Milvus collection schema."""
        return {
            "paper_id": self.paper_id,
            "dense_vector": list(vector),
            "document": self.document,
            "title": self.title,
            "abstract": self.abstract,
            "authors_json": json.dumps(self.authors, ensure_ascii=False),
            "area": self.area,
            "subarea": self.subarea,
            "paper_type": self.paper_type,
            "spotlight": self.spotlight,
            "arxiv_id": self.arxiv_id,
        }


def load_records(settings: Settings | None = None) -> list[PaperRecord]:
    """Join metadata and abstracts into the canonical corpus."""
    cfg = settings or get_settings()
    index = _read_json(cfg.index_path)
    abstracts = _read_json(cfg.abstracts_path)

    records = [
        PaperRecord(
            paper_id=str(paper["orid"]),
            title=str(paper["title"]),
            abstract=str(abstracts.get(paper["orid"], "") or ""),
            authors=tuple(str(author) for author in paper.get("authors", [])),
            area=str(paper.get("area", "") or ""),
            subarea=str(paper.get("sub", "") or ""),
            paper_type=str(paper.get("type", "") or ""),
            spotlight=bool(paper.get("spot", False)),
            arxiv_id=str(paper.get("arxiv", "") or ""),
        )
        for paper in index["papers"]
    ]
    if len({record.paper_id for record in records}) != len(records):
        raise ValueError("the corpus contains duplicate paper IDs")
    return records


def corpus_hash(records: Iterable[PaperRecord]) -> str:
    """Compute the canonical order-independent corpus SHA-256."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.paper_id):
        encoded = json.dumps(
            asdict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def paper_ids_hash(paper_ids: Iterable[str]) -> str:
    """Compute a stable SHA-256 over a paper ID set."""
    digest = hashlib.sha256()
    for paper_id in sorted(paper_ids):
        encoded = paper_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def data_file_hashes(settings: Settings | None = None) -> dict[str, str]:
    """Return hashes for the two source files that form the corpus."""
    cfg = settings or get_settings()
    return {
        "index.json": _file_hash(cfg.index_path),
        "abstracts.json": _file_hash(cfg.abstracts_path),
    }


@dataclass(frozen=True, slots=True)
class Catalog:
    """Filter values and counts exposed through MCP."""

    total: int
    areas: tuple[tuple[str, int], ...]
    paper_types: tuple[tuple[str, int], ...]
    spotlights: int

    def render(self) -> str:
        """Render compact filter guidance for an agent."""
        lines = [
            (
                f"ICML 2026 corpus: {self.total} papers, including "
                f"{self.spotlights} spotlight papers."
            ),
            "",
            "Available area values:",
            *(f"  {name} ({count})" for name, count in self.areas),
            "",
            "Available paper_type values:",
            *(f"  {name} ({count})" for name, count in self.paper_types),
        ]
        return "\n".join(lines)


def get_catalog(settings: Settings | None = None) -> Catalog:
    """Return cached corpus filter counts."""
    cfg = settings or get_settings()
    return _build_catalog(cfg.index_path)


@lru_cache(maxsize=4)
def _build_catalog(index_path: Path) -> Catalog:
    papers = _read_json(index_path)["papers"]
    areas = Counter(paper.get("area") or "Uncategorized" for paper in papers)
    types = Counter(paper.get("type") or "Unknown" for paper in papers)
    return Catalog(
        total=len(papers),
        areas=tuple(areas.most_common()),
        paper_types=tuple(types.most_common()),
        spotlights=sum(1 for paper in papers if paper.get("spot")),
    )


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required data file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _file_hash(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"required data file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
