"""Load local corpus metadata for indexing and filter-value guidance."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class PaperRecord:
    """Normalized paper record shared by Chroma and FTS5."""

    fields: dict[str, Any]
    text: str

    @property
    def paper_id(self) -> str:
        return self.fields["orid"]


def load_records(settings: Settings | None = None) -> list[PaperRecord]:
    """Assemble all index records from metadata and abstract snapshots."""
    cfg = settings or get_settings()
    index = _read_json(cfg.index_path)
    abstracts = _read_json(cfg.abstracts_path)

    records: list[PaperRecord] = []
    for paper in index["papers"]:
        abstract = abstracts.get(paper["orid"], "")
        text = f"{paper['title']}\n{abstract}".strip()
        records.append(
            PaperRecord(
                fields={
                    "orid": paper["orid"],
                    "i": paper["i"],
                    "title": paper["title"],
                    "abstract": abstract,
                    "text": text,
                    "authors": json.dumps(paper.get("authors", []), ensure_ascii=False),
                    "area": paper.get("area", ""),
                    "sub": paper.get("sub", ""),
                    "type": paper.get("type", ""),
                    "spot": bool(paper.get("spot", False)),
                    "pid": paper.get("pid", ""),
                    "arxiv": paper.get("arxiv", ""),
                },
                text=text,
            )
        )
    return records


def corpus_hash(records: list[PaperRecord]) -> str:
    """Compute the canonical order-independent corpus SHA-256."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.paper_id):
        payload = {"fields": record.fields, "text": record.text}
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def paper_ids_hash(paper_ids: Iterable[str]) -> str:
    """Compute a stable SHA-256 over the sorted paper ID set."""
    digest = hashlib.sha256()
    for paper_id in sorted(paper_ids):
        encoded = paper_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def data_file_hashes(settings: Settings | None = None) -> dict[str, str]:
    """Return SHA-256 hashes for the source files that form the corpus."""
    cfg = settings or get_settings()
    return {
        "index.json": _file_hash(cfg.index_path),
        "abstracts.json": _file_hash(cfg.abstracts_path),
    }


@dataclass(frozen=True, slots=True)
class Catalog:
    """Filterable dimension values and distributions for the corpus."""

    total: int
    areas: tuple[tuple[str, int], ...]
    paper_types: tuple[tuple[str, int], ...]
    spotlights: int

    def render(self) -> str:
        """Render filter-value guidance."""
        lines = [
            f"ICML 2026 corpus: {self.total} papers, including "
            f"{self.spotlights} spotlight papers."
        ]
        lines.append("")
        lines.append("Available area values (for the search_papers area parameter):")
        lines.extend(f"  {name}  ({count})" for name, count in self.areas)
        lines.append("")
        lines.append("Available paper_type values:")
        lines.extend(f"  {name}  ({count})" for name, count in self.paper_types)
        return "\n".join(lines)


def get_catalog(settings: Settings | None = None) -> Catalog:
    """Count area and type values, caching by path because the corpus is static."""
    cfg = settings or get_settings()
    return _build_catalog(cfg.index_path)


# Cache by path because Pydantic Settings objects are not hashable.
@lru_cache(maxsize=4)
def _build_catalog(index_path: Path) -> Catalog:
    index = _read_json(index_path)
    papers = index["papers"]
    areas = Counter(p.get("area") or "Uncategorized" for p in papers)
    types = Counter(p.get("type") or "Unknown" for p in papers)
    return Catalog(
        total=len(papers),
        areas=tuple(areas.most_common()),
        paper_types=tuple(types.most_common()),
        spotlights=sum(1 for p in papers if p.get("spot")),
    )


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file {path} is missing; ensure ICML_PAPERS_DATA_DIR points to a "
            "directory containing index.json and abstracts.json"
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _file_hash(path: Path) -> str:
    """Compute a data file's SHA-256 incrementally."""
    if not path.exists():
        raise FileNotFoundError(f"Data file {path} is missing")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
