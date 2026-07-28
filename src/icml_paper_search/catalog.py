"""本地语料元数据：读 index.json / abstracts.json，供入库与过滤取值提示使用。"""

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
    """Chroma 与 FTS5 共享的一条规范化论文记录。"""

    fields: dict[str, Any]
    text: str

    @property
    def paper_id(self) -> str:
        return self.fields["orid"]


def load_records(settings: Settings | None = None) -> list[PaperRecord]:
    """组装全量入库记录：index.json 的元数据 + abstracts.json 的摘要。"""
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
    """计算与输入顺序无关的规范语料 SHA-256。"""
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
    """计算排序后论文 id 集合的稳定 SHA-256。"""
    digest = hashlib.sha256()
    for paper_id in sorted(paper_ids):
        encoded = paper_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def data_file_hashes(settings: Settings | None = None) -> dict[str, str]:
    """返回构成语料的原始数据文件 SHA-256。"""
    cfg = settings or get_settings()
    return {
        "index.json": _file_hash(cfg.index_path),
        "abstracts.json": _file_hash(cfg.abstracts_path),
    }


@dataclass(frozen=True, slots=True)
class Catalog:
    """语料的可过滤维度取值与分布。"""

    total: int
    areas: tuple[tuple[str, int], ...]
    paper_types: tuple[tuple[str, int], ...]
    spotlights: int

    def render(self) -> str:
        """渲染成 filter 取值提示。"""
        lines = [f"ICML 2026 语料共 {self.total} 篇，其中 spotlight {self.spotlights} 篇。"]
        lines.append("")
        lines.append("area 可选值（用于 search_papers 的 area 参数）：")
        lines.extend(f"  {name}  ({count})" for name, count in self.areas)
        lines.append("")
        lines.append("paper_type 可选值：")
        lines.extend(f"  {name}  ({count})" for name, count in self.paper_types)
        return "\n".join(lines)


def get_catalog(settings: Settings | None = None) -> Catalog:
    """统计 area / type 分布；按数据文件路径缓存，语料是静态的。"""
    cfg = settings or get_settings()
    return _build_catalog(cfg.index_path)


# 缓存键用路径而非 Settings：pydantic 的 Settings 不可 hash，直接喂给 lru_cache 会炸。
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
            f"缺少数据文件 {path}；请确认 ICML_PAPERS_DATA_DIR 指向包含 "
            "index.json 与 abstracts.json 的目录"
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _file_hash(path: Path) -> str:
    """流式计算单个数据文件 SHA-256。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少数据文件 {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
