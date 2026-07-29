"""Build and query the read-only SQLite FTS5 BM25 sidecar."""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .catalog import PaperRecord
from .errors import IndexUnavailable
from .models import SearchFilters

BM25_TOKENIZER = "unicode61 remove_diacritics 2"
_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RankedPaperId:
    """Paper ID, rank, and raw score returned by one retrieval channel."""

    paper_id: str
    rank: int
    raw_score: float


class BM25IndexBuilder:
    """Build an FTS5 index file incrementally and idempotently."""

    def __init__(self, path: Path, *, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not resume:
            self.path.unlink()
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                paper_id UNINDEXED,
                title,
                abstract,
                area UNINDEXED,
                paper_type UNINDEXED,
                spotlight UNINDEXED,
                tokenize='{BM25_TOKENIZER}'
            )
            """
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._connection.commit()

    def add_records(self, records: Sequence[PaperRecord]) -> None:
        """Write a batch of complete records idempotently by paper ID."""
        with self._connection:
            for record in records:
                fields = record.fields
                self._connection.execute(
                    "DELETE FROM papers_fts WHERE paper_id = ?", (record.paper_id,)
                )
                self._connection.execute(
                    """
                    INSERT INTO papers_fts(
                        paper_id, title, abstract, area, paper_type, spotlight
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.paper_id,
                        str(fields.get("title", "")),
                        str(fields.get("abstract", "")),
                        str(fields.get("area", "")),
                        str(fields.get("type", "")),
                        "1" if bool(fields.get("spot", False)) else "0",
                    ),
                )

    def finalize(self, *, corpus_hash: str) -> None:
        """Write version metadata and compact the FTS index."""
        with self._connection:
            values = {
                "corpus_hash": corpus_hash,
                "record_count": str(self.count()),
                "tokenizer": BM25_TOKENIZER,
            }
            self._connection.executemany(
                "INSERT OR REPLACE INTO index_meta(key, value) VALUES (?, ?)",
                values.items(),
            )
            self._connection.execute(
                "INSERT INTO papers_fts(papers_fts) VALUES ('optimize')"
            )
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def count(self) -> int:
        """Return the current FTS document count."""
        row = self._connection.execute("SELECT count(*) FROM papers_fts").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the build connection."""
        self._connection.close()

    def __enter__(self) -> BM25IndexBuilder:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class BM25Index:
    """FTS5 index that can be read safely from multiple worker threads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise IndexUnavailable()
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        try:
            self.count()
        except sqlite3.Error as exc:
            self.close()
            raise IndexUnavailable() from exc

    def search(
        self, query: str, *, limit: int, filters: SearchFilters
    ) -> list[RankedPaperId]:
        """Run one fixed-LIMIT BM25 ranking over the filtered candidate domain."""
        if limit <= 0:
            return []
        expression = _fts_query(query)
        if not expression:
            return []

        clauses = ["papers_fts MATCH ?"]
        params: list[object] = [expression]
        if filters.area:
            clauses.append("area = ?")
            params.append(filters.area)
        if filters.paper_type:
            clauses.append("paper_type = ?")
            params.append(filters.paper_type)
        if filters.spotlight_only:
            clauses.append("spotlight = '1'")
        params.append(limit)
        sql = f"""
            SELECT paper_id, bm25(papers_fts, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0)
            FROM papers_fts
            WHERE {' AND '.join(clauses)}
            ORDER BY 2 ASC, paper_id ASC
            LIMIT ?
        """
        try:
            rows = self._connection().execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise IndexUnavailable() from exc
        return [
            RankedPaperId(paper_id=str(row[0]), rank=rank, raw_score=float(row[1]))
            for rank, row in enumerate(rows, start=1)
        ]

    def count(self) -> int:
        """Return the paper count in the read-only index."""
        row = self._connection().execute("SELECT count(*) FROM papers_fts").fetchone()
        return int(row[0]) if row else 0

    def corpus_hash(self) -> str:
        """Return the corpus hash written during the build."""
        row = self._connection().execute(
            "SELECT value FROM index_meta WHERE key = 'corpus_hash'"
        ).fetchone()
        if not row:
            raise IndexUnavailable()
        return str(row[0])

    def close(self) -> None:
        """Close every thread-local connection created by this instance."""
        with self._connections_lock:
            connections, self._connections = self._connections, set()
        for connection in connections:
            connection.close()
        self._local.connection = None

    def _connection(self) -> sqlite3.Connection:
        """Lazily open an immutable read-only connection for the current thread."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            uri = f"file:{self.path.resolve().as_posix()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True, timeout=5.0)
            connection.execute("PRAGMA query_only=ON")
            self._local.connection = connection
            with self._connections_lock:
                self._connections.add(connection)
        return connection


def _fts_query(query: str) -> str:
    """Convert natural language into an OR query that cannot inject FTS operators."""
    terms = _WORD_RE.findall(query)
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
