"""SQLite FTS5 BM25 sidecar 的构建与只读查询。"""

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
    """单路检索返回的论文 id、名次和原始分数。"""

    paper_id: str
    rank: int
    raw_score: float


class BM25IndexBuilder:
    """增量、幂等地构建一个 FTS5 索引文件。"""

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
        """以 paper id 幂等写入一批完整记录。"""
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
        """写入版本元数据并压缩 FTS 索引。"""
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
        """返回当前 FTS 文档数。"""
        row = self._connection.execute("SELECT count(*) FROM papers_fts").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """关闭构建连接。"""
        self._connection.close()

    def __enter__(self) -> BM25IndexBuilder:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class BM25Index:
    """可从多个工作线程安全读取的 FTS5 索引。"""

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
        """在过滤后的候选域执行一次固定 LIMIT 的 BM25 排名。"""
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
        """返回只读索引内的论文数。"""
        row = self._connection().execute("SELECT count(*) FROM papers_fts").fetchone()
        return int(row[0]) if row else 0

    def corpus_hash(self) -> str:
        """返回构建时写入的语料哈希。"""
        row = self._connection().execute(
            "SELECT value FROM index_meta WHERE key = 'corpus_hash'"
        ).fetchone()
        if not row:
            raise IndexUnavailable()
        return str(row[0])

    def close(self) -> None:
        """关闭本实例创建的全部线程连接。"""
        with self._connections_lock:
            connections, self._connections = self._connections, set()
        for connection in connections:
            connection.close()
        self._local.connection = None

    def _connection(self) -> sqlite3.Connection:
        """为当前线程延迟打开 immutable 只读连接。"""
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
    """把自然语言转换为不会注入 FTS 操作符的 OR 查询。"""
    terms = _WORD_RE.findall(query)
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
