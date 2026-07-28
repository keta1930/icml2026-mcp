"""开发用 CLI：建库、直接检索、交互检索。MCP 之外的第二个入口，便于排障。"""

from __future__ import annotations

import argparse
import logging
import sys

from .catalog import get_catalog
from .config import get_settings
from .formatting import format_search_outcome
from .ingest import build_collection
from .models import SearchFilters
from .service import PaperSearchService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="icml-papers", description="ICML 2026 论文库建库与检索"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="全量重建 Milvus collection")
    p_build.add_argument(
        "--keep-existing",
        action="store_true",
        help="collection 已存在时报错而非覆盖重建",
    )

    p_search = sub.add_parser("search", help="混合检索（留空 query 进入交互模式）")
    p_search.add_argument("query", nargs="*", help="检索语句")
    p_search.add_argument("-k", "--top-k", type=int, default=None, help="返回条数")
    p_search.add_argument(
        "-d", "--depth", type=int, default=None, help="相关性窗口深度（默认 top_k×4）"
    )
    p_search.add_argument("--area", default=None, help="按领域过滤")
    p_search.add_argument("--type", dest="paper_type", default=None, help="按类型过滤")
    p_search.add_argument("--spotlight", action="store_true", help="只要 spotlight")
    p_search.add_argument(
        "--session",
        action="store_true",
        help="交互模式下开启会话去重，已看过的论文不再出现",
    )

    sub.add_parser("info", help="查看配置与语料分布")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "build":
        rows = build_collection(drop_existing=not args.keep_existing)
        print(f"已写入 {rows} 篇论文")
        return 0

    if args.command == "info":
        cfg = get_settings()
        print(f"milvus     : {cfg.milvus_uri}")
        print(f"collection : {cfg.collection}")
        print(f"data_dir   : {cfg.data_dir}")
        print(f"embedding  : {cfg.embedding_model_path} ({cfg.embedding_device})")
        print(f"weights    : dense={cfg.dense_weight} bm25={cfg.bm25_weight}")
        print()
        print(get_catalog().render())
        return 0

    return _search(args)


def _search(args: argparse.Namespace) -> int:
    cfg = get_settings()
    service = PaperSearchService(cfg)
    filters = SearchFilters(
        area=args.area, paper_type=args.paper_type, spotlight_only=args.spotlight
    )

    def run(query: str, session_id: str | None) -> None:
        outcome = service.search(
            query,
            session_id=session_id,
            top_k=args.top_k,
            search_depth=args.depth,
            filters=filters,
        )
        print(format_search_outcome(outcome, cfg))

    if args.query:
        run(" ".join(args.query), None)
        return 0

    session_id = service.open_session(label="cli").session_id if args.session else None
    hint = "（会话去重开启）" if session_id else ""
    print(f"交互检索{hint}，空行或 Ctrl-D 退出")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        run(query, session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
