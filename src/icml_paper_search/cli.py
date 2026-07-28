"""开源版 CLI：零云预检、显式建库、检索与本地状态信息。"""

from __future__ import annotations

import argparse
import logging
import sys

from .catalog import get_catalog
from .config import get_settings
from .errors import AppError, public_error_text
from .formatting import format_search_outcome
from .ingest import build_index, plan_build
from .manifest import load_manifest, resolve_current
from .models import SearchFilters
from .service import PaperSearchService


def main(argv: list[str] | None = None) -> int:
    """解析命令并返回稳定进程退出码。"""
    parser = argparse.ArgumentParser(
        prog="icml-papers", description="ICML 2026 本地混合索引与检索"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build", help="预检或显式构建版本化本地索引")
    build.add_argument(
        "--execute",
        action="store_true",
        help="确认发起 DashScope 付费调用并构建索引；默认仅预检",
    )
    build.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="本次建库 Token 硬上限；--execute 时必填",
    )
    build.add_argument(
        "--resume", action="store_true", help="从匹配的完整批次 checkpoint 恢复"
    )

    search = subcommands.add_parser("search", help="混合检索（留空进入交互模式）")
    search.add_argument("query", nargs="*", help="检索语句")
    search.add_argument("-k", "--top-k", type=int, default=None, help="返回条数")
    search.add_argument("-d", "--depth", type=int, default=None, help="固定窗口深度")
    search.add_argument("--area", default=None, help="按领域过滤")
    search.add_argument("--type", dest="paper_type", default=None, help="按类型过滤")
    search.add_argument("--spotlight", action="store_true", help="只要 spotlight")
    search.add_argument("--session", action="store_true", help="交互模式启用 session 去重")

    subcommands.add_parser("info", help="查看本地索引、模型与语料信息")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "build":
            return _run_build(args)
        if args.command == "info":
            return _info()
        return _search(args)
    except (AppError, ValueError) as exc:
        print(public_error_text(exc), file=sys.stderr)
        return 2


def _run_build(args: argparse.Namespace) -> int:
    """先展示预检，再仅在显式确认与 hard cap 下执行。"""
    settings = get_settings()
    plan = plan_build(settings)
    print(f"build_id              : {plan.build_id}")
    print(f"records               : {plan.record_count}")
    print(f"corpus_hash           : {plan.corpus_hash}")
    print(f"estimated_input_tokens: {plan.estimated_input_tokens}")
    if not args.execute:
        print("mode                  : preflight（未调用 DashScope）")
        return 0
    if args.max_input_tokens is None or args.max_input_tokens <= 0:
        raise ValueError("--execute 必须同时提供正数 --max-input-tokens")
    result = build_index(
        settings,
        max_input_tokens=args.max_input_tokens,
        execute=True,
        resume=args.resume,
    )
    print(f"mode                  : executed")
    print(f"actual_input_tokens   : {result.total_tokens}")
    print(f"version_path          : {result.version_path}")
    return 0


def _info() -> int:
    """输出不包含 secret 值且不访问 provider 的运行信息。"""
    settings = get_settings()
    key_configured = bool(
        settings.dashscope_api_key is not None
        or settings.dashscope_api_key_file is not None
    )
    print(f"index_root : {settings.index_root}")
    print(f"state_db   : {settings.state_db_path}")
    print(f"embedding  : {settings.embedding_model} ({settings.embedding_dim}d)")
    print(f"key        : {'configured' if key_configured else 'unconfigured'}")
    print(
        f"rrf        : dense={settings.dense_weight} "
        f"bm25={settings.bm25_weight} k={settings.rrf_k}"
    )
    try:
        version = resolve_current(settings.index_root)
        manifest = load_manifest(version)
    except AppError:
        print("current    : not ready")
    else:
        print(f"current    : {manifest.build_id} ({manifest.record_count} papers)")
    print()
    print(get_catalog(settings).render())
    return 0


def _search(args: argparse.Namespace) -> int:
    """执行单次或交互检索，并始终关闭本地客户端。"""
    settings = get_settings()
    service = PaperSearchService(settings)
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
        print(format_search_outcome(outcome, settings))

    try:
        if args.query:
            run(" ".join(args.query), None)
            return 0

        session_id = (
            service.open_session(label="cli").session_id if args.session else None
        )
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
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
