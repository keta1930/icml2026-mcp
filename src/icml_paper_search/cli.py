"""Open-source CLI for offline preflight, explicit builds, search, and local status."""

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
    """Parse a command and return a stable process exit code."""
    parser = argparse.ArgumentParser(
        prog="icml-papers", description="Local hybrid indexing and search for ICML 2026"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser(
        "build", help="Preflight or explicitly build a versioned local index"
    )
    build.add_argument(
        "--execute",
        action="store_true",
        help="Confirm paid DashScope calls and build the index; the default is preflight only",
    )
    build.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Hard input-token cap for this build; required with --execute",
    )
    build.add_argument(
        "--resume", action="store_true", help="Resume from a matching complete-batch checkpoint"
    )

    search = subcommands.add_parser("search", help="Hybrid search (omit query for interactive mode)")
    search.add_argument("query", nargs="*", help="Search query")
    search.add_argument("-k", "--top-k", type=int, default=None, help="Number of results")
    search.add_argument("-d", "--depth", type=int, default=None, help="Fixed window depth")
    search.add_argument("--area", default=None, help="Filter by research area")
    search.add_argument("--type", dest="paper_type", default=None, help="Filter by paper type")
    search.add_argument("--spotlight", action="store_true", help="Return spotlight papers only")
    search.add_argument(
        "--session", action="store_true", help="Enable session deduplication in interactive mode"
    )

    subcommands.add_parser("info", help="Show local index, model, and corpus information")
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
    """Show preflight first, then execute only with confirmation and a hard cap."""
    settings = get_settings()
    plan = plan_build(settings)
    print(f"build_id              : {plan.build_id}")
    print(f"records               : {plan.record_count}")
    print(f"corpus_hash           : {plan.corpus_hash}")
    print(f"estimated_input_tokens: {plan.estimated_input_tokens}")
    if not args.execute:
        print("mode                  : preflight (DashScope was not called)")
        return 0
    if args.max_input_tokens is None or args.max_input_tokens <= 0:
        raise ValueError("--execute requires a positive --max-input-tokens value")
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
    """Print runtime information without secret values or provider access."""
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
    """Run one-shot or interactive search and always close local clients."""
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
        hint = " (session deduplication enabled)" if session_id else ""
        print(f"Interactive search{hint}; submit an empty line or press Ctrl-D to exit")
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
