"""Validate bilingual navigation, links, metadata, and generated includes."""

from __future__ import annotations

import json
import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1]
CONTENT = DOCS_ROOT / "content" / "docs"
GENERATED = DOCS_ROOT / "generated"
LANGUAGES = ("en", "zh")
EXPECTED_PAGE_COUNT = 24
EXPECTED_INCLUDES = {
    Path("reference/configuration.mdx"): "configuration.mdx",
    Path("reference/mcp-tools.mdx"): "mcp-tools.mdx",
}
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
INCLUDE = re.compile(r"<include>([^<]+)</include>")
DEEP_HEADING = re.compile(r"^#{4,}\s", re.MULTILINE)


def _page_route(language: str, relative: Path) -> str:
    """Map a locale-relative content path to its public route."""
    without_suffix = relative.with_suffix("")
    parts = list(without_suffix.parts)
    if parts[-1] == "index":
        parts.pop()
    suffix = f"/{'/'.join(parts)}" if parts else ""
    return f"/{language}/docs{suffix}"


def _frontmatter(text: str) -> dict[str, str] | None:
    """Parse the small scalar frontmatter contract used by the site."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    values: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _normalize_internal_route(target: str) -> str:
    """Drop a fragment, query, and trailing slash from an internal route."""
    route = target.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    return route or "/"


def _validate_navigation(
    language: str, pages: set[Path], errors: list[str]
) -> dict[Path, list[str]]:
    """Validate metadata targets and return locale-relative page arrays."""
    locale_root = CONTENT / language
    metadata: dict[Path, list[str]] = {}
    navigated_pages: set[Path] = set()
    for meta_path in sorted(locale_root.rglob("meta.json")):
        relative_meta = meta_path.relative_to(locale_root)
        try:
            value = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"{meta_path.relative_to(DOCS_ROOT)}: invalid JSON: {error}")
            continue
        entries = value.get("pages")
        if not isinstance(entries, list) or any(
            not isinstance(entry, str) or not entry for entry in entries
        ):
            errors.append(f"{meta_path.relative_to(DOCS_ROOT)}: invalid pages array")
            continue
        if len(entries) != len(set(entries)):
            errors.append(f"{meta_path.relative_to(DOCS_ROOT)}: duplicate navigation entry")
        metadata[relative_meta] = entries
        folder = relative_meta.parent
        for entry in entries:
            file_target = folder / f"{entry}.mdx"
            directory_target = folder / entry
            if file_target in pages:
                navigated_pages.add(file_target)
            elif (locale_root / directory_target).is_dir():
                if not (locale_root / directory_target / "meta.json").is_file():
                    errors.append(
                        f"{meta_path.relative_to(DOCS_ROOT)}: "
                        f"navigation folder lacks meta.json: {entry}"
                    )
            else:
                errors.append(
                    f"{meta_path.relative_to(DOCS_ROOT)}: missing navigation entry {entry}"
                )
    missing = sorted(pages - navigated_pages)
    if missing:
        errors.append(
            f"content/docs/{language}: pages omitted from navigation: "
            + ", ".join(str(path) for path in missing)
        )
    return metadata


def main() -> int:
    """Reject locale drift, broken links, and authoring contract violations."""
    errors: list[str] = []
    roots = sorted(path.name for path in CONTENT.iterdir() if path.is_dir())
    if roots != list(LANGUAGES):
        errors.append(f"content/docs: expected locale roots {LANGUAGES}, found {roots}")

    page_sets: dict[str, set[Path]] = {}
    metadata_sets: dict[str, dict[Path, list[str]]] = {}
    route_sets: dict[str, set[str]] = {}
    for language in LANGUAGES:
        locale_root = CONTENT / language
        pages = {
            path.relative_to(locale_root) for path in locale_root.rglob("*.mdx")
        }
        page_sets[language] = pages
        route_sets[language] = {_page_route(language, path) for path in pages}
        if len(pages) != EXPECTED_PAGE_COUNT:
            errors.append(
                f"content/docs/{language}: expected {EXPECTED_PAGE_COUNT} pages, "
                f"found {len(pages)}"
            )
        metadata_sets[language] = _validate_navigation(language, pages, errors)

        for relative in sorted(pages):
            path = locale_root / relative
            text = path.read_text(encoding="utf-8")
            display_path = path.relative_to(DOCS_ROOT)
            if len(text.splitlines()) > 300:
                errors.append(f"{display_path}: page exceeds 300 lines")
            frontmatter = _frontmatter(text)
            if frontmatter is None or not all(
                frontmatter.get(key) for key in ("title", "description")
            ):
                errors.append(f"{display_path}: missing title/description frontmatter")
            if DEEP_HEADING.search(text):
                errors.append(f"{display_path}: headings deeper than level 3 are forbidden")

            for raw_target in LINK.findall(text):
                target = raw_target.strip().strip("<>")
                route = _normalize_internal_route(target)
                if route == "/docs" or route.startswith("/docs/"):
                    errors.append(f"{display_path}: legacy unlocalized link {target}")
                    continue
                if route.startswith("/en/docs") or route.startswith("/zh/docs"):
                    expected_prefix = f"/{language}/docs"
                    if not (route == expected_prefix or route.startswith(expected_prefix + "/")):
                        errors.append(f"{display_path}: cross-locale link {target}")
                    elif route not in route_sets[language]:
                        errors.append(f"{display_path}: broken internal link {target}")

            includes = INCLUDE.findall(text)
            expected_include = EXPECTED_INCLUDES.get(relative)
            if expected_include:
                expected_path = (GENERATED / language / expected_include).resolve()
                resolved = [(path.parent / value).resolve() for value in includes]
                if resolved != [expected_path] or not expected_path.is_file():
                    errors.append(
                        f"{display_path}: expected exactly one include of "
                        f"generated/{language}/{expected_include}"
                    )
            elif includes:
                errors.append(f"{display_path}: unexpected generated include")

    if page_sets.get("en") != page_sets.get("zh"):
        english_only = sorted(page_sets.get("en", set()) - page_sets.get("zh", set()))
        chinese_only = sorted(page_sets.get("zh", set()) - page_sets.get("en", set()))
        errors.append(
            f"locale page parity failed: en_only={english_only}, zh_only={chinese_only}"
        )
    if metadata_sets.get("en") != metadata_sets.get("zh"):
        errors.append("locale navigation parity failed: meta.json paths or pages differ")

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"checked docs: {EXPECTED_PAGE_COUNT} pages per locale, "
        f"{len(metadata_sets['en'])} navigation files per locale"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
