"""校验文档导航、内部链接和页面元数据。"""

from __future__ import annotations

import json
import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1]
CONTENT = DOCS_ROOT / "content" / "docs"
LINK = re.compile(r"\[[^\]]+\]\((/docs[^)#?]*)")


def _page_url(path: Path) -> str:
    """把内容文件映射为 Fumadocs 页面 URL。"""
    relative = path.relative_to(CONTENT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "index":
        parts.pop()
    return "/docs" + (f"/{'/'.join(parts)}" if parts else "")


def main() -> int:
    """发现所有页面并拒绝断链或缺失导航项。"""
    pages = sorted(CONTENT.rglob("*.mdx"))
    urls = {_page_url(path) for path in pages}
    errors: list[str] = []

    for path in pages:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(DOCS_ROOT)
        if len(text.splitlines()) > 300:
            errors.append(f"{relative}: page exceeds 300 lines")
        if not text.startswith("---\n") or "\ntitle:" not in text or "\ndescription:" not in text:
            errors.append(f"{relative}: missing title/description frontmatter")
        for target in LINK.findall(text):
            normalized = target.rstrip("/") or "/docs"
            if normalized not in urls:
                errors.append(f"{relative}: broken internal link {target}")

    for meta_path in CONTENT.rglob("meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        folder = meta_path.parent
        for entry in meta.get("pages", []):
            file_target = folder / f"{entry}.mdx"
            directory_target = folder / entry
            if not file_target.exists() and not directory_target.is_dir():
                errors.append(
                    f"{meta_path.relative_to(DOCS_ROOT)}: missing navigation entry {entry}"
                )

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"checked docs: {len(pages)} pages, {len(urls)} internal routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
