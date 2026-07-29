# Documentation site

The project documentation uses Fumadocs and Next.js static export. English is the default language, and every public page has an explicit `/en` or `/zh` route.

- `content/docs/en/` and `content/docs/zh/` are the paired authoring trees.
- `generated/{en,zh}/` contains runtime-derived MCP and configuration tables.
- `i18n/reference.zh.json` is the strictly validated Chinese translation catalog for runtime schemas.
- `out/` is disposable build output and is not committed.

## Local development

```bash
pnpm install
pnpm dev
```

Open `http://127.0.0.1:20440/` and choose a language. The public documentation site is available at <https://keta1930.github.io/icml2026-mcp/>.

## Validation and build

After changing a FastMCP tool schema or `Settings`, explicitly refresh and check the committed Reference files:

```bash
pnpm generate:reference
pnpm check:reference
```

Normal validation is read-only for committed Reference files and fails before any generated drift can be repaired:

```bash
pnpm check:reference
pnpm check:docs
pnpm typecheck
pnpm build
```

`pnpm typecheck` always runs the locked `fumadocs-mdx` generator first, so ignored `.source/` modules are reproducible on a clean checkout. `pnpm build` never runs `generate:reference`; it fails on translation-catalog drift, stale or missing generated Reference files, locale page or navigation mismatch, broken localized links, missing static HTML or Markdown exports, mixed-locale search data, and invalid canonical or alternate metadata.

To reproduce the GitHub Pages project subpath locally:

```bash
GITHUB_PAGES=true \
NEXT_PUBLIC_BASE_PATH=/icml2026-mcp \
NEXT_PUBLIC_SITE_URL=https://keta1930.github.io/icml2026-mcp \
pnpm build
```

The static output includes `/llms.txt`, locale indexes at `/{lang}/llms.txt`, and clean Markdown for every page. GitHub Pages must use GitHub Actions as its source; `.github/workflows/docs-pages.yml` publishes `docs/out/` from the `opensource` branch.
