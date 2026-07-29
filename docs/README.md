# Documentation site

Fumadocs + Next.js 文档站。`content/docs/` 是人工内容源，`generated/` 由 Python runtime schema 生成。

在线地址：<https://keta1930.github.io/icml2026-mcp/>

```bash
pnpm install
pnpm dev
pnpm build
```

本地构建输出到 `out/`。模拟 GitHub Pages 项目子路径时设置：

```bash
GITHUB_PAGES=true \
NEXT_PUBLIC_BASE_PATH=/icml2026-mcp \
NEXT_PUBLIC_SITE_URL=https://keta1930.github.io/icml2026-mcp \
pnpm build
```

构建会更新 MCP/config Reference、检查页面导航与内部链接、导出静态站点，并派生 `/llms.txt` 与逐页 clean Markdown。`.source/`、`.next/` 和 `out/` 不提交。

`.github/workflows/docs-pages.yml` 在 `opensource` 分支更新后构建并发布 `docs/out/`。GitHub Pages 的 Source 必须设置为 GitHub Actions。
