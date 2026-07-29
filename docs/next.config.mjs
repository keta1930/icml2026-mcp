import { createMDX } from "fumadocs-mdx/next";

const isGitHubPages = process.env.GITHUB_PAGES === "true";
const basePath = isGitHubPages
  ? (process.env.NEXT_PUBLIC_BASE_PATH ?? "/icml2026-mcp")
  : "";

const config = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
  basePath,
  assetPrefix: basePath,
  images: {
    unoptimized: true,
  },
};

export default createMDX()(config);
