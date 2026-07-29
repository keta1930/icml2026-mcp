const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const siteConfig = {
  name: "ICML Paper Search MCP",
  description: "面向 ICML 2026 论文的本地混合检索与持久化调研会话。",
  publicMcpUrl: "https://icml.agent4sci.com:20441/mcp",
  url: process.env.NEXT_PUBLIC_SITE_URL ?? "http://127.0.0.1:20440",
  basePath,
};
