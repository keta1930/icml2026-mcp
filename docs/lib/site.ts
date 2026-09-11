import type { Language } from "@/lib/i18n";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const siteConfig = {
  name: "ICML Paper Search MCP",
  publicMcpUrl: "https://icml.agent4sci.com/mcp",
  url: process.env.NEXT_PUBLIC_SITE_URL ?? "http://127.0.0.1:20440",
  basePath,
};

const copy = {
  en: {
    description:
      "An open-source MCP service for searching 6,341 ICML 2026 papers.",
    navDocs: "Documentation",
    navMcp: "Public MCP",
    markdownAction: "View clean Markdown",
    homeEyebrow: "Open source · Self-hosted · Streamable HTTP",
    homeTitle: "6,341 ICML 2026 papers",
    homeTitlePrefix: "ICML Paper Search MCP",
    homeDescription:
      "Six tools for paper search, lookup, metadata, and search sessions. Use the public service or deploy the same code locally.",
    publicService: "Connect to the public service",
    localDeployment: "Deploy locally",
    imageAlt: "Vintage scientific journal cover for ICML Paper Search MCP",
    features: [
      ["Six MCP tools", "Search papers, fetch records, list metadata, and manage search sessions."],
      ["ICML 2026 snapshot", "Search a traceable dataset of 6,341 papers."],
      ["Open source", "Run the service with Milvus, SQLite, and a DashScope key."],
    ],
    docsFallback: "Project documentation",
    llmsStart: "Start here",
    llmsOptional: "Optional",
  },
  zh: {
    description: "用于检索 6,341 篇 ICML 2026 论文的开源 MCP 服务。",
    navDocs: "文档",
    navMcp: "公共 MCP",
    markdownAction: "查看 clean Markdown",
    homeEyebrow: "开源 · 可本地部署 · Streamable HTTP",
    homeTitle: "6,341 篇 ICML 2026 论文",
    homeTitlePrefix: "ICML Paper Search MCP",
    homeDescription: "提供六个工具，用于论文检索、记录读取、元数据查询和搜索会话管理。可以使用公共服务，也可以在本地部署相同代码。",
    publicService: "连接公共服务",
    localDeployment: "本地部署",
    imageAlt: "ICML Paper Search MCP 复古科学期刊风格封面",
    features: [
      ["六个 MCP 工具", "用于检索论文、读取记录、查看元数据和管理搜索会话。"],
      ["ICML 2026 快照", "检索一份来源可追溯的 6,341 篇论文数据。"],
      ["开源", "使用 Milvus、SQLite 和 DashScope key 本地运行。"],
    ],
    docsFallback: "项目文档",
    llmsStart: "从这里开始",
    llmsOptional: "其他页面",
  },
} as const;

export function siteCopy(language: Language) {
  return copy[language];
}

export function absoluteUrl(pathname: string): string {
  const root = siteConfig.url.replace(/\/$/, "");
  return `${root}${pathname.startsWith("/") ? pathname : `/${pathname}`}`;
}
