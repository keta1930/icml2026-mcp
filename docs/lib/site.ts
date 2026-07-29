import type { Language } from "@/lib/i18n";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const siteConfig = {
  name: "ICML Paper Search MCP",
  publicMcpUrl: "https://icml.agent4sci.com:20441/mcp",
  url: process.env.NEXT_PUBLIC_SITE_URL ?? "http://127.0.0.1:20440",
  basePath,
};

const copy = {
  en: {
    description:
      "Local hybrid retrieval and persistent research sessions for ICML 2026 papers.",
    navDocs: "Documentation",
    navMcp: "Public MCP",
    markdownAction: "View clean Markdown",
    homeEyebrow: "6,341 ICML 2026 papers · MCP native",
    homeTitle: "Keep finding new answers",
    homeTitlePrefix: "Help agents navigate a sea of papers and",
    homeDescription:
      "Local hybrid retrieval, persistent research sessions, and bounded public access for multi-round research workflows.",
    publicService: "Connect to the public service",
    localDeployment: "Deploy locally",
    imageAlt: "Vintage scientific journal cover for ICML Paper Search MCP",
    features: [
      ["Hybrid retrieval", "Chroma dense retrieval fused with SQLite FTS5 BM25."],
      ["Persistent sessions", "Suppress papers already returned across query rounds."],
      ["Local-first", "Keep the corpus, index, and session state on local storage."],
    ],
    docsFallback: "Project documentation",
    llmsStart: "Start here",
    llmsOptional: "Optional",
  },
  zh: {
    description: "面向 ICML 2026 论文的本地混合检索与持久化调研会话。",
    navDocs: "文档",
    navMcp: "公共 MCP",
    markdownAction: "查看 clean Markdown",
    homeEyebrow: "6,341 篇 ICML 2026 论文 · MCP 原生",
    homeTitle: "持续找到新答案",
    homeTitlePrefix: "让 agent 从论文海洋中",
    homeDescription: "本地混合检索、持久化调研会话和有界公共服务，专为多轮研究工作流设计。",
    publicService: "连接公共服务",
    localDeployment: "本地部署",
    imageAlt: "ICML Paper Search MCP 复古科学期刊风格封面",
    features: [
      ["混合检索", "融合 Chroma dense retrieval 与 SQLite FTS5 BM25。"],
      ["持久会话", "跨多轮 query 自动隐藏已经返回的论文。"],
      ["本地优先", "语料、索引与 session state 全部本地持久化。"],
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
