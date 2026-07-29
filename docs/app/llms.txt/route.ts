import { markdownUrl } from "@/lib/markdown";
import { siteConfig } from "@/lib/site";
import { source } from "@/lib/source";

export const revalidate = false;

const essential = new Set([
  "/docs",
  "/docs/getting-started/public-service",
  "/docs/getting-started/local-deployment",
  "/docs/concepts/architecture",
  "/docs/reference/mcp-tools",
]);

export function GET() {
  const pages = source.getPages();
  const origin = siteConfig.url.replace(/\/$/, "");
  const line = (page: (typeof pages)[number]) => {
    const description = page.data.description ?? "项目文档";
    return `- [${page.data.title}](${origin}${markdownUrl(page.url)}): ${description}`;
  };
  const primary = pages.filter((page) => essential.has(page.url)).map(line);
  const optional = pages.filter((page) => !essential.has(page.url)).map(line);
  const content = [
    `# ${siteConfig.name}`,
    "",
    `> ${siteConfig.description}`,
    "",
    "## Start here",
    "",
    ...primary,
    "",
    "## Optional",
    "",
    ...optional,
    "",
  ].join("\n");

  return new Response(content, {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}
