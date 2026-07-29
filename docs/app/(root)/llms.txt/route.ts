import { absoluteUrl, siteConfig } from "@/lib/site";

export const revalidate = false;

export function GET() {
  const content = [
    `# ${siteConfig.name}`,
    "",
    "> AI-readable documentation indexes. English is the default language.",
    "",
    "## Languages",
    "",
    `- [English](${absoluteUrl("/en/llms.txt")}): English documentation index.`,
    `- [简体中文](${absoluteUrl("/zh/llms.txt")}): 中文文档索引。`,
    "",
  ].join("\n");

  return new Response(content, {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}
