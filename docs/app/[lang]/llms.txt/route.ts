import { markdownUrl } from "@/lib/markdown";
import { languages, requireLanguage } from "@/lib/i18n";
import { absoluteUrl, siteConfig, siteCopy } from "@/lib/site";
import { source } from "@/lib/source";

export const revalidate = false;

const essential = new Set([
  "",
  "getting-started/public-service",
  "getting-started/local-deployment",
  "concepts/sessions",
  "reference/mcp-tools",
]);

type RouteProps = {
  params: Promise<{ lang: string }>;
};

export function generateStaticParams() {
  return languages.map((lang) => ({ lang }));
}

export async function GET(_request: Request, { params }: RouteProps) {
  const language = requireLanguage((await params).lang);
  const copy = siteCopy(language);
  const pages = source.getPages(language);
  const line = (page: (typeof pages)[number]) => {
    const description = page.data.description ?? copy.docsFallback;
    return `- [${page.data.title}](${absoluteUrl(markdownUrl(page.url))}): ${description}`;
  };
  const primary = pages
    .filter((page) => essential.has(page.slugs.join("/")))
    .map(line);
  const optional = pages
    .filter((page) => !essential.has(page.slugs.join("/")))
    .map(line);
  const content = [
    `# ${siteConfig.name}`,
    "",
    `> ${copy.description}`,
    "",
    `## ${copy.llmsStart}`,
    "",
    ...primary,
    "",
    `## ${copy.llmsOptional}`,
    "",
    ...optional,
    "",
  ].join("\n");

  return new Response(content, {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}
