import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from "fumadocs-ui/page";
import { getMDXComponents } from "@/components/mdx";
import { languages, requireLanguage } from "@/lib/i18n";
import { markdownUrl } from "@/lib/markdown";
import { absoluteUrl, siteConfig, siteCopy } from "@/lib/site";
import { source } from "@/lib/source";

type PageProps = {
  params: Promise<{ lang: string; slug?: string[] }>;
};

function languageAlternates(slug: string[] | undefined) {
  const pages = Object.fromEntries(
    languages.map((language) => {
      const page = source.getPage(slug, language);
      if (!page) throw new Error(`Missing ${language} page for slug: ${(slug ?? []).join("/")}`);
      return [language === "zh" ? "zh-CN" : "en", absoluteUrl(page.url)];
    }),
  );
  return pages;
}

export default async function Page({ params }: PageProps) {
  const { lang, slug } = await params;
  const language = requireLanguage(lang);
  const page = source.getPage(slug, language);
  if (!page) notFound();

  const MDX = page.data.body;
  return (
    <DocsPage toc={page.data.toc} full={page.data.full}>
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription>{page.data.description}</DocsDescription>
      <div className="mt-4 border-b border-fd-border pb-4 text-sm">
        <a
          className="text-fd-muted-foreground hover:text-fd-foreground"
          href={`${siteConfig.basePath}${markdownUrl(page.url)}`}
        >
          {siteCopy(language).markdownAction}
        </a>
      </div>
      <DocsBody>
        <MDX components={getMDXComponents()} />
      </DocsBody>
    </DocsPage>
  );
}

export function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { lang, slug } = await params;
  const language = requireLanguage(lang);
  const page = source.getPage(slug, language);
  if (!page) notFound();
  return {
    title: page.data.title,
    description: page.data.description,
    alternates: {
      canonical: absoluteUrl(page.url),
      languages: languageAlternates(slug),
    },
  };
}
