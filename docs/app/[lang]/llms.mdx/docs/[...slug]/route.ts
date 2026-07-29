import { notFound } from "next/navigation";
import { getLLMText } from "@/lib/get-llm-text";
import { requireLanguage } from "@/lib/i18n";
import { source } from "@/lib/source";

export const revalidate = false;

type RouteProps = {
  params: Promise<{ lang: string; slug: string[] }>;
};

export async function GET(_request: Request, { params }: RouteProps) {
  const { lang, slug } = await params;
  const language = requireLanguage(lang);
  const page = source.getPage(slug, language);
  if (!page) notFound();

  return new Response(await getLLMText(page), {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}

export function generateStaticParams() {
  return source.generateParams().filter((params) => params.slug.length > 0);
}
