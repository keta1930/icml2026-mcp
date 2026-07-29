import { notFound } from "next/navigation";
import { getLLMText } from "@/lib/get-llm-text";
import { languages, requireLanguage } from "@/lib/i18n";
import { source } from "@/lib/source";

export const revalidate = false;

type RouteProps = {
  params: Promise<{ lang: string }>;
};

export function generateStaticParams() {
  return languages.map((lang) => ({ lang }));
}

export async function GET(_request: Request, { params }: RouteProps) {
  const language = requireLanguage((await params).lang);
  const page = source.getPage([], language);
  if (!page) notFound();

  return new Response(await getLLMText(page), {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}
