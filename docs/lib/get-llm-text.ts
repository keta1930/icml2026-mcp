import { requireLanguage } from "@/lib/i18n";
import { source } from "@/lib/source";
import { absoluteUrl } from "@/lib/site";

export async function getLLMText(page: (typeof source)["$inferPage"]) {
  const processed = (await page.data.getText("processed")).replace(
    /\]\((\/(?:en|zh)\/docs[^)]*)\)/g,
    (_match, target: string) => `](${absoluteUrl(target)})`,
  );
  const description = page.data.description
    ? `\n> ${page.data.description}\n`
    : "";

  const language = requireLanguage(page.locale ?? "");
  const canonicalLabel = language === "zh" ? "规范地址" : "Canonical";
  return `# ${page.data.title}\n${description}\n${canonicalLabel}: ${absoluteUrl(page.url)}\n\n${processed}`;
}
