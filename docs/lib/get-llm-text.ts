import { source } from "@/lib/source";
import { siteConfig } from "@/lib/site";

export async function getLLMText(page: (typeof source)["$inferPage"]) {
  const processed = await page.data.getText("processed");
  const description = page.data.description
    ? `\n> ${page.data.description}\n`
    : "";

  const origin = siteConfig.url.replace(/\/$/, "");
  return `# ${page.data.title}\n${description}\nCanonical: ${origin}${page.url}\n\n${processed}`;
}
