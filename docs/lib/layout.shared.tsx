import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import type { Language } from "@/lib/i18n";
import { siteConfig, siteCopy } from "@/lib/site";

export function baseOptions(language: Language): BaseLayoutProps {
  const copy = siteCopy(language);
  return {
    nav: {
      title: siteConfig.name,
      url: `/${language}`,
    },
    links: [
      {
        text: copy.navDocs,
        url: `/${language}/docs`,
        active: "nested-url",
      },
      {
        text: copy.navMcp,
        url: siteConfig.publicMcpUrl,
        external: true,
      },
    ],
  };
}
