import type { Metadata } from "next";
import type { ReactNode } from "react";
import { RootProvider } from "fumadocs-ui/provider/next";
import SearchDialog from "@/components/search";
import {
  htmlLanguage,
  languages,
  requireLanguage,
  uiI18n,
} from "@/lib/i18n";
import { absoluteUrl, siteConfig, siteCopy } from "@/lib/site";
import "../global.css";

type LayoutProps = {
  children: ReactNode;
  params: Promise<{ lang: string }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  return languages.map((lang) => ({ lang }));
}

export async function generateMetadata({ params }: LayoutProps): Promise<Metadata> {
  const language = requireLanguage((await params).lang);
  const copy = siteCopy(language);
  return {
    metadataBase: new URL(siteConfig.url),
    title: {
      default: siteConfig.name,
      template: `%s · ${siteConfig.name}`,
    },
    description: copy.description,
    alternates: {
      canonical: absoluteUrl(`/${language}/`),
      languages: {
        en: absoluteUrl("/en/"),
        "zh-CN": absoluteUrl("/zh/"),
      },
    },
  };
}

export default async function LocaleLayout({ children, params }: LayoutProps) {
  const language = requireLanguage((await params).lang);
  return (
    <html lang={htmlLanguage(language)} suppressHydrationWarning>
      <body className="flex min-h-screen flex-col antialiased">
        <RootProvider
          i18n={uiI18n.provider(language)}
          search={{ SearchDialog }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
