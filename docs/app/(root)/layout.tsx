import type { Metadata } from "next";
import type { ReactNode } from "react";
import { RootProvider } from "fumadocs-ui/provider/next";
import { absoluteUrl, siteConfig, siteCopy } from "@/lib/site";
import "../global.css";

const copy = siteCopy("en");

export const metadata: Metadata = {
  metadataBase: new URL(siteConfig.url),
  title: siteConfig.name,
  description: copy.description,
  alternates: {
    canonical: absoluteUrl("/"),
    languages: {
      en: absoluteUrl("/en/"),
      "zh-CN": absoluteUrl("/zh/"),
    },
  },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="flex min-h-screen flex-col antialiased">
        <RootProvider search={{ enabled: false }}>{children}</RootProvider>
      </body>
    </html>
  );
}
