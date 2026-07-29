import type { ReactNode } from "react";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import { requireLanguage } from "@/lib/i18n";
import { baseOptions } from "@/lib/layout.shared";
import { source } from "@/lib/source";

type LayoutProps = {
  children: ReactNode;
  params: Promise<{ lang: string }>;
};

export default async function Layout({ children, params }: LayoutProps) {
  const language = requireLanguage((await params).lang);
  return (
    <DocsLayout tree={source.getPageTree(language)} {...baseOptions(language)}>
      {children}
    </DocsLayout>
  );
}
