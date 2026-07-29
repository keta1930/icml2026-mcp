import type { ReactNode } from "react";
import { HomeLayout } from "fumadocs-ui/layouts/home";
import { requireLanguage } from "@/lib/i18n";
import { baseOptions } from "@/lib/layout.shared";

type LayoutProps = {
  children: ReactNode;
  params: Promise<{ lang: string }>;
};

export default async function Layout({ children, params }: LayoutProps) {
  const language = requireLanguage((await params).lang);
  return <HomeLayout {...baseOptions(language)}>{children}</HomeLayout>;
}
